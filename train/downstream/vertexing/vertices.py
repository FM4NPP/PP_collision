"""
vertices.py: VertexFinder, primary and secondary vertices from TrackBuilder's tracks.

PV: covariance-weighted adaptive vertex fit (ACTS's algorithm: annealed weights from each track's chi2), seeded by
the track density on the beam line; with max_vertices > 1 further (pileup) vertices on the tracks left.
SVs (find_secondary): displaced tracks are paired into seeds, each seed is fitted like the PV, and the PV is
refitted with SV tracks shared between vertices.
"""
import math

import torch
import torch.nn as nn

from .common import _arc_length, _circle_point

def _fit_vertex_by_closest_approach(
    track_position,
    track_direction,
    track_weight,
    track_sigma,
    minimum_total_weight,
    regularization=1e-9,
    minimum_eigenvalue_ratio=1e-4,
):
    """
    Covariance-weighted least-squares point closest to straight tracks, per event.

    Args:
        track_position, track_direction: (B, T, 3) point and unit direction.
        track_weight: (B, T), 0 = ignored.
        track_sigma: (B, T, 2) d0 and z0 errors (cm); d0-z0 correlation ignored.
        minimum_total_weight, minimum_eigenvalue_ratio, regularization: validity limits and solve regulariser.

    Returns:
        dict: vertex_estimate (B, 3), chi_square (B,), fit_is_valid (B,), track_closest_approach_point (B, T, 3),
        track_dca (B, T), track_chi2 (B, T) (r^T W r), vertex_covariance (B, 3, 3); NaN where not valid.
    """
    weight = track_weight.clamp(min=0.0)

    I3 = torch.eye(3, device=track_position.device, dtype=track_position.dtype)
    projector = I3 - track_direction.unsqueeze(-1) * track_direction.unsqueeze(-2)
    # W = U U^T / sigma_d0^2 + V V^T / (sigma_z0 sin theta)^2; W projector = W
    U, V, transverse = _track_frame(track_direction)
    P = (U.unsqueeze(-1) * U.unsqueeze(-2) / track_sigma[..., 0, None, None] ** 2
         + V.unsqueeze(-1) * V.unsqueeze(-2) / (track_sigma[..., 1] * transverse)[..., None, None] ** 2)

    normal_matrix = (weight[..., None, None] * P).sum(dim=1)
    weighted_position_sum = (weight[..., None] *
                             torch.matmul(P, track_position.unsqueeze(-1)).squeeze(-1)).sum(dim=1)

    # degenerate (fewer than 2 non-parallel tracks) if the eigenvalue ratio is tiny
    eigenvalues = torch.linalg.eigvalsh(normal_matrix)
    fit_is_valid = ((eigenvalues[:, 0] / eigenvalues[:, -1].clamp(min=regularization) > minimum_eigenvalue_ratio)
                    & (weight.sum(dim=-1) > minimum_total_weight))

    # invalid events solve against the identity instead; their results are set to NaN below
    regularized_normal_matrix = torch.where(fit_is_valid.view(-1, 1, 1), normal_matrix + regularization * I3, I3)
    PV = torch.linalg.solve(regularized_normal_matrix, weighted_position_sum)

    dca_vector = torch.matmul(projector, (track_position - PV.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
    track_dca = dca_vector.norm(dim=-1)
    # each track's unweighted chi2 to the vertex: ACTS's vertex compatibility r^T W r
    track_chi2 = _track_compatibility(PV, track_position, track_direction, track_sigma)
    chi_square = (weight * track_chi2).sum(dim=-1)
    # the vertex covariance: the inverse of the normal matrix
    vertex_covariance = torch.linalg.inv(regularized_normal_matrix)

    invalid = ~fit_is_valid
    return {
        "vertex_estimate": PV.masked_fill(invalid.unsqueeze(-1), float("nan")),
        "chi_square": chi_square,
        "fit_is_valid": fit_is_valid,
        "track_closest_approach_point": (PV.unsqueeze(1) + dca_vector).masked_fill(invalid[:, None, None], float("nan")),
        "track_dca": track_dca.masked_fill(invalid.unsqueeze(-1), float("nan")),
        "track_chi2": track_chi2.masked_fill(invalid.unsqueeze(-1), float("nan")),
        "vertex_covariance": vertex_covariance.masked_fill(invalid.view(-1, 1, 1), float("nan")),
    }

def _select(mask, new, old):
    """Per event (mask (B,)): new's values where mask is True, else old's, for each of old's keys."""
    return {k: torch.where(mask.view(-1, *[1] * (v.dim() - 1)), new[k], v) for k, v in old.items()}

def _track_frame(track_direction):
    """
    Each track's d0 and z0 directions perpendicular to it: U (transverse, in x-y) and V = direction x U (B, T, 3),
    and the transverse length of the direction (sin theta) (B, T).
    """
    transverse = track_direction[..., :2].norm(dim=-1).clamp(min=1e-12)
    U = torch.stack([-track_direction[..., 1], track_direction[..., 0], torch.zeros_like(transverse)], dim=-1) \
        / transverse.unsqueeze(-1)
    return U, torch.cross(track_direction, U, dim=-1), transverse

def _track_compatibility(point, track_position, track_direction, track_sigma):
    """
    Compatibility chi2 (B, T) of a point (B, 3) with each track, in the fit's metric.
    """
    r = point.unsqueeze(1) - track_position
    r = r - (r * track_direction).sum(dim=-1, keepdim=True) * track_direction
    U, V, transverse = _track_frame(track_direction)
    return ((r * U).sum(dim=-1) / track_sigma[..., 0]) ** 2 + ((r * V).sum(dim=-1) / (track_sigma[..., 1] * transverse)) ** 2

def _density_seed(d0, z0, track_sigma, track_weight, d0_significance, z0_significance):
    """
    z of the highest track density on the beam line, as ACTS's Gaussian track-density seeder (d0-z0 correlation
    ignored): each track with |d0| / sigma_d0 <= d0_significance adds weight x N(d0, z0 - z) within its
    z0_significance window; the maximum is searched from each track's z0 with two Gaussian-shaped Newton steps.

    Args:
        d0, z0: (B, T) perigee to the beam line, cm; track_sigma: (B, T, 2) their errors; track_weight: (B, T).

    Returns:
        z (B,), NaN where no track contributes.
    """
    sigma_d0, sigma_z0 = track_sigma[..., 0], track_sigma[..., 1]
    d0_chi2 = (d0 / sigma_d0) ** 2
    use = (track_weight > 0) & (d0_chi2 <= d0_significance ** 2) & torch.isfinite(z0)
    # log of each track's density on the beam line at its own z0, and its z window (ACTS: chi2 <= 4 z0_significance^2)
    log_peak = torch.where(use, torch.log(track_weight.clamp(min=1e-30)) - 0.5 * d0_chi2.nan_to_num()
                           - torch.log(2 * math.pi * sigma_d0 * sigma_z0), -torch.inf)
    half_window = sigma_z0 * (4 * z0_significance ** 2 - d0_chi2.nan_to_num()).clamp(min=0).sqrt()
    z0, sigma_z0 = z0.nan_to_num().unsqueeze(1), sigma_z0.unsqueeze(1)

    def density(z):
        """Density and its first two derivatives at candidates z (B, C)."""
        dz = z.unsqueeze(-1) - z0                                               # (B, C, T)
        delta = torch.where(dz.abs() < half_window.unsqueeze(1),
                            torch.exp(log_peak.unsqueeze(1) - 0.5 * (dz / sigma_z0) ** 2), 0.0)
        slope = -dz / sigma_z0 ** 2
        return delta.sum(-1), (delta * slope).sum(-1), (delta * (slope ** 2 - 1 / sigma_z0 ** 2)).sum(-1)

    # from each track's z0: a candidate counts while density > 0 and it is a maximum (second derivative < 0)
    z, ok = z0.squeeze(1), use
    candidates, values = [], []
    for step in range(3):
        y, dy, ddy = density(z)
        ok = ok & (y > 0) & (ddy < 0)
        candidates.append(z)
        values.append(torch.where(ok, y, -torch.inf))
        if step < 2:
            den = dy * dy - y * ddy
            z = z + torch.where(den != 0, y * dy / torch.where(den != 0, den, 1.0), 0.0).nan_to_num()
    values, candidates = torch.cat(values, dim=-1), torch.cat(candidates, dim=-1)
    best = values.argmax(dim=-1, keepdim=True)
    return torch.where(torch.isfinite(values.gather(-1, best)), candidates.gather(-1, best), torch.nan).squeeze(-1)
def _fit_vertex_by_helix_closest_approach(
    circle_centre,
    R_s,
    zslope,
    z0,
    track_weight,
    start,
    n_iterations,
    tolerance,
    beam_position,
    **linear_fit_kwargs,
):
    """
    Helix vertex fit: repeats the straight-line fit on each helix's tangent at its closest point to the vertex.

    Args:
        circle_centre (B, T, 2), R_s, zslope, z0 (B, T): finite helices (stand-ins with weight 0 for unfitted ones).
        track_weight: (B, T); start: (B, 3) first vertex.
        n_iterations, tolerance (cm): each event stops once its vertex moves less than tolerance.
        beam_position: (x, y), cm; **linear_fit_kwargs: for _fit_vertex_by_closest_approach.

    Returns:
        dict as _fit_vertex_by_closest_approach.
    """
    n_events = R_s.shape[0]
    PV = start.nan_to_num()
    still_moving = torch.ones(n_events, dtype=torch.bool, device=R_s.device)
    x0, y0 = circle_centre.unbind(-1)

    result = None
    for _ in range(n_iterations):
        # each circle's nearest point to the vertex and the helix tangent there
        pca_circle, t_xy = _circle_point(circle_centre, R_s, PV[:, None, :2])

        pca_z, dz = _arc_length(pca_circle[..., 0], pca_circle[..., 1], x0, y0, R_s, beam_position) * zslope + z0, zslope
        pca = torch.cat([pca_circle, pca_z.unsqueeze(-1)], dim=-1)
        tangent = torch.cat([t_xy, dz.unsqueeze(-1)], dim=-1)
        # clamped: a stand-in track (centre at the origin) with the vertex at the origin has no direction
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        step = _fit_vertex_by_closest_approach(pca, tangent, track_weight, **linear_fit_kwargs)
        # each event stops at its own convergence (independent of the rest of the batch)
        result = step if result is None else _select(still_moving, step, result)
        moved = (step["vertex_estimate"] - PV).norm(dim=-1) > tolerance
        # advance PV only where the fit is valid, so one bad event can't spread NaN
        PV = torch.where((still_moving & step["fit_is_valid"]).unsqueeze(-1), step["vertex_estimate"], PV)
        still_moving = still_moving & step["fit_is_valid"] & moved

        if not still_moving.any():
            break
    return result


class VertexFinder(nn.Module):
    """
    Primary and secondary vertices from TrackBuilder's tracks.

    Args:
        beam_position: (x, y), cm (as TrackBuilder's).
        helix_iterations (>= 1), convergence_tolerance (cm), minimum_total_weight: fit limits.
        annealing_temperatures, annealing_cut_off, min_track_weight, adaptive_max_iterations, max_relative_shift:
            the adaptive fit (ACTS's defaults).
        seed_d0_significance, seed_z0_significance: the cuts of the track-density seed on the beam line (ACTS's
            Gaussian seeder), with whose compatibility the adaptive fit's first weights start.
        seed_z_window, seed_z_window_softness (cm): the PV fit's track weights x sigmoid((window - |z0 - z_seed|) /
            softness), a soft form of ACTS AMVF's 1 mm z preselection.
        max_vertices: primary vertices to look for (pileup; 1: the PV only); each further one is seeded and fitted on
            the tracks the earlier ones left, kept with >= min_vertex_tracks tracks of fit weight >= 0.5 and unless
            within max_merge_significance sigma in z of an earlier one; the PV is the one with the highest sum of fit
            weight x pT^2.
        find_secondary: find SVs; refit_pv: report the PV refitted with SV tracks shared.
        sv_min_track_pv_chi2, sv_min_track_beam_significance: SV candidate cuts.
        max_pair_doca (cm): seed pairs; sv_max_track_chi2: tracks attached to a seed.
        min_sv_tracks, max_sv_chi2_per_dof, sv_min_displacement_significance: SV cuts.
        seed_merge_distance (cm): one SV per group closer than this; max_sv_per_event: K.
    """
    def __init__(
        self,
        beam_position: tuple = (0.0, 0.0),
        helix_iterations: int = 100,
        convergence_tolerance: float = 1e-4,
        minimum_total_weight: float = 1e-6,
        annealing_temperatures: tuple = (64.0, 16.0, 4.0, 2.0, 1.5, 1.0),
        annealing_cut_off: float = 9.0,
        min_track_weight: float = 1e-4,
        adaptive_max_iterations: int = 30,
        max_relative_shift: float = 0.01,
        seed_d0_significance: float = 3.5,
        seed_z0_significance: float = 12.0,
        seed_z_window: float = 0.2,
        seed_z_window_softness: float = 0.05,
        max_vertices: int = 1,
        min_vertex_tracks: int = 2,
        max_merge_significance: float = 3.0,
        find_secondary: bool = True,
        sv_min_track_pv_chi2: float = 9.0,
        sv_min_track_beam_significance: float = 0.0,
        sv_max_track_chi2: float = 9.0,
        sv_min_displacement_significance: float = 3.0,
        max_pair_doca: float = 1.0,
        seed_merge_distance: float = 0.1,
        min_sv_tracks: int = 2,
        max_sv_chi2_per_dof: float = 10.0,
        max_sv_per_event: int = 6,
        refit_pv: bool = True,
    ):
        super().__init__()
        if adaptive_max_iterations < 1 or max_vertices < 1 or helix_iterations < 1:
            raise ValueError("adaptive_max_iterations, max_vertices and helix_iterations must be at least 1")
        self.beam_position = tuple(beam_position)
        self.helix_iterations = helix_iterations
        self.convergence_tolerance = convergence_tolerance
        self.minimum_total_weight = minimum_total_weight
        self.annealing_temperatures = tuple(float(t) for t in annealing_temperatures)
        self.annealing_cut_off = annealing_cut_off
        self.min_track_weight = min_track_weight
        self.adaptive_max_iterations = adaptive_max_iterations
        self.max_relative_shift = max_relative_shift
        self.seed_d0_significance = seed_d0_significance
        self.seed_z0_significance = seed_z0_significance
        self.seed_z_window = seed_z_window
        self.seed_z_window_softness = seed_z_window_softness
        self.max_vertices = max_vertices
        self.min_vertex_tracks = min_vertex_tracks
        self.max_merge_significance = max_merge_significance
        self.find_secondary = find_secondary
        self.sv_min_track_pv_chi2 = sv_min_track_pv_chi2
        self.sv_min_track_beam_significance = sv_min_track_beam_significance
        self.sv_max_track_chi2 = sv_max_track_chi2
        self.sv_min_displacement_significance = sv_min_displacement_significance
        self.max_pair_doca = max_pair_doca
        self.seed_merge_distance = seed_merge_distance
        self.min_sv_tracks = min_sv_tracks
        self.max_sv_chi2_per_dof = max_sv_chi2_per_dof
        self.max_sv_per_event = max_sv_per_event
        self.refit_pv = refit_pv

    def _fit(self, track_position, track_direction, track_weight, params, start=None, single_step=False):
        """
        Vertex fit with these weights: straight-line fit, then helix iterations (or one helix update from start
        with single_step). start: (B, 3), NaN rows / None = straight-line vertex.
        """
        kwargs = dict(minimum_total_weight=self.minimum_total_weight, track_sigma=params["track_sigma"])
        linear = None if single_step else _fit_vertex_by_closest_approach(
            track_position, track_direction, track_weight, **kwargs)
        begin = start if single_step else (linear["vertex_estimate"] if start is None else
                                           torch.where(torch.isfinite(start), start, linear["vertex_estimate"]))
        fit = _fit_vertex_by_helix_closest_approach(
            *params["helix"], track_weight, begin, n_iterations=1 if single_step else self.helix_iterations,
            tolerance=self.convergence_tolerance, beam_position=self.beam_position, **kwargs)
        if linear is not None:
            keep_linear = ~fit["fit_is_valid"]
            fit = _select(keep_linear, linear, fit)
        return fit

    def _fit_adaptive(self, track_position, track_direction, track_weight, params, competing_chi2=None, start=None,
                      seed=None):
        """
        ACTS's adaptive (annealed) fit of one vertex. competing_chi2: (B, T, K) chi2 to other vertices (inf if none),
        for shared track weights; seed: (B, 3) the first weights come from compatibility with it (NaN rows: with
        the all-track fit). Returns the fit and each track's weight (B, T).
        """
        fit = self._fit(track_position, track_direction, track_weight, params, start=start)
        if seed is not None:
            seeded = torch.isfinite(seed).all(dim=-1, keepdim=True)
            fit = {**fit, "vertex_estimate": torch.where(seeded, seed, fit["vertex_estimate"]),
                   "track_chi2": torch.where(seeded, _track_compatibility(seed.nan_to_num(), track_position,
                                                                          track_direction, params["track_sigma"]),
                                             fit["track_chi2"])}
        n_temperatures = len(self.annealing_temperatures)
        done, previous_weight = torch.zeros_like(fit["fit_is_valid"]), torch.zeros_like(track_weight)    # finished events keep their fit
        for i in range(self.adaptive_max_iterations):
            # one annealing step per update, then the last temperature
            t2 = 2.0 * self.annealing_temperatures[min(i, n_temperatures - 1)]
            # AnnealingUtility weight 1 / (1 + exp((chi2 - cutOff)/2T) + sum_k exp((chi2 - chi2_k)/2T)), as
            # exp(-logsumexp(0, (chi2 - cutOff)/2T, (chi2 - chi2_k)/2T...)): no overflow for far tracks
            chi2 = fit["track_chi2"].unsqueeze(-1)
            terms = [torch.zeros_like(chi2), (chi2 - self.annealing_cut_off) / t2]
            terms += [] if competing_chi2 is None else [(chi2 - competing_chi2) / t2]
            vertex_weight = torch.exp(-torch.logsumexp(torch.cat(terms, dim=-1), dim=-1)).nan_to_num(0.0)
            vertex_weight = vertex_weight.masked_fill(vertex_weight <= self.min_track_weight, 0.0).to(track_weight.dtype)
            vertex_weight = previous_weight = torch.where(done.unsqueeze(-1), previous_weight, vertex_weight)
            new = self._fit(track_position, track_direction, track_weight * vertex_weight, params,
                            start=fit["vertex_estimate"], single_step=True)
            valid = new["fit_is_valid"]
            dx = (new["vertex_estimate"] - fit["vertex_estimate"]).nan_to_num().unsqueeze(-1)
            # a singular covariance (rare) gives no shift measure: not converged (max iterations still apply)
            shift, singular = torch.linalg.solve_ex(new["vertex_covariance"].nan_to_num(nan=0.0)
                                                    + (~valid).view(-1, 1, 1) * torch.eye(3, dtype=dx.dtype, device=dx.device), dx)
            relative_shift = torch.where(singular == 0, (dx * shift).sum(dim=(-2, -1)), torch.inf)
            update = valid & ~done
            fit = _select(update, new, fit)
            # at the last temperature each event stops once its shift is small
            if i >= n_temperatures - 1:
                done = done | (relative_shift <= self.max_relative_shift) | ~valid
                if bool(done.all()):
                    break
        return fit, vertex_weight

    def _find_secondary_vertices(self, pv_fit, track_position, track_direction, track_weight, params, pv_weight):
        """
        SV finding after the PV fit: the sv_* outputs, displaced_mask, displaced_weight and, with an SV, pv_refit
        (with pv_weight, the PV fit's weights; default track_weight).
        """
        n_events, n_tracks = track_weight.shape
        K = self.max_sv_per_event
        device, dtype = track_position.device, track_position.dtype
        pv_position = pv_fit["vertex_estimate"].nan_to_num()

        # candidates: incompatible with the PV (and displaced from the beam line), in units of their errors
        track_sigma = params["track_sigma"]
        displaced_mask = ((pv_fit["track_chi2"].nan_to_num() > self.sv_min_track_pv_chi2)
                          & (params["beam_dca"].nan_to_num() / track_sigma[..., 0] > self.sv_min_track_beam_significance)
                          & params["usable"])
        displaced_weight = track_weight * displaced_mask.to(track_weight.dtype)

        # seeds: closest-approach midpoints of candidate pairs with DOCA < max_pair_doca, tightest first
        P1, P2 = track_position.unsqueeze(2), track_position.unsqueeze(1)
        d1, d2 = track_direction.unsqueeze(2), track_direction.unsqueeze(1)
        dp = P2 - P1
        c = (d1 * d2).sum(dim=-1)
        dp_d1, dp_d2 = (dp * d1).sum(dim=-1), (dp * d2).sum(dim=-1)
        denom = (1.0 - c ** 2).clamp(min=1e-12)
        cpa1 = P1 + ((dp_d1 - c * dp_d2) / denom).unsqueeze(-1) * d1
        cpa2 = P2 + ((c * dp_d1 - dp_d2) / denom).unsqueeze(-1) * d2
        doca = (cpa2 - cpa1).norm(dim=-1)
        flat_valid = (displaced_mask.unsqueeze(2) & displaced_mask.unsqueeze(1)
                      & torch.triu(torch.ones(n_tracks, n_tracks, device=device, dtype=torch.bool), diagonal=1)
                      & (doca < self.max_pair_doca)).view(n_events, -1)
        top_idx = doca.view(n_events, -1).masked_fill(~flat_valid, float("inf")).argsort(dim=-1)[:, :K]
        seeds = ((cpa1 + cpa2) * 0.5).view(n_events, -1, 3).gather(1, top_idx.unsqueeze(-1).expand(-1, -1, 3))
        seed_valid = flat_valid.gather(1, top_idx)
        # fewer pairs than slots (n_tracks^2 < K): pad with empty slots
        n_missing = K - top_idx.shape[1]
        if n_missing > 0:
            seeds = torch.cat([seeds, seeds.new_zeros(n_events, n_missing, 3)], dim=1)
            seed_valid = torch.cat([seed_valid, seed_valid.new_zeros(n_events, n_missing)], dim=1)
        seeds = seeds.masked_fill(~seed_valid.unsqueeze(-1), float("nan"))

        sv_estimate = torch.full((n_events, K, 3), float("nan"), device=device, dtype=dtype)
        sv_chi2 = torch.full((n_events, K), float("nan"), device=device, dtype=dtype)
        sv_chi2_per_dof = torch.full((n_events, K), float("nan"), device=device, dtype=dtype)
        sv_is_valid = torch.zeros((n_events, K), device=device, dtype=torch.bool)
        sv_n_tracks = torch.zeros((n_events, K), device=device, dtype=torch.long)
        sv_displacement = torch.full((n_events, K), float("nan"), device=device, dtype=dtype)
        sv_track_mask = torch.zeros((n_events, K, n_tracks), device=device, dtype=torch.bool)
        sv_track_dca = torch.full((n_events, K, n_tracks), float("nan"), device=device, dtype=dtype)
        sv_track_chi2 = torch.full((n_events, K, n_tracks), float("nan"), device=device, dtype=dtype)

        # every seeded (event, slot) pair is one row of a single fit
        event, slot = seed_valid.nonzero(as_tuple=True)
        if len(event):
            # candidates compatible with the seed (chi2, the fit's metric)
            in_sv = ((_track_compatibility(seeds[event, slot], track_position[event], track_direction[event],
                                           track_sigma[event]) <= self.sv_max_track_chi2)
                     & (displaced_weight[event] > 0))
            sv_weight = displaced_weight[event] * in_sv.to(displaced_weight.dtype)
            fit = self._fit(track_position[event], track_direction[event], sv_weight,
                            {"track_sigma": track_sigma[event], "helix": tuple(t[event] for t in params["helix"])})

            n_used = sv_weight.gt(0).sum(dim=-1)
            ndf = (2 * n_used.clamp(min=1) - 3).clamp(min=1).to(fit["chi_square"].dtype)
            chi2_per_dof = (fit["chi_square"] / ndf).masked_fill(~fit["fit_is_valid"], float("nan"))
            displacement = (fit["vertex_estimate"] - pv_position[event]).norm(dim=-1)

            # significance of the SV-PV distance, with both vertex covariances
            d = (fit["vertex_estimate"] - pv_position[event]).nan_to_num().unsqueeze(-1)
            # (an invalid PV fit has no covariance: the SV's alone; a singular sum, rare: not far enough)
            both = (fit["vertex_covariance"].nan_to_num(nan=0.0) + pv_fit["vertex_covariance"][event].nan_to_num(nan=0.0)
                    + (~fit["fit_is_valid"]).view(-1, 1, 1) * torch.eye(3, dtype=dtype, device=device))
            solved, singular = torch.linalg.solve_ex(both, d)
            far_enough = (singular == 0) & ((d * solved).sum(dim=(-2, -1)) >= self.sv_min_displacement_significance ** 2)
            is_ok = (fit["fit_is_valid"] & (n_used >= self.min_sv_tracks) & far_enough
                     & (chi2_per_dof.nan_to_num(nan=float("inf")) <= self.max_sv_chi2_per_dof))

            sv_estimate[event, slot] = fit["vertex_estimate"].masked_fill(~is_ok.unsqueeze(-1), float("nan"))
            sv_chi2[event, slot] = fit["chi_square"].masked_fill(~is_ok, float("nan"))
            sv_chi2_per_dof[event, slot] = chi2_per_dof.masked_fill(~is_ok, float("nan"))
            sv_is_valid[event, slot] = is_ok
            sv_n_tracks[event, slot] = n_used.masked_fill(~is_ok, 0)
            sv_displacement[event, slot] = displacement.masked_fill(~is_ok, float("nan"))
            sv_track_mask[event, slot] = in_sv & is_ok.unsqueeze(-1)
            sv_track_dca[event, slot] = fit["track_dca"].masked_fill(~is_ok.unsqueeze(-1), float("nan"))
            sv_track_chi2[event, slot] = fit["track_chi2"].masked_fill(~is_ok.unsqueeze(-1), float("nan"))

        # one SV per group within seed_merge_distance: the lowest chi2/ndf (ties: the lowest slot)
        dist = (sv_estimate.unsqueeze(2) - sv_estimate.unsqueeze(1)).nan_to_num().norm(dim=-1)
        q = sv_chi2_per_dof.nan_to_num(nan=float("inf"))
        q_i, q_j = q.unsqueeze(2), q.unsqueeze(1)                     # [b, i, j]: does SV j beat SV i?
        lower_slot = torch.ones(K, K, device=device, dtype=torch.bool).tril(diagonal=-1)   # j < i
        sv_is_valid = sv_is_valid & ~(sv_is_valid.unsqueeze(1) & (dist < self.seed_merge_distance)
                                      & ((q_j < q_i) | ((q_j == q_i) & lower_slot))).any(dim=2)
        result = {
            "sv_estimate": sv_estimate.masked_fill(~sv_is_valid.unsqueeze(-1), float("nan")),
            "sv_is_valid": sv_is_valid,
            "sv_chi2": sv_chi2.masked_fill(~sv_is_valid, float("nan")),
            "sv_chi2_per_dof": sv_chi2_per_dof.masked_fill(~sv_is_valid, float("nan")),
            "sv_displacement": sv_displacement.masked_fill(~sv_is_valid, float("nan")),
            "sv_n_tracks": sv_n_tracks.masked_fill(~sv_is_valid, 0),
            "sv_track_mask": sv_track_mask & sv_is_valid.unsqueeze(-1),
            "sv_track_dca": sv_track_dca.masked_fill(~sv_is_valid.unsqueeze(-1), float("nan")),
            "displaced_mask": displaced_mask,
            "displaced_weight": displaced_weight,
        }
        if self.refit_pv and result["sv_is_valid"].any():     # without an SV the refit is never reported
            # multi-vertex weights: SV tracks stay in the PV fit, shared with their SVs
            competing = torch.where(result["sv_track_mask"], sv_track_chi2.nan_to_num(nan=float("inf")),
                                    float("inf")).transpose(1, 2)
            result["pv_refit"], result["pv_refit_vertex_weight"] = self._fit_adaptive(
                track_position, track_direction, pv_weight, params,
                competing_chi2=competing, start=pv_fit["vertex_estimate"])
        return result

    def forward(self, tracks):
        """
        tracks: TrackBuilder's output. Returns the vertex outputs of VertexHead.forward.
        """
        track_position, track_direction, track_weight = (tracks[k] for k in ("track_position", "track_direction",
                                                                             "track_weight"))
        bad, dtype = tracks["_bad"], torch.float64
        d0, z0 = tracks["track_parameters"][..., 0], tracks["track_parameters"][..., 1]
        params = {"pT": tracks["_pT"], "beam_dca": tracks["_beam_dca"], "usable": tracks["track_quality_mask"],
                  "helix": tracks["_helix"], "track_sigma": tracks["track_sigma"]}

        # primary vertices, as ACTS AMVF iterates: seed, z window and adaptive fit on the tracks not yet used, up to
        # max_vertices; a later vertex needs min_vertex_tracks tracks of fit weight >= 0.5, and the tracks it uses
        # leave the pool (weight x (1 - fit weight))
        candidates, available = [], track_weight
        for k in range(self.max_vertices):
            seed, window = self._seed_and_window(d0, z0, params["track_sigma"], available)
            vertex_fit, annealing = self._fit_adaptive(track_position, track_direction, available * window, params,
                                                       seed=seed)
            fit_weight = (annealing * window).masked_fill(bad | (track_weight <= 0), 0.0)
            valid = vertex_fit["fit_is_valid"]
            if k > 0:
                valid = valid & ((fit_weight >= 0.5).sum(dim=-1) >= self.min_vertex_tracks)
                valid = valid & torch.isfinite(seed).all(dim=-1)
                # a duplicate of an earlier vertex (z within max_merge_significance sigma, ACTS) is not a new one
                for earlier in candidates:
                    dz = vertex_fit["vertex_estimate"][:, 2] - earlier["fit"]["vertex_estimate"][:, 2]
                    sigma = (vertex_fit["vertex_covariance"][:, 2, 2] + earlier["fit"]["vertex_covariance"][:, 2, 2]).sqrt()
                    valid = valid & ~(earlier["valid"] & (dz.abs() < self.max_merge_significance * sigma))
            candidates.append({"fit": vertex_fit, "annealing": annealing, "window": window,
                               "fit_input": available * window, "fit_weight": fit_weight, "valid": valid})
            # the candidate's tracks leave the pool whether or not it is kept, so its seed is not found again
            used = vertex_fit["fit_is_valid"].unsqueeze(-1).to(fit_weight.dtype)
            available = available * (1 - fit_weight * used).clamp(min=0.0)
        # the PV: the valid vertex with the highest sum of fit weight x pT^2 (the first without a valid one)
        pT2 = params["pT"].nan_to_num(0.0).to(dtype) ** 2
        pv_index = torch.stack([(c["fit_weight"] * pT2).sum(dim=-1).masked_fill(~c["valid"], -torch.inf) for c in candidates],
                               dim=1).argmax(dim=1)
        pick = lambda values: torch.stack(values, dim=1)[torch.arange(len(pv_index), device=pv_index.device), pv_index]  # noqa: E731
        fit = {key: pick([c["fit"][key] for c in candidates]) for key in candidates[0]["fit"]}
        vertex_weight, window, pv_weight = (pick([c[k] for c in candidates]) for k in ("annealing", "window", "fit_input"))
        # SV finding keeps track_weight, minus the tracks of the other primary vertices
        others = torch.stack([c["fit_weight"] * c["valid"].unsqueeze(-1) for c in candidates], dim=1).masked_fill(
            torch.nn.functional.one_hot(pv_index, len(candidates)).bool().unsqueeze(-1), 0.0)
        sv_weight = track_weight * (1 - others.max(dim=1).values)

        result = {}
        if self.find_secondary:
            secondary = self._find_secondary_vertices(fit, track_position, track_direction, sv_weight, params, pv_weight)
            refit = secondary.pop("pv_refit", None)
            refit_vertex_weight = secondary.pop("pv_refit_vertex_weight", None)
            secondary["sv_track_dca"] = secondary["sv_track_dca"].masked_fill(bad.unsqueeze(1), float("nan"))
            result.update(secondary)
            # the refit PV where the event has an SV and the refit is valid; else the original PV fit
            if refit is not None:
                used = secondary["sv_is_valid"].any(dim=1) & refit["fit_is_valid"]
                fit = _select(used, refit, fit)
                vertex_weight = torch.where(used.unsqueeze(-1), refit_vertex_weight, vertex_weight)
        # the PV fit's outputs; per-track ones of tracks without a fit (bad) come from finite stand-ins: NaN
        for key in ("vertex_estimate", "chi_square", "fit_is_valid", "vertex_covariance", "track_chi2", "track_dca",
                    "track_closest_approach_point"):
            value = fit[key]
            result[key] = value.masked_fill(bad.view(*bad.shape, *[1] * (value.dim() - 2)), float("nan")) \
                if key.startswith("track_") else value
        # vertex quality, as ACTS judges it: each fitted track's weight in the fit, its annealing weight x the z window
        # (tracks with track_weight > 0)
        weight = (vertex_weight * window).masked_fill(bad | (track_weight <= 0), 0.0)
        result.update({"track_vertex_weight": vertex_weight.masked_fill(bad, 0.0), "vertex_n_tracks": (weight >= 0.5).sum(dim=-1),
                       "vertex_ndf": 2 * weight.sum(dim=-1) - 3})
        if len(candidates) > 1:
            # all primary vertices (pileup), each track's (fit weight >= 0.5, else -1) and which one is the PV
            valid = torch.stack([c["valid"] for c in candidates], dim=1)
            fit_weight = torch.stack([c["fit_weight"] for c in candidates], dim=1) * valid.unsqueeze(-1).to(dtype)
            best, index = fit_weight.max(dim=1)
            result.update({
                "primary_vertices": torch.stack([c["fit"]["vertex_estimate"] for c in candidates], dim=1).masked_fill(
                    ~valid.unsqueeze(-1), float("nan")),
                "primary_vertex_valid": valid,
                "primary_vertex_n_tracks": (fit_weight >= 0.5).sum(dim=-1).masked_fill(~valid, 0),
                "pv_index": pv_index,
                "track_primary_vertex": torch.where(best >= 0.5, index, -1),
            })
        return result
    def _seed_and_window(self, d0, z0, track_sigma, track_weight):
        """
        The density seed (B, 3) on the beam line (NaN rows without one) and each track's z-window factor (B, T)
        (1 without a seed).
        """
        dtype = track_weight.dtype
        z = _density_seed(d0, z0, track_sigma, track_weight, self.seed_d0_significance, self.seed_z0_significance).to(dtype)
        seed = torch.stack([torch.full_like(z, self.beam_position[0]), torch.full_like(z, self.beam_position[1]), z], -1)
        # NaN z0 (no fit) -> infinitely far (window 0); NaN seeds take window 1 below
        distance = (z0.to(dtype).nan_to_num(torch.inf) - seed[:, 2:].nan_to_num()).abs()
        fade = torch.sigmoid((self.seed_z_window - distance) / self.seed_z_window_softness)
        return seed, torch.where(torch.isfinite(seed[:, 2:]), fade.nan_to_num(0.0), 1.0)
