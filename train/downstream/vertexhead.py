"""
vertexhead.py: primary- and secondary-vertex fit from track-finder output.

Track parameters come from MomentumHead (momentumhead.py). The PV is the weighted least-squares
point of closest approach to the usable tracks (each anchored at its innermost hit, weight = soft
hit count x P(real)), refined by an iterative helix fit when use_helix and Bz != 0.

Secondary vertices (find_secondary=True, the default):
    1. candidates: DCA to the PV > min_track_pv_dca and to the beam line > min_track_beam_dca
    2. seeds: midpoints of closest approach of candidate pairs with DOCA < max_pair_doca
    3. each seed is fitted (same fit as the PV) from the candidates within max_track_sv_doca of it
    4. kept if valid, >= min_sv_tracks, >= min_sv_displacement from the PV, chi2/ndf <= max
    5. SVs within seed_merge_distance of a lower chi2/ndf SV are dropped
    6. the PV is refitted without SV tracks and reported where the event has an SV and the refit
       is valid; elsewhere the original fit (pv_original) is reported

Inputs (MambaAttentionHead.forward, TPCBatchDataset): class_probs (B, T, 2), mask_probs (B, N, T),
normalized points (B, N, 4), padding_mask (B, N). B events, N hits, T track slots,
K = max_sv_per_event SV slots. Positions in cm.
"""
import torch
import torch.nn as nn

from momentumhead import MomentumHead

def _fit_vertex_by_closest_approach(
    track_position,
    track_direction,
    track_weight,
    minimum_total_weight=1e-6,
    regularization=1e-9,
    minimum_eigenvalue_ratio=1e-4,
):
    """
    Weighted least-squares point of closest approach to straight tracks, per event.

    Args:
        track_position, track_direction: (B, T, 3) a point on each line and its unit direction.
        track_weight: (B, T) non-negative; 0 = ignored.
        minimum_total_weight, minimum_eigenvalue_ratio: limits below which an event is not valid.
        regularization: added to the normal matrix's diagonal.

    Returns:
        dict:
            vertex_estimate (B, 3); chi_square (B,), the weighted sum of squared DCAs.
            fit_is_valid (B,): False without 2 non-parallel weighted tracks.
            track_closest_approach_point (B, T, 3), track_dca (B, T).
            NaN where not valid (except chi_square).
    """
    weight = track_weight.clamp(min=0.0)

    I3 = torch.eye(3, device=track_position.device, dtype=track_position.dtype)
    P = I3 - track_direction.unsqueeze(-1) * track_direction.unsqueeze(-2)

    normal_matrix = (weight[..., None, None] * P).sum(dim=1)
    weighted_position_sum = (weight[..., None] *
                             torch.matmul(P, track_position.unsqueeze(-1)).squeeze(-1)).sum(dim=1)

    # degeneracy check before solving: smallest/largest eigenvalue ratio (0 tracks: all ~0;
    # 1 track or parallel tracks: one ~0). A rank-deficient matrix can make the solve raise.
    normal_matrix_eigenvalues = torch.linalg.eigvalsh(normal_matrix)
    eigenvalue_ratio = normal_matrix_eigenvalues[:, 0] / normal_matrix_eigenvalues[:, -1].clamp(min=regularization)
    fit_is_valid = ((eigenvalue_ratio > minimum_eigenvalue_ratio) &
                     (weight.sum(dim=-1) > minimum_total_weight))

    # invalid events solve against the identity instead; their results are set to NaN below
    regularized_normal_matrix = torch.where(fit_is_valid.view(-1, 1, 1),
                                            normal_matrix + regularization * I3,
                                            I3.expand_as(normal_matrix))
    PV = torch.linalg.solve(regularized_normal_matrix, weighted_position_sum)

    dca_vector = torch.matmul(P, (track_position - PV.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
    track_dca = dca_vector.norm(dim=-1)
    chi_square = (weight * track_dca ** 2).sum(dim=-1)

    track_closest_approach_point = PV.unsqueeze(1) + dca_vector

    PV = PV.masked_fill(~fit_is_valid.unsqueeze(-1), float("nan"))
    track_closest_approach_point = track_closest_approach_point.masked_fill(~fit_is_valid[:, None, None], float("nan"))
    track_dca = track_dca.masked_fill(~fit_is_valid.unsqueeze(-1), float("nan"))

    return {
        "vertex_estimate": PV,
        "chi_square": chi_square,
        "fit_is_valid": fit_is_valid,
        "track_closest_approach_point": track_closest_approach_point,
        "track_dca": track_dca,
    }

def _fit_vertex_by_helix_closest_approach(
    circle_centre,
    R_s,
    zslope,
    z0,
    track_weight,
    n_iterations=100,
    linear_fit=None,
    d_angle=0.005,
    is_cosmics = False,
    **linear_fit_kwargs,
):
    """
    Helix-aware vertex fit: repeats the straight-line fit on each track's tangent at its point of
    closest approach to the current vertex.

    Args:
        circle_centre (B, T, 2), R_s (B, T): MomentumHead's circle; sign(R_s) = sign(q Bz).
        zslope, z0: (B, T) z line z = zslope * r + z0 (r -> x if is_cosmics).
        track_weight: (B, T); tracks with NaN parameters are excluded.
        n_iterations: maximum; stops once every vertex moves < 1e-4; below 1 returns linear_fit.
        linear_fit: straight-line fit result whose vertex is the start (origin if None or NaN).
        d_angle: step along the circle (rad) that forms the tangent.
        **linear_fit_kwargs: passed to _fit_vertex_by_closest_approach.

    Returns:
        dict as _fit_vertex_by_closest_approach, from the last iteration; excluded tracks get NaN.
    """
    n_events = R_s.shape[0]

    if n_iterations < 1:
        return linear_fit

    PV = linear_fit["vertex_estimate"].nan_to_num() if linear_fit is not None else torch.zeros(n_events, 3, dtype=R_s.dtype, device=R_s.device)

    still_moving = torch.ones(n_events, dtype=torch.bool, device=R_s.device)

    # tracks without a fitted circle or z line get finite stand-ins, so they can't put NaN into the
    # solve, and zero weight, so they don't pull; their outputs are set to NaN after the loop
    failed = ~(torch.isfinite(circle_centre).all(dim=-1) & torch.isfinite(R_s) & torch.isfinite(zslope) & torch.isfinite(z0))
    circle_centre = circle_centre.masked_fill(failed.unsqueeze(-1), 0.0)
    R_s, zslope, z0 = R_s.masked_fill(failed, 1.0), zslope.masked_fill(failed, 0.0), z0.masked_fill(failed, 0.0)
    track_weight = track_weight.masked_fill(failed, 0.0)

    radius = R_s.abs()
    x0, y0 = circle_centre.unbind(-1)

    for _ in range(n_iterations):
        diff = PV[:, None, :2] - circle_centre
        pca_circle = (circle_centre + radius.unsqueeze(-1) * diff / 
                      diff.norm(dim=-1, keepdim=True).clamp(min=1e-12))

        angle_pca = torch.atan2(diff[..., 1], diff[..., 0])

        if is_cosmics:
            pca_z = pca_circle[..., 0] * zslope + z0
        else:
            pca_z = pca_circle.norm(dim=-1) * zslope + z0
        pca = torch.cat([pca_circle, pca_z.unsqueeze(-1)], dim=-1)

        # second point a small step along the circle gives the local tangent
        new_angle = angle_pca - torch.sign(R_s) * d_angle
        newx = radius * torch.cos(new_angle) + x0
        newy = radius * torch.sin(new_angle) + y0
        new_xy = torch.stack([newx, newy], dim=-1)
        
        if is_cosmics:
            new_z = newx * zslope + z0
        else:
            new_z = new_xy.norm(dim=-1) * zslope + z0

        second_point_pca = torch.cat([new_xy, new_z.unsqueeze(-1)], dim=-1)

        raw_tangent = second_point_pca - pca
        tangent = raw_tangent / raw_tangent.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        result = _fit_vertex_by_closest_approach(pca, tangent, track_weight, **linear_fit_kwargs)

        still_moving = (still_moving & result["fit_is_valid"] &
                         ((result["vertex_estimate"] - PV).norm(dim=-1) > 1e-4))
        
        # advance PV only where the fit is valid, so one bad event can't spread NaN
        PV = torch.where(result["fit_is_valid"].unsqueeze(-1), result["vertex_estimate"], PV)
       
        if not still_moving.any():
            break

    result["track_dca"] = result["track_dca"].masked_fill(failed, float("nan"))
    result["track_closest_approach_point"] = result["track_closest_approach_point"].masked_fill(failed.unsqueeze(-1), float("nan"))
    return result

def _pairwise_doca_and_midpoint(track_position, track_direction, eps=1e-12):
    """
    All-pairs straight-line DOCA per event; nearly parallel pairs (1 - c^2 < eps) get a finite but
    meaningless value.

    Returns:
        doca (B, T, T) and the closest-approach midpoint (B, T, T, 3).
    """
    P1, P2 = track_position.unsqueeze(2), track_position.unsqueeze(1)
    d1, d2 = track_direction.unsqueeze(2), track_direction.unsqueeze(1)

    dp = P2 - P1
    c = (d1 * d2).sum(dim=-1)
    dp_d1 = (dp * d1).sum(dim=-1)
    dp_d2 = (dp * d2).sum(dim=-1)

    denom = (1.0 - c ** 2).clamp(min=eps)
    cpa1 = P1 + ((dp_d1 - c * dp_d2) / denom).unsqueeze(-1) * d1
    cpa2 = P2 + ((c * dp_d1 - dp_d2) / denom).unsqueeze(-1) * d2
    return (cpa2 - cpa1).norm(dim=-1), (cpa1 + cpa2) * 0.5

def _track_dca_to_point(point, track_position, track_direction):
    """Straight-line DCA (B, T) of each track to one point (B, 3) per event."""
    diff = point.unsqueeze(1) - track_position
    return (diff - (diff * track_direction).sum(-1, keepdim=True) * track_direction).norm(dim=-1)

class VertexHead(nn.Module):

    """
    Primary-vertex fit, with secondary-vertex finding by default (see the module docstring). No
    learnable parameters unless learn_weights.

    Args:
        learn_weights, weight_hidden_dim: a small network that rescales each track's weight.
        use_helix, helix_iterations: helix refinement of the straight-line fit (when Bz != 0).
        Bz, is_cosmics, scaling_factor, noise_pt_threshold: MomentumHead settings.
        momentum_head: ready-built MomentumHead to use instead; the four settings above are ignored.
        find_secondary: find SVs and report the refitted PV; False = original PV-only fit.
        min_track_pv_dca, min_track_beam_dca: cm, SV candidate cuts.
        max_pair_doca, max_track_sv_doca: cm, seeding and attachment cuts.
        seed_merge_distance: cm, duplicate-SV radius.
        min_sv_tracks, min_sv_displacement (cm), max_sv_chi2_per_dof: SV cuts; ndf = 2 n_tracks - 3.
        max_sv_per_event: SV slots per event (K).
        refit_pv: report the PV refitted without SV tracks where valid; False = original fit.
    """
    def __init__(
        self,
        learn_weights: bool = False,
        weight_hidden_dim: int = 32,
        use_helix: bool = True,
        helix_iterations: int = 100,
        Bz: float = 1.4,
        is_cosmics: bool = False,
        scaling_factor: float = 100.0,
        noise_pt_threshold: float = 0.25,
        momentum_head: MomentumHead = None,
        find_secondary: bool = True,
        min_track_pv_dca: float = 0.05,
        min_track_beam_dca: float = 0.05,
        max_pair_doca: float = 1.0,
        max_track_sv_doca: float = 0.5,
        seed_merge_distance: float = 0.1,
        min_sv_tracks: int = 2,
        min_sv_displacement: float = 0.02,
        max_sv_chi2_per_dof: float = 10.0,
        max_sv_per_event: int = 6,
        refit_pv: bool = True,
    ):
        super().__init__()
        self.learn_weights = learn_weights
        self.use_helix = use_helix
        self.helix_iterations = helix_iterations
        self.find_secondary = find_secondary
        self.min_track_pv_dca = min_track_pv_dca
        self.min_track_beam_dca = min_track_beam_dca
        self.max_pair_doca = max_pair_doca
        self.max_track_sv_doca = max_track_sv_doca
        self.seed_merge_distance = seed_merge_distance
        self.min_sv_tracks = min_sv_tracks
        self.min_sv_displacement = min_sv_displacement
        self.max_sv_chi2_per_dof = max_sv_chi2_per_dof
        self.max_sv_per_event = max_sv_per_event
        self.refit_pv = refit_pv
        self.momentum_head = momentum_head if momentum_head is not None else MomentumHead(
            Bz=Bz, is_cosmics=is_cosmics, scaling_factor=scaling_factor, noise_pt_threshold=noise_pt_threshold)
        if learn_weights:
            # 6 inputs: class_probs (2) + track_reg_result (4); position deliberately excluded
            self.track_weight_network = nn.Sequential(
                nn.Linear(6, weight_hidden_dim),
                nn.GELU(),
                nn.Linear(weight_hidden_dim, 1),
                nn.Softplus(),  # weights must be >= 0
            )

    def dontUseHelix(self, use_helix: bool = False):
        self.use_helix = use_helix

    def isCosmics(self, is_cosmics: bool = True):
        self.momentum_head.is_cosmics = is_cosmics

    def setScalingFactor(self, scaling_factor: float = 100.0):
        self.momentum_head.scaling_factor = scaling_factor

    def setNoisePtThreshold(self, noise_pt_threshold: float = 0.25):
        self.momentum_head.noise_pt_threshold = noise_pt_threshold

    @property
    def Bz(self):
        return self.momentum_head.Bz

    @property
    def is_cosmics(self):
        return self.momentum_head.is_cosmics

    def _fit(self, track_position, track_direction, track_weight, params):
        """
        Straight-line fit, refined by the helix fit when use_helix and Bz != 0 (PV, SVs and refit).
        """
        fit = _fit_vertex_by_closest_approach(track_position, track_direction, track_weight)
        if self.use_helix and self.Bz != 0.0:
            fit = _fit_vertex_by_helix_closest_approach(
                params["circle_centre"],
                params["R_s"],
                params["zslope"],
                params["z0"],
                track_weight,               # already zeroed for bad tracks
                n_iterations=self.helix_iterations,
                linear_fit=fit,
                is_cosmics=self.is_cosmics,
            )
        return fit

    def _seed_secondary_vertices(self, track_position, track_direction, displaced_mask):
        """
        Up to K seeds per event, tightest candidate pair first.

        Returns:
            seeds (B, K, 3), NaN where empty, and seed_valid (B, K).
        """
        n_events, n_tracks, _ = track_position.shape
        doca, midpoint = _pairwise_doca_and_midpoint(track_position, track_direction)

        # each unordered pair once (i < j), both tracks candidates, DOCA within the cut
        upper_tri = torch.triu(torch.ones(n_tracks, n_tracks, device=track_position.device, dtype=torch.bool), diagonal=1)
        pair_valid = (displaced_mask.unsqueeze(2) & displaced_mask.unsqueeze(1) & upper_tri.unsqueeze(0)
                      & (doca < self.max_pair_doca))

        flat_valid = pair_valid.view(n_events, -1)
        top_idx = doca.view(n_events, -1).masked_fill(~flat_valid, float("inf")).argsort(dim=-1)[:, :self.max_sv_per_event]

        seeds = midpoint.view(n_events, -1, 3).gather(1, top_idx.unsqueeze(-1).expand(-1, -1, 3))
        seed_valid = flat_valid.gather(1, top_idx)
        # fewer than max_sv_per_event pairs (n_tracks^2 < K): pad with empty slots
        n_missing = self.max_sv_per_event - top_idx.shape[1]
        if n_missing > 0:
            seeds = torch.cat([seeds, seeds.new_zeros(n_events, n_missing, 3)], dim=1)
            seed_valid = torch.cat([seed_valid, seed_valid.new_zeros(n_events, n_missing)], dim=1)
        return seeds.masked_fill(~seed_valid.unsqueeze(-1), float("nan")), seed_valid

    def _suppress_duplicate_secondaries(self, sv_estimate, sv_chi2_per_dof, sv_is_valid):
        """
        keep (B, K): False for an SV within seed_merge_distance of a valid SV with lower chi2/ndf.
        """
        K = sv_estimate.shape[1]
        dist = (sv_estimate.unsqueeze(2) - sv_estimate.unsqueeze(1)).nan_to_num().norm(dim=-1)
        q1 = sv_chi2_per_dof.unsqueeze(2).nan_to_num(nan=float("inf"))
        q2 = sv_chi2_per_dof.unsqueeze(1).nan_to_num(nan=float("inf"))
        eye = torch.eye(K, device=sv_estimate.device, dtype=torch.bool).unsqueeze(0)
        dominated = (sv_is_valid.unsqueeze(1) & (dist < self.seed_merge_distance) & (q2 < q1) & ~eye).any(dim=2)
        return sv_is_valid & ~dominated

    def _find_secondary_vertices(self, pv_fit, track_position, track_direction, track_weight, params):
        """
        SV search after the PV fit (module docstring steps 1-6): the sv_* outputs, displaced_mask,
        displaced_weight and, with refit_pv, pv_refit.
        """
        n_events, n_tracks = track_weight.shape
        K = self.max_sv_per_event
        device, dtype = track_position.device, track_position.dtype
        pv_position = pv_fit["vertex_estimate"].nan_to_num()

        displaced_mask = ((pv_fit["track_dca"].nan_to_num() > self.min_track_pv_dca)
                          & (params["beam_dca"].nan_to_num() > self.min_track_beam_dca)
                          & params["usable"])
        displaced_weight = track_weight * displaced_mask.to(track_weight.dtype)

        seeds, seed_valid = self._seed_secondary_vertices(track_position, track_direction, displaced_mask)

        sv_estimate = torch.full((n_events, K, 3), float("nan"), device=device, dtype=dtype)
        sv_chi2 = torch.full((n_events, K), float("nan"), device=device, dtype=dtype)
        sv_chi2_per_dof = torch.full((n_events, K), float("nan"), device=device, dtype=dtype)
        sv_is_valid = torch.zeros((n_events, K), device=device, dtype=torch.bool)
        sv_n_tracks = torch.zeros((n_events, K), device=device, dtype=torch.long)
        sv_displacement = torch.full((n_events, K), float("nan"), device=device, dtype=dtype)
        sv_track_mask = torch.zeros((n_events, K, n_tracks), device=device, dtype=torch.bool)
        sv_track_dca = torch.full((n_events, K, n_tracks), float("nan"), device=device, dtype=dtype)

        for k in range(K):
            # candidates close enough to this seed (an empty slot's NaN seed becomes the origin)
            in_sv = ((_track_dca_to_point(seeds[:, k].nan_to_num(), track_position, track_direction)
                      < self.max_track_sv_doca) & (displaced_weight > 0))
            sv_weight = displaced_weight * in_sv.to(displaced_weight.dtype)
            fit = self._fit(track_position, track_direction, sv_weight, params)

            n_used = sv_weight.gt(0).sum(dim=-1)
            ndf = (2 * n_used.clamp(min=1) - 3).clamp(min=1).to(fit["chi_square"].dtype)
            chi2_per_dof = (fit["chi_square"] / ndf).masked_fill(~fit["fit_is_valid"], float("nan"))
            displacement = (fit["vertex_estimate"] - pv_position).norm(dim=-1)

            is_ok = (seed_valid[:, k] & fit["fit_is_valid"] & (n_used >= self.min_sv_tracks)
                     & (displacement >= self.min_sv_displacement)
                     & (chi2_per_dof.nan_to_num(nan=float("inf")) <= self.max_sv_chi2_per_dof))

            sv_estimate[:, k] = fit["vertex_estimate"].masked_fill(~is_ok.unsqueeze(-1), float("nan"))
            sv_chi2[:, k] = fit["chi_square"].masked_fill(~is_ok, float("nan"))
            sv_chi2_per_dof[:, k] = chi2_per_dof.masked_fill(~is_ok, float("nan"))
            sv_is_valid[:, k] = is_ok
            sv_n_tracks[:, k] = n_used.masked_fill(~is_ok, 0)
            sv_displacement[:, k] = displacement.masked_fill(~is_ok, float("nan"))
            sv_track_mask[:, k] = in_sv & is_ok.unsqueeze(-1)
            sv_track_dca[:, k] = fit["track_dca"].masked_fill(~is_ok.unsqueeze(-1), float("nan"))

        sv_is_valid = sv_is_valid & self._suppress_duplicate_secondaries(sv_estimate, sv_chi2_per_dof, sv_is_valid)
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
        if self.refit_pv:
            refit_weight = track_weight * (~result["sv_track_mask"].any(dim=1)).to(track_weight.dtype)
            result["pv_refit"] = self._fit(track_position, track_direction, refit_weight, params)
        return result

    def forward(
        self,
        class_probs,
        mask_probs,
        points,
        padding_mask,
    ):
        """
        Args:
            class_probs: (B, T, 2) P(slot empty), P(slot real).
            mask_probs: (B, N, T) hit-to-track assignment.
            points: (B, N, 4) normalized (E, eta, phi, r) hits, as TPCBatchDataset yields them.
            padding_mask: (B, N) True = real hit.

        Returns:
            dict, the PV (with find_secondary: the refit where used_refit, else the original fit):
                vertex_estimate (B, 3), chi_square (B,), fit_is_valid (B,); NaN where not valid.
                track_closest_approach_point (B, T, 3), track_dca (B, T): each track to the PV.
                track_position, track_direction (B, T, 3): innermost hit, straight-line direction.
                track_weight, track_quality_mask (B, T): fit weight, pT / beam-line cut.
            With find_secondary also (NaN / False / 0 where sv_is_valid is False):
                sv_estimate (B, K, 3); sv_is_valid, sv_chi2, sv_chi2_per_dof, sv_displacement,
                    sv_n_tracks (B, K); sv_track_mask, sv_track_dca (B, K, T).
                displaced_mask, displaced_weight (B, T): SV candidates and their weight.
                pv_original: dict, the original PV fit.
                used_refit (B,): the reported PV is the refit.
        """
        params = self.momentum_head(mask_probs, points, padding_mask)
        track_position, total_hit_weight = params["track_position"], params["total_hit_weight"]
        track_reg_result = params["track_reg_result"]

        theta = track_reg_result[..., 1]
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        track_direction = torch.stack([sin_theta * track_reg_result[..., 3], sin_theta * track_reg_result[..., 2], cos_theta], dim=-1)
        track_direction = track_direction / track_direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        probability_track_is_real = class_probs[..., 1]

        if self.learn_weights:
            track_quality_features = torch.cat([class_probs, track_reg_result], dim=-1)
            learned_weight_scale = self.track_weight_network(track_quality_features).squeeze(-1)
            track_weight = probability_track_is_real * total_hit_weight * learned_weight_scale
        else:
            track_weight = probability_track_is_real * total_hit_weight

        track_weight = track_weight * params["usable"].to(dtype=track_weight.dtype)

        fit = self._fit(track_position, track_direction, track_weight, params)

        result = {
            "track_position": track_position,
            "track_direction": track_direction,
            "track_weight": track_weight,
            "track_quality_mask": params["usable"],
        }
        if self.find_secondary:
            secondary = self._find_secondary_vertices(fit, track_position, track_direction, track_weight, params)
            refit = secondary.pop("pv_refit", None)
            result.update(secondary)
            result["pv_original"] = fit
            # the refit PV where the event has an SV and the refit is valid; else the original PV fit
            used_refit = secondary["sv_is_valid"].any(dim=1)
            if refit is not None:
                used_refit = used_refit & refit["fit_is_valid"]
                fit = {k: torch.where(used_refit.view(-1, *[1] * (v.dim() - 1)), refit[k], v) for k, v in fit.items()}
            else:
                used_refit = torch.zeros_like(used_refit)
            result["used_refit"] = used_refit
        result.update({
            "vertex_estimate": fit["vertex_estimate"],
            "chi_square": fit["chi_square"],
            "fit_is_valid": fit["fit_is_valid"],
            "track_closest_approach_point": fit["track_closest_approach_point"],
            "track_dca": fit["track_dca"],
        })
        return result