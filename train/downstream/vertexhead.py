"""
vertexhead.py: primary- and secondary-vertex fit from track-finder output.

Tracks are ACTS bound perigee parameters (d0, z0, phi, theta, q/p, t) w.r.t. the beam spot, in cm: either
fitted from each track's assigned hits by the track model below (a Taubin circle in x-y and z against
signed arc length along it; with use_acts, ACTS propagates that fitted helix to the beam spot), or given
directly (track_parameters, e.g. from ACTS, in mm). The PV is the
weighted least-squares point of closest approach to the usable tracks' straight lines at their perigee
(weight = soft hit count x P(real)), refined by an iterative helix fit; an event where the helix fit
gives no valid vertex (or Bz = 0) keeps the straight-line result.

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
import atexit
import json
import math
import os
import struct
import subprocess
import sys

import torch
import torch.nn as nn

# ------------------------------------------------------------------------------------------------
# Track model: hit normalization, hard hit-to-track assignment, and each track's circle and z line
# (moved here from momentumhead.py, which is experimental and imports these; its momentum-only outputs,
# track_reg_result and radius, stay there)
# ------------------------------------------------------------------------------------------------

# TPCBatchDataset's hit normalization (fm4npp/datasets/dataset.py, TPCBatchDataset.__init__): eta, phi and
# transverse r (cm) are min-max scaled to [0, 1], E is z-scored. Keep in line with the dataset.
TPC_NORMALIZATION = {
    "eta_lim": (-2.0, 2.0),
    "phi_lim": (-torch.pi, torch.pi),
    "r_lim": (31.371997833251953, 75.38493347167969),
    "E_mean": 253.0982,
    "E_std": 268.7093,
}

def _per_track_sum(values, track, n_tracks):
    """Sum per-hit values (B, N) into their track slots (B, T); hits on no track must carry 0."""
    return values.new_zeros(values.shape[0], n_tracks).scatter_add_(1, track, values)

def _line_fit(u, v, track, assigned, n_tracks, min_hits=2, eps=1e-12):
    """
    Deming (orthogonal-distance) line fit v = slope * u + intercept, per track.

    Args:
        u, v: (B, N) per hit; track: (B, N) each hit's track slot; assigned: (B, N) bool, the hit is on
            that track; n_tracks: T.

    Returns:
        slope, intercept: (B, T); NaN with fewer than min_hits hits.
    """
    mask = assigned.to(u.dtype)
    u, v = u.masked_fill(~assigned, 0.0), v.masked_fill(~assigned, 0.0)    # unassigned hits can't add NaN
    n_valid = _per_track_sum(mask, track, n_tracks).clamp(min=eps)
    valid = n_valid >= min_hits

    mean_u = _per_track_sum(u, track, n_tracks) / n_valid
    mean_v = _per_track_sum(v, track, n_tracks) / n_valid

    # each hit is centred on its own track's mean
    du = (u - mean_u.gather(1, track)) * mask
    dv = (v - mean_v.gather(1, track)) * mask

    ssd_u  = _per_track_sum(du * du, track, n_tracks)
    ssd_v  = _per_track_sum(dv * dv, track, n_tracks)
    ssd_uv = _per_track_sum(du * dv, track, n_tracks)

    slope = ((ssd_v - ssd_u + torch.sqrt((ssd_v - ssd_u) ** 2 + 4 * ssd_uv ** 2)) / 2.0 /
              torch.where(ssd_uv.abs() < eps, eps, ssd_uv))
    intercept = mean_v - slope * mean_u

    slope = slope.masked_fill(~valid, float("nan"))
    intercept = intercept.masked_fill(~valid, float("nan"))
    return slope, intercept

def _circle_fit(x, y, track, assigned, n_tracks, min_hits=3, newton_iters=20, eps=1e-12):
    """
    Taubin circle fit per track, in float64.

    Args:
        x, y: (B, N) per hit; track: (B, N) each hit's track slot; assigned: (B, N) bool, the hit is on
            that track; n_tracks: T.

    Returns:
        radius, x_centre, y_centre: (B, T) float64; NaN with fewer than min_hits hits.
    """
    mask = assigned.double()
    x, y = x.double().masked_fill(~assigned, 0.0), y.double().masked_fill(~assigned, 0.0)
    n_valid = _per_track_sum(mask, track, n_tracks).clamp(min=eps)
    valid = n_valid >= min_hits
    mean_x = _per_track_sum(x, track, n_tracks) / n_valid
    mean_y = _per_track_sum(y, track, n_tracks) / n_valid
    # each hit is centred on its own track's mean
    Xi, Yi = (x - mean_x.gather(1, track)) * mask, (y - mean_y.gather(1, track)) * mask
    Zi = Xi * Xi + Yi * Yi
    Mxy, Mxx, Myy = (_per_track_sum(v, track, n_tracks) / n_valid for v in (Xi * Yi, Xi * Xi, Yi * Yi))
    Mxz, Myz, Mzz = (_per_track_sum(v, track, n_tracks) / n_valid for v in (Xi * Zi, Yi * Zi, Zi * Zi))

    Mz = Mxx + Myy
    Cov_xy = Mxx * Myy - Mxy * Mxy
    Var_z = Mzz - Mz * Mz
    A3, A2 = 4 * Mz, -3 * Mz * Mz - Mzz
    A1 = Var_z * Mz + 4 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz
    A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy

    xk, yk = torch.zeros_like(Mz), A0.clone()
    improving_any = torch.ones_like(Mz, dtype=torch.bool)
    for _ in range(newton_iters):
        Dy = A1 + xk * (2 * A2 + 3 * A3 * xk)
        xnew = xk - yk / torch.where(Dy.abs() < eps, eps, Dy)
        ynew = A0 + xnew * (A1 + xnew * (A2 + xnew * A3))
        better = (ynew.abs() < yk.abs()) & torch.isfinite(xnew) & improving_any
        xk, yk = torch.where(better, xnew, xk), torch.where(better, ynew, yk)
        improving_any = improving_any & better

    DET = xk * xk - xk * Mz + Cov_xy
    DET = torch.where(DET.abs() < eps, eps, DET)
    Xc = (Mxz * (Myy - xk) - Myz * Mxy) / DET / 2
    Yc = (Myz * (Mxx - xk) - Mxz * Mxy) / DET / 2
    radius = torch.sqrt((Xc * Xc + Yc * Yc + Mz).clamp(min=0.0))

    return (radius.masked_fill(~valid, float("nan")),
            Xc.masked_fill(~valid, float("nan")) + mean_x,
            Yc.masked_fill(~valid, float("nan")) + mean_y)

def _extreme_hit(r, track, assigned, n_tracks, outermost=False):
    """
    Hit index (B, T) of each track's innermost (or outermost) assigned hit by r: the first such hit on
    ties, and 0 for a track with none, as argmin / argmax over a masked (B, N, T) grid would give.
    """
    n_events, n_hits = r.shape
    key = torch.where(assigned, r if outermost else -r, -torch.inf)
    best = key.new_full((n_events, n_tracks), -torch.inf).scatter_reduce_(1, track, key, "amax")
    hit = torch.arange(n_hits, device=r.device).expand_as(track)
    first = torch.full((n_events, n_tracks), n_hits, dtype=torch.long, device=r.device).scatter_reduce_(
        1, track, torch.where(assigned & (key == best.gather(1, track)), hit, n_hits), "amin")
    return first.masked_fill(first == n_hits, 0)

def _arc_length(x, y, xc, yc, R_s, beam_position=(0.0, 0.0)):
    """
    Signed arc length along a circle (centre xc, yc; signed radius R_s, sign(R_s) = sign(q Bz)) from its
    closest approach to the beam line (at beam_position in x-y, cm) to the circle point at the angle of
    (x, y); positive in the direction of travel (anticlockwise <=> R_s < 0). All inputs broadcast.
    """
    # in place, so a long per-hit array costs one allocation
    s = torch.atan2(y - yc, x - xc)
    beam_angle = torch.atan2(-(yc - beam_position[1]), -(xc - beam_position[0]))
    return s.sub_(beam_angle).add_(torch.pi).remainder_(2 * torch.pi).sub_(torch.pi).mul_(-R_s)

def _unnormalize_hits(points, normalization=TPC_NORMALIZATION):
    """
    Invert the dataset's hit normalization (z-scored E; eta, phi, r min-max scaled to [0, 1]).

    Args:
        normalization: dict eta_lim, phi_lim, r_lim (cm), E_mean, E_std; TPC_NORMALIZATION by default.

    Returns:
        points (..., 4) as (E, x, y, z) in cm, and the transverse radius r (...).
    """
    eta_lim, phi_lim, r_lim = normalization["eta_lim"], normalization["phi_lim"], normalization["r_lim"]
    E_mean, E_std = normalization["E_mean"], normalization["E_std"]

    E = points[..., 0] * E_std + E_mean
    eta = points[..., 1] * (eta_lim[1] - eta_lim[0]) + eta_lim[0]
    phi = points[..., 2] * (phi_lim[1] - phi_lim[0]) + phi_lim[0]
    r = points[..., 3] * (r_lim[1] - r_lim[0]) + r_lim[0]
    return torch.stack([E, r * torch.cos(phi), r * torch.sin(phi), r * torch.sinh(eta)], dim=-1), r

def _track_position_from_hits(points, hit_r, mask_probs, padding_mask):
    """
    Hard hit-to-track assignment and each track's reference point.

    Args:
        points: (B, N, 4) (E, x, y, z); hit_r: (B, N); mask_probs: (B, N, T); padding_mask: (B, N).

    Returns:
        track_position: (B, T, 3) innermost assigned hit; 0 for a track with none.
        total_hit_weight: (B, T) sum of mask_probs over real hits.
        is_assigned: (B, N, T) each real hit on its argmax track (none if all probabilities are 0).
    """
    n_tracks = mask_probs.shape[-1]
    total_hit_weight = (mask_probs * padding_mask.unsqueeze(-1).to(mask_probs.dtype)).sum(dim=1)

    # each real hit belongs to the track slot with the largest mask_probs; hits with zero probability for
    # every track are unassigned
    track = mask_probs.argmax(dim=-1)
    assigned = padding_mask & (mask_probs.max(dim=-1).values > 0)
    is_assigned = (torch.arange(n_tracks, device=track.device) == track.unsqueeze(-1)) & assigned.unsqueeze(-1)

    innermost = _extreme_hit(hit_r, track, assigned, n_tracks)
    track_position = points[..., 1:4].gather(1, innermost.unsqueeze(-1).expand(-1, -1, 3))
    track_position = track_position.masked_fill(~is_assigned.any(dim=1).unsqueeze(-1), 0.0)
    return track_position, total_hit_weight, is_assigned

def _helix_from_hits(points, hit_r, is_assigned, noise_pt_threshold, Bz=1.4, scaling_factor=100.0,
                     valid_radius=1.0, min_hits=3, is_cosmics=False, beam_position=(0.0, 0.0)):
    """
    Each track's helix (circle and z line) and quality labels from its assigned hits.

    Travel from the innermost to the outermost hit gives the charge and phi at the innermost hit:
    centre = innermost hit + R_s (sin phi, -cos phi), sign(R_s) = sign(q Bz).

    Args:
        points, hit_r, is_assigned: as from _track_position_from_hits.
        noise_pt_threshold: GeV; lower fitted pT -> not usable.
        Bz: Tesla; scaling_factor: converts metres to the units of points.
        valid_radius: cm; the circle must pass within this of the beam line to be usable.
        min_hits: fewer assigned hits -> not fitted; is_cosmics: fit z against x instead of s.
        beam_position: (x, y) of the beam line, cm; arc length and beam_dca are measured from it.

    Returns:
        dict, float64, NaN (or False) where fit_ok is False:
            usable (B, T): pT >= noise_pt_threshold and beam_dca < valid_radius.
            pT, theta, beam_dca, fit_ok (B, T).
            circle_centre (B, T, 2), R_s, zslope, z0 (B, T): the circle and z line z = zslope * s + z0
                (s from _arc_length).
    """
    # per hit (B, N): each hit's track slot (0 if unassigned) and coordinates
    n_tracks = is_assigned.shape[-1]
    assigned = is_assigned.any(dim=-1)
    track = is_assigned.to(torch.uint8).argmax(dim=-1)
    x, y, z = (points[..., k].double() for k in (1, 2, 3))
    r = hit_r.double()

    radius, xc, yc = _circle_fit(x, y, track, assigned, n_tracks, min_hits=min_hits)

    # innermost and outermost assigned hit of each track
    i_in, i_out = (_extreme_hit(r, track, assigned, n_tracks, outermost=o) for o in (False, True))
    xin, yin = x.gather(1, i_in), y.gather(1, i_in)
    xout, yout = x.gather(1, i_out), y.gather(1, i_out)

    # travel inner -> outer: anticlockwise about the centre <=> R_s < 0
    anticlockwise = ((xin - xc) * (yout - yc) - (yin - yc) * (xout - xc)) > 0
    sign_Rs = torch.where(anticlockwise, -1.0, 1.0).to(radius.dtype)

    # a helix is linear in arc length, not in r: fit z against the signed arc length along the circle
    # from its closest approach to the beam line (positive in the direction of travel)
    s = x if is_cosmics else _arc_length(x, y, xc.gather(1, track), yc.gather(1, track),
                                         (sign_Rs * radius).gather(1, track), beam_position)
    zslope, z0 = _line_fit(s, z, track, assigned, n_tracks, min_hits=min_hits)
    ux, uy = (xc - xin) / radius, (yc - yin) / radius
    sin_phi, cos_phi = sign_Rs * ux, -sign_Rs * uy
    q = sign_Rs * (1.0 if Bz >= 0 else -1.0)
    pT = 0.299792458 * abs(Bz) * radius / scaling_factor     # pT [GeV] = c [1e9 m/s] * B [T] * R [m]
    # cot(theta) = dz/ds: the arc-length slope, or dz/dx * dx/ds = zslope cos(phi) for cosmics
    cot_theta = zslope * cos_phi if is_cosmics else zslope
    theta = torch.atan2(torch.ones_like(cot_theta), cot_theta)
    dx, dy = xc - beam_position[0], yc - beam_position[1]
    beam_dca = (torch.sqrt(dx * dx + dy * dy) - radius).abs()

    fit_ok = torch.isfinite(torch.stack([q / (pT + 1.0), theta, sin_phi, cos_phi], dim=-1)).all(dim=-1)
    nan = torch.full_like(pT, float("nan"))
    pT, theta, beam_dca, R_s, zslope, z0 = (torch.where(fit_ok, v, nan)
                                            for v in (pT, theta, beam_dca, sign_Rs * radius, zslope, z0))

    return {
        "usable": (pT >= noise_pt_threshold) & (beam_dca < valid_radius),    # NaN where fit_ok is False -> False
        "pT": pT, "theta": theta, "beam_dca": beam_dca, "fit_ok": fit_ok,
        "circle_centre": torch.where(fit_ok.unsqueeze(-1), torch.stack([xc, yc], dim=-1), nan.unsqueeze(-1)),
        "R_s": R_s, "zslope": zslope, "z0": z0,
    }


# ------------------------------------------------------------------------------------------------
# ACTS bound perigee parameters (Acts::BoundIndices order: d0, z0, phi, theta, q/p, t), with the perigee
# line parallel to z through the beam spot (beam_position, z reference 0) and ACTS's d0 sign (LineSurface):
# perigee point = beam + d0 (-sin phi, cos phi). Lengths in cm here (ACTS uses mm), q/p in e/GeV, t = 0.
# ------------------------------------------------------------------------------------------------

def _perigee_from_circle(circle_centre, R_s, zslope, z0, qop, beam_position=(0.0, 0.0)):
    """
    Perigee parameters (B, T, 6) of the track model's helix.

    Args:
        circle_centre (B, T, 2), R_s (B, T): circle; sign(R_s) = sign(q Bz), centre = point + R_s (sin phi, -cos phi).
        zslope, z0: (B, T) z line z = zslope * s + z0, s the signed arc length from the closest approach to
            the beam line (so zslope = cot theta and z0 is the perigee z).
        qop: (B, T) q/p, e/GeV.
        beam_position: (x, y) of the beam line, cm.
    """
    bx, by = beam_position
    ux, uy = bx - circle_centre[..., 0], by - circle_centre[..., 1]
    norm = torch.sqrt(ux * ux + uy * uy).clamp(min=1e-12)
    ux, uy = ux / norm, uy / norm
    h, radius = torch.sign(R_s), R_s.abs()
    phi = torch.atan2(-h * ux, h * uy)                         # heading at the perigee, along the direction of travel
    d0 = (-(circle_centre[..., 0] + radius * ux - bx) * torch.sin(phi)
          + (circle_centre[..., 1] + radius * uy - by) * torch.cos(phi))
    theta = torch.atan2(torch.ones_like(zslope), zslope)
    return torch.stack([d0, z0, phi, theta, qop, torch.zeros_like(d0)], dim=-1)

def _circle_from_perigee(perigee, Bz, scaling_factor=100.0, beam_position=(0.0, 0.0)):
    """
    The track model's helix from perigee parameters (B, T, 6) in cm: the inverse of _perigee_from_circle.

    Returns:
        circle_centre (B, T, 2), R_s, zslope, z0, pT (GeV), beam_dca = |d0| (B, T).
    """
    d0, z0, phi, theta, qop = perigee[..., :5].unbind(-1)
    sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
    pT = torch.sin(theta) / qop.abs()
    R_s = torch.sign(qop) * (1.0 if Bz >= 0 else -1.0) * pT * scaling_factor / (0.299792458 * abs(Bz))
    pca_x, pca_y = beam_position[0] - d0 * sin_phi, beam_position[1] + d0 * cos_phi
    circle_centre = torch.stack([pca_x + R_s * sin_phi, pca_y - R_s * cos_phi], dim=-1)
    return circle_centre, R_s, torch.cos(theta) / torch.sin(theta), z0, pT, d0.abs()

# ------------------------------------------------------------------------------------------------
# ACTS worker (use_acts): unmodified ACTS, through its Python bindings, in its own container
# (scripts/acts), run as one persistent process per configuration. This file never imports ACTS; it
# streams track states to scripts/acts/acts_worker.py and reads the perigee parameters back.
# ------------------------------------------------------------------------------------------------

ACTS_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  "scripts", "acts", "acts_worker.py")
_ACTS_WORKERS = {}

class _ActsWorker:
    """One ACTS worker process: command (e.g. singularity exec <image> acts-python) + worker script + config."""
    def __init__(self, command, config):
        self.process = subprocess.Popen(list(command) + [ACTS_WORKER_SCRIPT, json.dumps(config)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        while True:                     # start-up output (e.g. ACTS's setup) until the ready line
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"ACTS worker exited during start-up (code {self.process.wait()}): "
                                   f"{' '.join(command)}; its error output is above")
            if line.strip() == b"ACTS_WORKER_READY":
                break
            sys.stderr.write(line.decode(errors="replace"))

    def propagate(self, position, direction, qop, backward):
        """numpy (n, 3), (n, 3), (n,), (n,) bool in mm, e/GeV -> perigee parameters (n, 6) and ok (n,)."""
        import numpy as np
        n = len(qop)
        self.process.stdin.write(struct.pack("<Q", n) + np.ascontiguousarray(position, "<f8").tobytes()
                                 + np.ascontiguousarray(direction, "<f8").tobytes()
                                 + np.ascontiguousarray(qop, "<f8").tobytes() + np.asarray(backward, np.uint8).tobytes())
        self.process.stdin.flush()
        reply = self.process.stdout.read(n * 49)
        if len(reply) != n * 49:
            raise RuntimeError(f"ACTS worker stopped (code {self.process.poll()}); its error output is above")
        return (np.frombuffer(reply[: n * 48], "<f8").reshape(n, 6).copy(),
                np.frombuffer(reply[n * 48:], np.uint8).astype(bool))

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=30)

def _acts_worker(command, config):
    """The running worker for this command and config, started on first use (and again if it died)."""
    key = (tuple(command), json.dumps(config, sort_keys=True))
    if key not in _ACTS_WORKERS or _ACTS_WORKERS[key].process.poll() is not None:
        _ACTS_WORKERS[key] = _ActsWorker(command, config)
    return _ACTS_WORKERS[key]

atexit.register(lambda: [worker.close() for worker in _ACTS_WORKERS.values()])

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
    is_cosmics=False,
    tolerance=1e-4,
    beam_position=(0.0, 0.0),
    **linear_fit_kwargs,
):
    """
    Helix-aware vertex fit: repeats the straight-line fit on each track's tangent at its point of
    closest approach to the current vertex.

    Args:
        circle_centre (B, T, 2), R_s (B, T): the track model's circle; sign(R_s) = sign(q Bz).
        zslope, z0: (B, T) z line z = zslope * s + z0, s the signed arc length from the circle's
            closest approach to the beam line (_arc_length; s -> x if is_cosmics).
        track_weight: (B, T); tracks with NaN parameters are excluded.
        n_iterations: maximum; stops once every vertex moves < tolerance; below 1 returns linear_fit.
        linear_fit: straight-line fit result whose vertex is the start (origin if None or NaN).
        tolerance: cm; convergence threshold on the vertex step.
        beam_position: (x, y) of the beam line, cm, as in the track model (the origin of s).
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
        # nearest point on each circle to the vertex in x-y, and the exact tangent there along the
        # direction of travel (anticlockwise <=> R_s < 0); dz/ds = zslope (dz/dx * dx/ds for cosmics)
        n = PV[:, None, :2] - circle_centre
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        pca_circle = circle_centre + radius.unsqueeze(-1) * n
        t_xy = torch.sign(R_s).unsqueeze(-1) * torch.stack([n[..., 1], -n[..., 0]], dim=-1)

        if is_cosmics:
            pca_z, dz = pca_circle[..., 0] * zslope + z0, zslope * t_xy[..., 0]
        else:
            pca_z, dz = _arc_length(pca_circle[..., 0], pca_circle[..., 1], x0, y0, R_s, beam_position) * zslope + z0, zslope
        pca = torch.cat([pca_circle, pca_z.unsqueeze(-1)], dim=-1)
        tangent = torch.cat([t_xy, dz.unsqueeze(-1)], dim=-1)
        # clamped: a stand-in track (centre at the origin) with the vertex at the origin has no direction
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        result = _fit_vertex_by_closest_approach(pca, tangent, track_weight, **linear_fit_kwargs)

        still_moving = (still_moving & result["fit_is_valid"] &
                         ((result["vertex_estimate"] - PV).norm(dim=-1) > tolerance))
        
        # advance PV only where the fit is valid, so one bad event can't spread NaN
        PV = torch.where(result["fit_is_valid"].unsqueeze(-1), result["vertex_estimate"], PV)
       
        if not still_moving.any():
            break

    result["track_dca"] = result["track_dca"].masked_fill(failed, float("nan"))
    result["track_closest_approach_point"] = result["track_closest_approach_point"].masked_fill(failed.unsqueeze(-1), float("nan"))
    return result

class VertexHead(nn.Module):

    """
    Primary-vertex fit, with secondary-vertex finding by default (see the module docstring). No
    learnable parameters unless learn_weights. Detector-dependent values are arguments (defaults: sPHENIX).

    Args:
        learn_weights: train the track weights. Adds a small network that rescales each track's weight,
            and makes the usable cuts (noise_pt_threshold, valid_radius) learnable gates. Everything
            starts at the pure-maths result: the network outputs 1 and the gates are the hard cuts in
            the forward pass (sigmoids of width gate_width_pt / gate_width_r carry the gradient).
        weight_hidden_dim: hidden width of that network.
        gate_width_pt (GeV), gate_width_r (cm): initial widths of the learnable gates; they only shape the
            gradient (the forward pass uses the hard cuts), so wider widths give more tracks a gradient.
        helix_iterations: maximum helix-fit iterations (the helix refines the straight-line fit).
        convergence_tolerance: cm; the helix fit stops once every vertex moves less than this.
        minimum_total_weight: an event needs more total track weight than this for a valid fit.
        Bz: solenoid field along z, Tesla (uniform).
        is_cosmics: fit z against x instead of arc length.
        scaling_factor: converts the radius pT / (0.2998 Bz), in metres, to the units of points (100: cm).
        noise_pt_threshold: GeV; tracks with lower fitted pT are labelled noise (not usable).
        valid_radius: cm; a track is usable only if its circle passes within this of the beam line.
        min_hits: assigned hits needed for the circle and z-line fits.
        beam_position: (x, y) of the beam line, cm.
        normalization: the dataset's hit normalization (see _unnormalize_hits); TPC_NORMALIZATION if None.
        momentum_head: an (experimental) MomentumHead whose config() supplies the eight settings above.
        track_parameter_length_scale: cm per length unit of forward's track_parameters (ACTS mm: 0.1).
        use_acts: ACTS builds the perigee parameters from the hit fit: the fitted helix's state nearest the
            innermost hit, propagated to the beam spot by ACTS (scripts/acts/acts_worker.py, one persistent
            process); tracks ACTS fails on are dropped. Not with is_cosmics.
        acts_image: the ACTS container image (scripts/acts: acts-ref-48.2.0.sif) for use_acts.
        acts_command: instead of acts_image, the command that runs Python with ACTS (a list).
        acts_field_map: a text field map for ACTS (scripts/acts/convert_field_map.py, mm and T), as the
            container sees it; None = constant Bz. The hit fit itself still uses Bz.
        acts_field_map_kind: "xyz" or "rz" map.
        acts_max_steps: ACTS's propagation step limit (None: ACTS's default).
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
        helix_iterations: int = 100,
        Bz: float = 1.4,
        is_cosmics: bool = False,
        scaling_factor: float = 100.0,
        noise_pt_threshold: float = 0.25,
        valid_radius: float = 1.0,
        min_hits: int = 3,
        beam_position: tuple = (0.0, 0.0),
        normalization: dict = None,
        momentum_head=None,
        track_parameter_length_scale: float = 0.1,
        use_acts: bool = False,
        acts_image: str = None,
        acts_command: list = None,
        acts_field_map: str = None,
        acts_field_map_kind: str = "xyz",
        acts_max_steps: int = None,
        convergence_tolerance: float = 1e-4,
        minimum_total_weight: float = 1e-6,
        gate_width_pt: float = 0.05,
        gate_width_r: float = 0.2,
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
        self.helix_iterations = helix_iterations
        self.convergence_tolerance = convergence_tolerance
        self.track_parameter_length_scale = track_parameter_length_scale
        self.use_acts = use_acts
        self.acts_image = acts_image
        self.acts_command = acts_command
        self.acts_field_map = acts_field_map
        self.acts_field_map_kind = acts_field_map_kind
        self.acts_max_steps = acts_max_steps
        self.minimum_total_weight = minimum_total_weight
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
        # track-model settings (an experimental MomentumHead's config() overrides them)
        settings = dict(Bz=Bz, is_cosmics=is_cosmics, scaling_factor=scaling_factor,
                        noise_pt_threshold=noise_pt_threshold, valid_radius=valid_radius, min_hits=min_hits,
                        beam_position=beam_position, normalization=normalization)
        if momentum_head is not None:
            settings.update(momentum_head.config())
        for name, value in settings.items():
            setattr(self, name, value)
        self.beam_position = tuple(self.beam_position)
        self.normalization = dict(TPC_NORMALIZATION if self.normalization is None else self.normalization)
        if learn_weights:
            # 6 inputs: class_probs (2) + (q/(pT+1), theta, sin phi, cos phi) at the perigee; position excluded
            self.track_weight_network = nn.Sequential(
                nn.Linear(6, weight_hidden_dim),
                nn.GELU(),
                nn.Linear(weight_hidden_dim, 1),
                nn.Softplus(),  # weights must be >= 0
            )
            # start at the pure-maths weights: zero last layer, softplus(log(e - 1)) = 1
            nn.init.zeros_(self.track_weight_network[2].weight)
            nn.init.constant_(self.track_weight_network[2].bias, math.log(math.e - 1))
            # the usable cuts as learnable gates, starting at noise_pt_threshold / valid_radius; widths as logs (> 0)
            self.pt_threshold = nn.Parameter(torch.tensor(float(self.noise_pt_threshold)))
            self.beam_radius = nn.Parameter(torch.tensor(float(self.valid_radius)))
            self.log_gate_width_pt = nn.Parameter(torch.tensor(math.log(gate_width_pt)))
            self.log_gate_width_r = nn.Parameter(torch.tensor(math.log(gate_width_r)))

    def isCosmics(self, is_cosmics: bool = True):
        self.is_cosmics = is_cosmics

    def setScalingFactor(self, scaling_factor: float = 100.0):
        self.scaling_factor = scaling_factor

    def setNoisePtThreshold(self, noise_pt_threshold: float = 0.25):
        self.noise_pt_threshold = noise_pt_threshold

    def acts_worker_command(self):
        """The command that runs Python with ACTS for use_acts."""
        if self.acts_command:
            return list(self.acts_command)
        if not self.acts_image:
            raise ValueError("use_acts needs acts_image (the ACTS container, scripts/acts) or acts_command")
        return ["singularity", "exec", self.acts_image, "acts-python"]

    def acts_worker_config(self):
        """The ACTS worker's settings (ACTS units: mm, T)."""
        return {"reference": [self.beam_position[0] / self.track_parameter_length_scale,
                              self.beam_position[1] / self.track_parameter_length_scale, 0.0],
                "bz_tesla": self.Bz, "field_map": self.acts_field_map, "field_map_kind": self.acts_field_map_kind,
                "max_steps": self.acts_max_steps}

    def track_fit_config(self):
        """The track-model settings (the keys of an experimental MomentumHead's config())."""
        return {k: getattr(self, k) for k in ("Bz", "is_cosmics", "scaling_factor", "noise_pt_threshold",
                                              "valid_radius", "min_hits", "beam_position", "normalization")}

    def _fit(self, track_position, track_direction, track_weight, params):
        """
        Straight-line fit, refined by the helix fit (PV, SVs and refit); an event whose helix fit is not
        valid, and every event when Bz = 0, keeps the straight-line result.
        Tracks without a fitted circle and z line get NaN track_dca and track_closest_approach_point.
        """
        fit = _fit_vertex_by_closest_approach(track_position, track_direction, track_weight,
                                              minimum_total_weight=self.minimum_total_weight)
        if self.Bz != 0.0:
            linear, fit = fit, _fit_vertex_by_helix_closest_approach(
                params["circle_centre"],
                params["R_s"],
                params["zslope"],
                params["z0"],
                track_weight,               # already zeroed for bad tracks
                n_iterations=self.helix_iterations,
                linear_fit=fit,
                is_cosmics=self.is_cosmics,
                tolerance=self.convergence_tolerance,
                beam_position=self.beam_position,
                minimum_total_weight=self.minimum_total_weight,
            )
            keep_linear = ~fit["fit_is_valid"]
            fit = {k: torch.where(keep_linear.view(-1, *[1] * (v.dim() - 1)), linear[k], v) for k, v in fit.items()}
        fit["track_dca"] = fit["track_dca"].masked_fill(~params["fit_ok"], float("nan"))
        fit["track_closest_approach_point"] = fit["track_closest_approach_point"].masked_fill(
            ~params["fit_ok"].unsqueeze(-1), float("nan"))
        return fit

    def _find_secondary_vertices(self, pv_fit, track_position, track_direction, track_weight, params):
        """
        SV search after the PV fit (module docstring steps 1-6): the sv_* outputs, displaced_mask,
        displaced_weight and, with refit_pv and any SV in the batch, pv_refit.
        """
        n_events, n_tracks = track_weight.shape
        K = self.max_sv_per_event
        device, dtype = track_position.device, track_position.dtype
        pv_position = pv_fit["vertex_estimate"].nan_to_num()

        displaced_mask = ((pv_fit["track_dca"].nan_to_num() > self.min_track_pv_dca)
                          & (params["beam_dca"].nan_to_num() > self.min_track_beam_dca)
                          & params["usable"])
        displaced_weight = track_weight * displaced_mask.to(track_weight.dtype)

        # seeds: closest-approach midpoints of candidate pairs (each once, i < j) with straight-line
        # DOCA < max_pair_doca, tightest first; nearly parallel pairs (1 - c^2 < 1e-12) get a finite
        # but meaningless DOCA
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

        for k in range(K):
            if not seed_valid[:, k].any():
                continue                 # no event has a seed in this slot: its outputs stay empty
            # candidates within max_track_sv_doca (straight line) of this seed; an empty slot's NaN
            # seed becomes the origin
            diff = seeds[:, k].nan_to_num().unsqueeze(1) - track_position
            in_sv = (((diff - (diff * track_direction).sum(-1, keepdim=True) * track_direction).norm(dim=-1)
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

        # drop SVs within seed_merge_distance of a valid SV with a strictly lower chi2/ndf
        dist = (sv_estimate.unsqueeze(2) - sv_estimate.unsqueeze(1)).nan_to_num().norm(dim=-1)
        q = sv_chi2_per_dof.nan_to_num(nan=float("inf"))
        sv_is_valid = sv_is_valid & ~(sv_is_valid.unsqueeze(1) & (dist < self.seed_merge_distance)
                                      & (q.unsqueeze(1) < q.unsqueeze(2))
                                      & ~torch.eye(K, device=device, dtype=torch.bool)).any(dim=2)
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
            refit_weight = track_weight * (~result["sv_track_mask"].any(dim=1)).to(track_weight.dtype)
            result["pv_refit"] = self._fit(track_position, track_direction, refit_weight, params)
        return result

    def forward(
        self,
        class_probs=None,
        mask_probs=None,
        points=None,
        padding_mask=None,
        track_parameters=None,
        track_weight=None,
    ):
        """
        Two inputs: track-finder output (class_probs, mask_probs, points, padding_mask), whose hits the
        track model fits (with use_acts, ACTS then builds the perigee parameters); or track_parameters,
        e.g. from ACTS, used as they are.

        Args:
            class_probs: (B, T, 2) P(slot empty), P(slot real); with track_parameters optional (P(real) = 1).
            mask_probs: (B, N, T) hit-to-track assignment.
            points: (B, N, 4) normalized (E, eta, phi, r) hits, as TPCBatchDataset yields them.
            padding_mask: (B, N) True = real hit.
            track_parameters: (B, T, 6) ACTS bound perigee parameters (d0, z0, phi, theta, q/p, t) w.r.t. the
                beam spot, lengths in the units given by track_parameter_length_scale (ACTS: mm); NaN rows =
                no track. Not with is_cosmics.
            track_weight: (B, T) weight of each given track (default 1); only with track_parameters.

        Returns:
            dict, the PV (with find_secondary: the refit where used_refit, else the original fit):
                vertex_estimate (B, 3), chi_square (B,), fit_is_valid (B,); NaN where not valid.
                track_closest_approach_point (B, T, 3), track_dca (B, T): each track to the PV.
                track_parameters (B, T, 6): perigee parameters in cm; NaN without a fit. t = 0 from the hit fit;
                    with use_acts, ACTS's c t from the innermost-hit state (t = 0) back to the perigee (< 0).
                acts_failed (B, T), with use_acts: fitted tracks ACTS failed on (dropped).
                track_position, track_direction (B, T, 3): perigee point and direction there.
                track_weight, track_quality_mask (B, T): fit weight, pT / beam-line cut.
            With find_secondary also (NaN / False / 0 where sv_is_valid is False):
                sv_estimate (B, K, 3); sv_is_valid, sv_chi2, sv_chi2_per_dof, sv_displacement,
                    sv_n_tracks (B, K); sv_track_mask, sv_track_dca (B, K, T).
                displaced_mask, displaced_weight (B, T): SV candidates and their weight.
                pv_original: dict, the original PV fit.
                used_refit (B,): the reported PV is the refit.
        """
        if self.is_cosmics and self.use_acts:
            raise NotImplementedError("use_acts is not supported with is_cosmics")
        acts_failed = None
        if track_parameters is None:
            # track model: hits in cm (padded hits are -100 in every column, which overflows sinh(eta); any
            # finite value will do), hard hit assignment, then each track's circle and z line
            hits, hit_r = _unnormalize_hits(points.masked_fill(~padding_mask.unsqueeze(-1), 0.0), self.normalization)
            innermost_hit, total_hit_weight, is_assigned = _track_position_from_hits(hits, hit_r, mask_probs, padding_mask)
            params = _helix_from_hits(hits, hit_r, is_assigned, self.noise_pt_threshold, Bz=self.Bz,
                                      scaling_factor=self.scaling_factor, valid_radius=self.valid_radius,
                                      min_hits=self.min_hits, is_cosmics=self.is_cosmics,
                                      beam_position=self.beam_position)
            # the helix fit runs in the dtype of the hits
            params.update({k: params[k].to(hits.dtype) for k in ("theta", "circle_centre", "R_s", "zslope", "z0")})
            theta = params["theta"].double()
            q = torch.sign(params["R_s"].double()) * (1.0 if self.Bz >= 0 else -1.0)
            if self.use_acts:
                # ACTS builds the perigee: the fitted helix's state nearest the innermost hit (in mm), propagated
                # back to the beam spot; then used as given track_parameters; tracks ACTS fails on are dropped
                centre, R_s = params["circle_centre"].double(), params["R_s"].double()
                n = innermost_hit[..., :2].double() - centre
                n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                start_xy = centre + R_s.abs().unsqueeze(-1) * n
                start_z = (_arc_length(start_xy[..., 0], start_xy[..., 1], centre[..., 0], centre[..., 1], R_s,
                                       self.beam_position) * params["zslope"].double() + params["z0"].double())
                theta = torch.atan2(torch.ones_like(theta), params["zslope"].double())
                direction = torch.cat([torch.sin(theta).unsqueeze(-1) * torch.sign(R_s).unsqueeze(-1)
                                       * torch.stack([n[..., 1], -n[..., 0]], dim=-1), torch.cos(theta).unsqueeze(-1)], dim=-1)
                acts_params, acts_ok = _acts_worker(self.acts_worker_command(), self.acts_worker_config()).propagate(
                    (torch.cat([start_xy, start_z.unsqueeze(-1)], dim=-1) / self.track_parameter_length_scale
                     ).reshape(-1, 3).cpu().numpy(), direction.reshape(-1, 3).cpu().numpy(),
                    (q * torch.sin(theta) / params["pT"]).reshape(-1).cpu().numpy(), [True] * q.numel())
                track_parameters = torch.from_numpy(acts_params).reshape(*q.shape, 6).to(hits.device, total_hit_weight.dtype)
                acts_failed = params["fit_ok"] & ~torch.from_numpy(acts_ok).reshape(q.shape).to(hits.device)
                track_weight = total_hit_weight
            else:
                perigee = _perigee_from_circle(params["circle_centre"].double(), params["R_s"].double(),
                                               params["zslope"].double(), params["z0"].double(),
                                               q * torch.sin(theta) / params["pT"], self.beam_position)
                if self.is_cosmics:      # z line against x: the perigee z is the line at the perigee x
                    pca_x = self.beam_position[0] - perigee[..., 0] * torch.sin(perigee[..., 2])
                    perigee[..., 1] = params["zslope"].double() * pca_x + params["z0"].double()
                    perigee[..., 3] = theta
                base_weight = class_probs[..., 1] * total_hit_weight
        if track_parameters is not None:
            if self.is_cosmics:
                raise NotImplementedError("track_parameters input is not supported with is_cosmics")
            perigee = track_parameters.double().clone()
            perigee[..., [0, 1, 5]] = perigee[..., [0, 1, 5]] * self.track_parameter_length_scale
            fit_ok = torch.isfinite(perigee[..., :5]).all(dim=-1)
            circle_centre, R_s, zslope, z0, pT, beam_dca = _circle_from_perigee(
                perigee, self.Bz, self.scaling_factor, self.beam_position)
            dtype = track_parameters.dtype
            params = {"circle_centre": circle_centre.to(dtype), "R_s": R_s.to(dtype), "zslope": zslope.to(dtype),
                      "z0": z0.to(dtype), "pT": pT, "beam_dca": beam_dca, "fit_ok": fit_ok,
                      "usable": fit_ok & (pT >= self.noise_pt_threshold) & (beam_dca < self.valid_radius)}
            if class_probs is None:
                class_probs = torch.stack([torch.zeros_like(pT), torch.ones_like(pT)], dim=-1).to(dtype)
            base_weight = class_probs[..., 1] * (torch.ones_like(pT).to(dtype) if track_weight is None else track_weight)
        dtype = base_weight.dtype
        perigee = perigee.masked_fill(~params["fit_ok"].unsqueeze(-1), float("nan"))

        # straight line of each track at its perigee; tracks without a fit get a finite stand-in (weight 0)
        d0, z0, phi, theta = perigee[..., 0], perigee[..., 1], perigee[..., 2], perigee[..., 3]
        track_position = torch.stack([self.beam_position[0] - d0 * torch.sin(phi),
                                      self.beam_position[1] + d0 * torch.cos(phi), z0], dim=-1)
        track_direction = torch.stack([torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi),
                                       torch.cos(theta)], dim=-1)
        fit_ok = params["fit_ok"].unsqueeze(-1)
        track_position = torch.where(fit_ok, track_position, 0.0).to(dtype)
        track_direction = torch.where(fit_ok, track_direction, track_direction.new_tensor([1.0, 0.0, 0.0])).to(dtype)

        if self.learn_weights:
            # features: class_probs (2) + (q/(pT+1), theta, sin phi, cos phi) at the perigee; filler without a fit
            pT = params["pT"].nan_to_num(0.0)
            q = torch.sign(perigee[..., 4]).nan_to_num(1.0)
            track_features = torch.stack([q / (pT + 1.0), theta.nan_to_num(torch.pi / 2), torch.sin(phi).nan_to_num(0.0),
                                          torch.cos(phi).nan_to_num(1.0)], dim=-1).to(dtype)
            learned_weight_scale = self.track_weight_network(torch.cat([class_probs, track_features], dim=-1)).squeeze(-1)
            track_weight = base_weight * learned_weight_scale
        else:
            track_weight = base_weight

        usable = params["usable"].to(dtype=track_weight.dtype)
        if self.learn_weights:
            # straight-through gates: the hard cuts forward, sigmoids of the learnable thresholds backward
            # (unfitted tracks: pT -> 0 and beam_dca -> 1e6 cm, finite so the gradient stays 0, not NaN)
            pT, beam_dca = params["pT"].nan_to_num(0.0), params["beam_dca"].nan_to_num(1e6)
            soft = (torch.sigmoid((pT.to(track_weight.dtype) - self.pt_threshold) / self.log_gate_width_pt.exp())
                    * torch.sigmoid((self.beam_radius - beam_dca.to(track_weight.dtype)) / self.log_gate_width_r.exp()))
            hard = ((pT >= self.pt_threshold.detach()) & (beam_dca < self.beam_radius.detach())).to(track_weight.dtype)
            usable = hard + soft - soft.detach()
        track_weight = track_weight * usable

        fit = self._fit(track_position, track_direction, track_weight, params)

        result = {
            "track_parameters": perigee.to(dtype),
            "track_position": track_position,
            "track_direction": track_direction,
            "track_weight": track_weight,
            "track_quality_mask": params["usable"],
        }
        if acts_failed is not None:
            result["acts_failed"] = acts_failed
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
