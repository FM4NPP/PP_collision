"""
vertexhead.py: primary- and secondary-vertex fit from track-finder output.

Tracks: a helix fitted to each track's hits (Taubin circle, z against arc length), as ACTS perigee parameters
(d0, z0, phi, theta, q/p, t) in cm with a covariance from the hit errors; with use_acts, ACTS propagates them to
the beam spot through the detector (scripts/acts).
PV: covariance-weighted adaptive vertex fit (ACTS's algorithm: annealed weights from each track's chi2).
SVs (find_secondary): displaced tracks are paired into seeds, each seed is fitted like the PV, and the PV is
refitted with SV tracks shared between vertices.

Inputs: class_probs (B, T, 2), mask_probs (B, N, T), normalized points (B, N, 4), padding_mask (B, N);
B events, N hits, T track slots, K = max_sv_per_event. Positions in cm.
"""
import atexit
import inspect
import json
import math
import os
import struct
import subprocess
import sys
import time

import torch
import torch.nn as nn

# ------------------------------------------------------------------------------------------------
# Track model: hit normalization, hard hit-to-track assignment, and each track's circle and z line
# ------------------------------------------------------------------------------------------------

# TPCBatchDataset's hit normalization (fm4npp/datasets/dataset.py): eta, phi, r (cm) min-max, E z-scored
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
    for iteration in range(newton_iters):
        Dy = A1 + xk * (2 * A2 + 3 * A3 * xk)
        xnew = xk - yk / torch.where(Dy.abs() < eps, eps, Dy)
        ynew = A0 + xnew * (A1 + xnew * (A2 + xnew * A3))
        better = (ynew.abs() < yk.abs()) & torch.isfinite(xnew) & improving_any
        xk, yk = torch.where(better, xnew, xk), torch.where(better, ynew, yk)
        improving_any = improving_any & better
        # stop once no track improves (tested every 4 iterations: each test waits for a GPU)
        if iteration % 4 == 3 and not improving_any.any():
            break

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

    # each real hit goes to its most probable track slot (none if all probabilities are 0)
    track = mask_probs.argmax(dim=-1)
    assigned = padding_mask & (mask_probs.max(dim=-1).values > 0)
    is_assigned = (torch.arange(n_tracks, device=track.device) == track.unsqueeze(-1)) & assigned.unsqueeze(-1)

    innermost = _extreme_hit(hit_r, track, assigned, n_tracks)
    track_position = points[..., 1:4].gather(1, innermost.unsqueeze(-1).expand(-1, -1, 3))
    track_position = track_position.masked_fill(~is_assigned.any(dim=1).unsqueeze(-1), 0.0)
    return track_position, total_hit_weight, is_assigned

def _helix_from_hits(points, hit_r, is_assigned, noise_pt_threshold, Bz=1.4, scaling_factor=100.0,
                     valid_radius=1.0, min_hits=3, beam_position=(0.0, 0.0), field=None):
    """
    Each track's helix (circle and z line) and quality labels from its assigned hits.

    Travel from the innermost to the outermost hit gives the charge and phi at the innermost hit:
    centre = innermost hit + R_s (sin phi, -cos phi), sign(R_s) = sign(q Bz).

    Args:
        points, hit_r, is_assigned: as from _track_position_from_hits.
        noise_pt_threshold: GeV; lower fitted pT -> not usable.
        Bz: Tesla (its sign gives the charge); field: (B, T) each track's |Bz| for its pT (a field map), else |Bz|.
        scaling_factor: converts metres to the units of points.
        valid_radius: cm; the circle must pass within this of the beam line to be usable.
        min_hits: fewer assigned hits -> not fitted.
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

    # a helix is linear in arc length: fit z against the signed arc length from the beam-line closest approach
    s = _arc_length(x, y, xc.gather(1, track), yc.gather(1, track),
                                         (sign_Rs * radius).gather(1, track), beam_position)
    zslope, z0 = _line_fit(s, z, track, assigned, n_tracks, min_hits=min_hits)
    ux, uy = (xc - xin) / radius, (yc - yin) / radius
    sin_phi, cos_phi = sign_Rs * ux, -sign_Rs * uy
    q = sign_Rs * (1.0 if Bz >= 0 else -1.0)
    # pT [GeV] = c [1e9 m/s] * B [T] * R [m]; B: each track's field (map) or the uniform |Bz|
    pT = 0.299792458 * (abs(Bz) if field is None else field) * radius / scaling_factor
    # cot(theta) = dz/ds, the arc-length slope
    theta = torch.atan2(torch.ones_like(zslope), zslope)
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

def _track_covariance(s, track, assigned, n_tracks, theta, qop, s_reference, sigma_rphi, sigma_z, Bz=1.4,
                      scaling_factor=100.0, curvilinear=False, min_hits=3):
    """
    Covariance (B, T, 5, 5) of each helix at arc length s_reference, from the hit errors alone (no material):
    linearised least squares on (l0, l1, phi, theta, q/p), exact for a circle. Same axes as ACTS's bound
    covariance (l0 = d0 axis; l1 = z0, or curvilinear loc1).

    Args:
        s: (B, N) hit arc lengths (_arc_length); track, assigned: as in _line_fit.
        theta, qop: (B, T) the helix; s_reference: (B, T) where the covariance is wanted.
        sigma_rphi, sigma_z: per-hit errors (units of s); Bz: T, or (B, T) per track (a field map).
        curvilinear: l1 is curvilinear loc1 instead of z0.

    Returns:
        (B, T, 5, 5); NaN with too few hits or a singular system.
    """
    c = 0.299792458 * (Bz.gather(1, track) if torch.is_tensor(Bz) else Bz) / scaling_factor
    sin_t, cos_t = (f(theta).gather(1, track) for f in (torch.sin, torch.cos))
    ds = (s - s_reference.gather(1, track)).masked_fill(~assigned, 0.0)
    k_qop = c / sin_t                                               # dk / d(q/p)
    k_theta = -c * qop.gather(1, track) * cos_t / sin_t ** 2        # dk / dtheta
    kappa = -c * qop.gather(1, track) / sin_t                       # left curvature
    turn = kappa * ds                                               # turning angle a (left positive)
    sinc = lambda x: torch.sinc(x / torch.pi)                       # noqa: E731 - sin(x) / x
    bend = -0.5 * ds ** 2 * sinc(turn / 2) ** 2                     # d(normal offset) / dk
    # the helix point also slides along the track (tau), moving the hit's z by -cot(theta) tau
    small = turn.abs() < 1e-4
    g = torch.where(small, turn / 6, (1 - sinc(turn)) / torch.where(small, 1.0, turn))
    cot_t = cos_t / sin_t
    zero, one = torch.zeros_like(ds), torch.ones_like(ds)
    sideways = torch.stack([torch.cos(turn), zero, ds * sinc(turn), bend * k_theta, bend * k_qop], dim=-1) / sigma_rphi
    along_z = torch.stack([-cot_t * torch.sin(turn), one / sin_t if curvilinear else one,
                           -cot_t * 0.5 * ds ** 2 * kappa * sinc(turn / 2) ** 2,
                           -ds / sin_t ** 2 + cot_t * ds ** 2 * g * k_theta, cot_t * ds ** 2 * g * k_qop], dim=-1) / sigma_z
    information = (sideways.unsqueeze(-1) * sideways.unsqueeze(-2) + along_z.unsqueeze(-1) * along_z.unsqueeze(-2))
    information = information.masked_fill(~assigned[..., None, None], 0.0).nan_to_num(0.0).flatten(-2)
    information = information.new_zeros(s.shape[0], n_tracks, 25).scatter_add_(
        1, track.unsqueeze(-1).expand(-1, -1, 25), information).unflatten(-1, (5, 5))
    n_hits = _per_track_sum(assigned.to(s.dtype), track, n_tracks)
    eye = torch.eye(5, dtype=s.dtype, device=s.device)
    covariance, info = torch.linalg.inv_ex(torch.where((n_hits >= min_hits)[..., None, None], information, eye))
    bad = (n_hits < min_hits) | (info != 0) | ~torch.isfinite(covariance).all(dim=(-2, -1)) \
        | ~torch.isfinite(theta) | ~torch.isfinite(qop)
    return covariance.masked_fill(bad[..., None, None], float("nan"))


# ------------------------------------------------------------------------------------------------
# ACTS perigee parameters (d0, z0, phi, theta, q/p, t) w.r.t. the beam line: point = beam + d0 (-sin phi, cos phi)
# ------------------------------------------------------------------------------------------------

def _perigee_from_circle(circle_centre, R_s, zslope, z0, qop, beam_position=(0.0, 0.0)):
    """
    Perigee parameters (B, T, 6) of the fitted helix (cm, q/p in e/GeV, t = 0).

    Args:
        circle_centre (B, T, 2), R_s (B, T): circle, sign(R_s) = sign(q Bz).
        zslope, z0: (B, T) z = zslope * s + z0 (zslope = cot theta, z0 = perigee z).
        qop: (B, T); beam_position: (x, y), cm.
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
    Inverse of _perigee_from_circle: circle_centre, R_s, zslope, z0, pT (GeV), beam_dca = |d0|. Bz: T, or (B, T)
    each track's field at its perigee (a field map).
    """
    d0, z0, phi, theta, qop = perigee[..., :5].unbind(-1)
    sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
    pT = torch.sin(theta) / qop.abs()
    if torch.is_tensor(Bz):
        R_s = torch.sign(qop) * torch.sign(Bz) * pT * scaling_factor / (0.299792458 * Bz.abs())
    else:
        R_s = torch.sign(qop) * (1.0 if Bz >= 0 else -1.0) * pT * scaling_factor / (0.299792458 * abs(Bz))
    pca_x, pca_y = beam_position[0] - d0 * sin_phi, beam_position[1] + d0 * cos_phi
    circle_centre = torch.stack([pca_x + R_s * sin_phi, pca_y - R_s * cos_phi], dim=-1)
    return circle_centre, R_s, torch.cos(theta) / torch.sin(theta), z0, pT, d0.abs()

# ------------------------------------------------------------------------------------------------
# Field map: the ACTS text map the detector file names (field.map), for VertexHead's own track maths
# ------------------------------------------------------------------------------------------------

_FIELD_MAPS = {}

class _FieldMap:
    """
    Bz of an ACTS text field map (columns x y z Bx By Bz, or r z Br Bz; mm and T; a full regular grid, as
    convert_field_map.py writes it), interpolated (trilinear, or bilinear for rz) and clamped to the grid.
    """
    def __init__(self, path, kind="xyz"):
        import numpy as np
        table = np.loadtxt(path, comments="#")
        # the stored order, slowest axis first: xyz (x, y, z); rz (z, r)
        columns, bz = ((0, 1, 2), 5) if kind == "xyz" else ((1, 0), 3)
        axes = [np.unique(table[:, c]) for c in columns]
        self.kind = kind
        self.axes = [(float(a[0]), float(a[1] - a[0]), len(a)) for a in axes]
        self.values = torch.from_numpy(table[:, bz].reshape(*(len(a) for a in axes)).copy())

    def bz(self, x, y, z, mm_per_unit=10.0):
        """Bz (T) at points x, y, z (any shape, in units with mm_per_unit mm each); NaN coordinates count as 0."""
        x, y, z = (c.double().nan_to_num() * mm_per_unit for c in (x, y, z))
        coordinates = (x, y, z) if self.kind == "xyz" else (z, torch.hypot(x, y))
        values = self.values.to(x.device)
        lower, fraction = [], []
        for c, (start, step, n) in zip(coordinates, self.axes):
            t = ((c - start) / step).clamp(0, n - 1 - 1e-9)
            lower.append(t.floor().long())
            fraction.append(t - t.floor())
        result = torch.zeros_like(x)
        for corner in range(2 ** len(lower)):
            bits = [(corner >> d) & 1 for d in range(len(lower))]
            weight = math.prod(f if b else 1 - f for f, b in zip(fraction, bits))
            result = result + weight * values[tuple(i + b for i, b in zip(lower, bits))]
        return result

def _field_map(path, kind):
    """The field map at path, loaded once per process."""
    if (path, kind) not in _FIELD_MAPS:
        _FIELD_MAPS[(path, kind)] = _FieldMap(path, kind)
    return _FIELD_MAPS[(path, kind)]

# ------------------------------------------------------------------------------------------------
# ACTS worker (use_acts): one persistent process per configuration, in the ACTS container (scripts/acts)
# ------------------------------------------------------------------------------------------------

ACTS_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  "scripts", "acts", "acts_worker.py")
_ACTS_WORKERS = {}
# seconds spent starting and running ACTS workers (the eval resets and reports it)
ACTS_TIMING = {"start_s": 0.0, "propagate_s": 0.0, "states": 0}

class _ActsWorker:
    """
    One persistent ACTS worker process (scripts/acts/acts_worker.py), started in the background; the first
    propagate waits until it is ready.
    """
    def __init__(self, command, config):
        self.command = command
        self.ready = False
        self.process = subprocess.Popen(list(command) + [ACTS_WORKER_SCRIPT, json.dumps(config)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def wait_ready(self):
        """Waits for the worker's ready line (its start-up output passed to stderr); the wait counts as start time."""
        t0 = time.perf_counter()
        while not self.ready:           # start-up output (e.g. ACTS's setup) until the ready line
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"ACTS worker exited during start-up (code {self.process.wait()}): "
                                   f"{' '.join(self.command)}; its error output is above")
            self.ready = line.strip() == b"ACTS_WORKER_READY"
            if not self.ready:
                sys.stderr.write(line.decode(errors="replace"))
        ACTS_TIMING["start_s"] += time.perf_counter() - t0

    def propagate(self, position, direction, qop, covariance):
        """
        States in mm (position, direction (n, 3), qop (n,), covariance (n, 6, 6)) propagated backward to the
        perigee: parameters (n, 6), ok (n,) and covariance (n, 6, 6).
        """
        import numpy as np
        self.wait_ready()
        t0 = time.perf_counter()
        n = len(qop)
        self.process.stdin.write(struct.pack("<Q", n) + b"".join(
            np.ascontiguousarray(a, "<f8").tobytes() for a in (position, direction, qop, covariance)))
        self.process.stdin.flush()
        reply = self.process.stdout.read(n * 337)
        if len(reply) != n * 337:
            raise RuntimeError(f"ACTS worker stopped (code {self.process.poll()}); its error output is above")
        ACTS_TIMING["propagate_s"] += time.perf_counter() - t0
        ACTS_TIMING["states"] += n
        return (np.frombuffer(reply[: n * 48], "<f8").reshape(n, 6).copy(),
                np.frombuffer(reply[n * 336:], np.uint8).astype(bool),
                np.frombuffer(reply[n * 48: n * 336], "<f8").reshape(n, 6, 6).copy())

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=30)

def _acts_worker(command, config):
    """
    The running worker for this command and config (started on first use, restarted if it died).
    """
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
    track_sigma=None,
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
    # W = U U^T / sigma_d0^2 + V V^T / (sigma_z0 sin theta)^2, V = direction x U; W projector = W
    transverse = track_direction[..., :2].norm(dim=-1).clamp(min=1e-12)
    U = torch.stack([-track_direction[..., 1], track_direction[..., 0], torch.zeros_like(transverse)], dim=-1) \
        / transverse.unsqueeze(-1)
    V = torch.cross(track_direction, U, dim=-1)
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
    track_chi2 = (dca_vector * torch.matmul(P, dca_vector.unsqueeze(-1)).squeeze(-1)).sum(dim=-1)
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

def _track_compatibility(point, track_position, track_direction, track_sigma):
    """
    Compatibility chi2 (B, T) of a point (B, 3) with each track, in the fit's metric.
    """
    r = point.unsqueeze(1) - track_position
    r = r - (r * track_direction).sum(dim=-1, keepdim=True) * track_direction
    transverse = track_direction[..., :2].norm(dim=-1).clamp(min=1e-12)
    U = torch.stack([-track_direction[..., 1], track_direction[..., 0], torch.zeros_like(transverse)], dim=-1) / transverse.unsqueeze(-1)
    V = torch.cross(track_direction, U, dim=-1)
    return ((r * U).sum(dim=-1) / track_sigma[..., 0]) ** 2 + ((r * V).sum(dim=-1) / (track_sigma[..., 1] * transverse)) ** 2

def _density_seed(d0, z0, track_sigma, track_weight, d0_significance=3.5, z0_significance=12.0):
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
            z = z + (y * dy / (dy * dy - y * ddy)).nan_to_num()
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
    n_iterations=100,
    tolerance=1e-4,
    beam_position=(0.0, 0.0),
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
    radius = R_s.abs()
    x0, y0 = circle_centre.unbind(-1)

    result = None
    for _ in range(n_iterations):
        # each circle's nearest point to the vertex and the helix tangent there (anticlockwise <=> R_s < 0)
        n = PV[:, None, :2] - circle_centre
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        pca_circle = circle_centre + radius.unsqueeze(-1) * n
        t_xy = torch.sign(R_s).unsqueeze(-1) * torch.stack([n[..., 1], -n[..., 0]], dim=-1)

        pca_z, dz = _arc_length(pca_circle[..., 0], pca_circle[..., 1], x0, y0, R_s, beam_position) * zslope + z0, zslope
        pca = torch.cat([pca_circle, pca_z.unsqueeze(-1)], dim=-1)
        tangent = torch.cat([t_xy, dz.unsqueeze(-1)], dim=-1)
        # clamped: a stand-in track (centre at the origin) with the vertex at the origin has no direction
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        step = _fit_vertex_by_closest_approach(pca, tangent, track_weight, **linear_fit_kwargs)
        # each event stops at its own convergence (independent of the rest of the batch)
        result = step if result is None else {
            k: torch.where(still_moving.view(-1, *[1] * (v.dim() - 1)), v, result[k]) for k, v in step.items()}
        moved = (step["vertex_estimate"] - PV).norm(dim=-1) > tolerance
        # advance PV only where the fit is valid, so one bad event can't spread NaN
        PV = torch.where((still_moving & step["fit_is_valid"]).unsqueeze(-1), step["vertex_estimate"], PV)
        still_moving = still_moving & step["fit_is_valid"] & moved

        if not still_moving.any():
            break
    return result

class VertexHead(nn.Module):

    """
    PV fit with SV finding (see the module docstring). Detector values are arguments (defaults: sPHENIX).

    Args:
        learn_weights: learnable per-track weight network and soft pT / beam-line gates (start at the plain fit);
            weight_hidden_dim, gate_width_pt (GeV), gate_width_r (cm) configure them.
        Bz: uniform field, T; field_map (+ _kind): ACTS text field map (mm, T) instead, for ACTS and the tracks' pT
            (Bz at their hits, and at the perigee for ACTS tracks); scaling_factor: metres to the units of points.
        noise_pt_threshold (GeV), valid_radius (cm), min_hits: track usability cuts.
        beam_position: (x, y), cm; normalization: the dataset's hit normalization (TPC_NORMALIZATION).
        hit_sigma_rphi, hit_sigma_z: per-hit errors for the track covariance, cm.
        track_error_scale: multiplies the track errors (2.5: PV chi2/ndf about 1).
        helix_iterations, convergence_tolerance (cm), minimum_total_weight: fit limits.
        annealing_temperatures, annealing_cut_off, min_track_weight, adaptive_max_iterations, max_relative_shift:
            the adaptive fit (ACTS's defaults).
        density_seed: the adaptive fit's first weights from compatibility with the track-density seed on the beam line
            (ACTS's Gaussian seeder), not the all-track fit; seed_d0_significance, seed_z0_significance: its cuts.
        seed_z_window, seed_z_window_softness (cm): the PV fit's track weights x sigmoid((window - |z0 - z_seed|) /
            softness), a soft form of ACTS AMVF's 1 mm z preselection; None: off.
        max_vertices: primary vertices to look for (pileup; 1: the PV only); each further one is seeded and fitted on
            the tracks the earlier ones left, kept with >= min_vertex_tracks tracks of fit weight >= 0.5 and unless
            within max_merge_significance sigma in z of an earlier one; the PV is the one with the highest sum of fit
            weight x pT^2.
        use_acts: ACTS builds each track's perigee parameters and covariance (tracks it fails on are dropped);
            acts_image or acts_command, acts_geometry + acts_geometry_config, acts_material_map, acts_energy_loss,
            acts_max_steps configure it.
        track_parameter_length_scale: cm per length unit of forward's track_parameters (ACTS mm: 0.1).
        find_secondary: find SVs; refit_pv: report the PV refitted with SV tracks shared.
        sv_min_track_pv_chi2, sv_min_track_beam_significance: SV candidate cuts.
        max_pair_doca (cm): seed pairs; sv_max_track_chi2: tracks attached to a seed.
        min_sv_tracks, max_sv_chi2_per_dof, sv_min_displacement_significance: SV cuts.
        seed_merge_distance (cm): one SV per group closer than this; max_sv_per_event: K.
    """
    def __init__(
        self,
        learn_weights: bool = False,
        weight_hidden_dim: int = 32,
        helix_iterations: int = 100,
        Bz: float = 1.4,
        scaling_factor: float = 100.0,
        noise_pt_threshold: float = 0.25,
        valid_radius: float = 1.0,
        min_hits: int = 3,
        beam_position: tuple = (0.0, 0.0),
        normalization: dict = None,
        track_parameter_length_scale: float = 0.1,
        use_acts: bool = False,
        acts_image: str = None,
        acts_command: list = None,
        field_map: str = None,
        field_map_kind: str = "xyz",
        acts_max_steps: int = None,
        acts_geometry: str = None,
        acts_geometry_config: str = None,
        acts_material_map: str = None,
        acts_energy_loss: bool = True,
        hit_sigma_rphi: float = 0.034,
        hit_sigma_z: float = 0.1,
        track_error_scale: float = 2.5,
        annealing_temperatures: tuple = (64.0, 16.0, 4.0, 2.0, 1.5, 1.0),
        annealing_cut_off: float = 9.0,
        min_track_weight: float = 1e-4,
        adaptive_max_iterations: int = 30,
        max_relative_shift: float = 0.01,
        density_seed: bool = True,
        seed_d0_significance: float = 3.5,
        seed_z0_significance: float = 12.0,
        seed_z_window: float = 0.2,
        seed_z_window_softness: float = 0.05,
        max_vertices: int = 1,
        min_vertex_tracks: int = 2,
        max_merge_significance: float = 3.0,
        sv_min_track_pv_chi2: float = 9.0,
        sv_min_track_beam_significance: float = 0.0,
        sv_max_track_chi2: float = 9.0,
        sv_min_displacement_significance: float = 3.0,
        convergence_tolerance: float = 1e-4,
        minimum_total_weight: float = 1e-6,
        gate_width_pt: float = 0.05,
        gate_width_r: float = 0.2,
        find_secondary: bool = True,
        max_pair_doca: float = 1.0,
        seed_merge_distance: float = 0.1,
        min_sv_tracks: int = 2,
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
        self.field_map = field_map
        self.field_map_kind = field_map_kind
        self.acts_max_steps = acts_max_steps
        self.acts_geometry = acts_geometry
        self.acts_geometry_config = acts_geometry_config
        self.acts_material_map = acts_material_map
        self.acts_energy_loss = acts_energy_loss
        if (acts_geometry is None) != (acts_geometry_config is None) or (acts_material_map and not acts_geometry):
            raise ValueError("acts_geometry and acts_geometry_config go together; acts_material_map needs both")
        self.hit_sigma_rphi = hit_sigma_rphi
        self.hit_sigma_z = hit_sigma_z
        self.track_error_scale = track_error_scale
        self.annealing_temperatures = tuple(float(t) for t in annealing_temperatures)
        self.annealing_cut_off = annealing_cut_off
        self.min_track_weight = min_track_weight
        self.adaptive_max_iterations = adaptive_max_iterations
        self.max_relative_shift = max_relative_shift
        self.density_seed = density_seed
        self.seed_d0_significance = seed_d0_significance
        self.seed_z0_significance = seed_z0_significance
        self.seed_z_window = seed_z_window
        self.seed_z_window_softness = seed_z_window_softness
        self.max_vertices = max_vertices
        self.min_vertex_tracks = min_vertex_tracks
        self.max_merge_significance = max_merge_significance
        self.sv_min_track_pv_chi2 = sv_min_track_pv_chi2
        self.sv_min_track_beam_significance = sv_min_track_beam_significance
        self.sv_max_track_chi2 = sv_max_track_chi2
        self.sv_min_displacement_significance = sv_min_displacement_significance
        self.minimum_total_weight = minimum_total_weight
        self.find_secondary = find_secondary
        self.max_pair_doca = max_pair_doca
        self.seed_merge_distance = seed_merge_distance
        self.min_sv_tracks = min_sv_tracks
        self.max_sv_chi2_per_dof = max_sv_chi2_per_dof
        self.max_sv_per_event = max_sv_per_event
        self.refit_pv = refit_pv
        # track-model settings
        self.Bz = Bz
        self.scaling_factor = scaling_factor
        self.noise_pt_threshold = noise_pt_threshold
        self.valid_radius = valid_radius
        self.min_hits = min_hits
        self.beam_position = tuple(beam_position)
        self.normalization = dict(TPC_NORMALIZATION if normalization is None else normalization)
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

    @classmethod
    def from_detector(cls, path, **overrides):
        """
        A VertexHead from a detector file (scripts/acts/detectors/<name>/<name>.yaml); keyword arguments override it.
        """
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        base = os.path.dirname(os.path.abspath(path))
        resolve = lambda p: p if p is None else os.path.join(base, os.path.expanduser(p))  # noqa: E731 - absolute stays
        field, acts, hit_errors = d.get("field", {}), d.get("acts", {}) or {}, d.get("hit_errors_cm", {})
        kwargs = {
            "Bz": field.get("bz_tesla", 1.4),
            "beam_position": tuple(d.get("beam_position_cm", (0.0, 0.0))),
            "normalization": d.get("normalization"),
            "hit_sigma_rphi": hit_errors.get("rphi", 0.034),
            "hit_sigma_z": hit_errors.get("z", 0.1),
            "use_acts": bool(acts.get("image") or acts.get("command")),
            "acts_image": resolve(acts.get("image")),
            "acts_command": acts.get("command"),
            "field_map": resolve(field.get("map")),
            "field_map_kind": field.get("map_kind", "xyz"),
            "acts_geometry": resolve(acts.get("geometry")),
            "acts_geometry_config": resolve(acts.get("geometry_config")),
            "acts_material_map": resolve(acts.get("material_map")),
            "acts_energy_loss": acts.get("energy_loss", True),
            "acts_max_steps": acts.get("max_steps"),
            **(d.get("vertex_head") or {}),
        }
        unknown = sorted(set(d.get("vertex_head") or {}) - set(inspect.signature(cls.__init__).parameters))
        if unknown:
            raise ValueError(f"{path}: vertex_head sets {unknown}, which VertexHead does not take (removed in the "
                             "2026-11-05 simplification: vertex_weighting and adaptive_fit, as the covariance-weighted "
                             "adaptive fit is now the only fit)")
        return cls(**{**kwargs, **overrides})

    def start_acts(self):
        """Starts the ACTS worker in the background (with use_acts), so it is ready by the first forward pass."""
        if self.use_acts:
            _acts_worker(self.acts_worker_command(), self.acts_worker_config())

    def acts_worker_command(self):
        """
        The command that runs the ACTS worker (singularity exec, with scripts/acts and the ACTS input directories bound).
        """
        if self.acts_command:
            return list(self.acts_command)
        if not self.acts_image:
            raise ValueError("use_acts needs acts_image (the ACTS container, scripts/acts) or acts_command")
        inputs = [ACTS_WORKER_SCRIPT, self.field_map, self.acts_geometry, self.acts_geometry_config,
                  self.acts_material_map]
        binds = sorted({os.path.dirname(os.path.abspath(p)) for p in inputs if p})
        return ["singularity", "exec", *(["--bind", ",".join(binds)] if binds else []), self.acts_image, "acts-python"]

    def acts_worker_config(self):
        """
        The ACTS worker's settings (mm, T).
        """
        return {"reference": [self.beam_position[0] / self.track_parameter_length_scale,
                              self.beam_position[1] / self.track_parameter_length_scale, 0.0],
                "bz_tesla": self.Bz, "field_map": self.field_map, "field_map_kind": self.field_map_kind,
                "max_steps": self.acts_max_steps, "geometry": self.acts_geometry,
                "geometry_config": self.acts_geometry_config, "material_map": self.acts_material_map,
                "energy_loss": self.acts_energy_loss}

    def track_fit_config(self):
        """The track-model settings."""
        return {k: getattr(self, k) for k in ("Bz", "scaling_factor", "noise_pt_threshold",
                                              "valid_radius", "min_hits", "beam_position", "normalization")}

    def _fit(self, track_position, track_direction, track_weight, params, start=None, single_step=False):
        """
        Vertex fit with these weights: straight-line fit, then helix iterations (or one helix update from start
        with single_step); the straight-line fit when Bz = 0. start: (B, 3), NaN rows / None = straight-line vertex.
        """
        kwargs = dict(minimum_total_weight=self.minimum_total_weight, track_sigma=params["track_sigma"])
        if self.Bz == 0.0 or self.helix_iterations < 1:
            return _fit_vertex_by_closest_approach(track_position, track_direction, track_weight, **kwargs)
        linear = None if single_step else _fit_vertex_by_closest_approach(
            track_position, track_direction, track_weight, **kwargs)
        begin = start if single_step else (linear["vertex_estimate"] if start is None else
                                           torch.where(torch.isfinite(start), start, linear["vertex_estimate"]))
        fit = _fit_vertex_by_helix_closest_approach(
            *params["helix"], track_weight, begin, n_iterations=1 if single_step else self.helix_iterations,
            tolerance=self.convergence_tolerance, beam_position=self.beam_position, **kwargs)
        if linear is not None:
            keep_linear = ~fit["fit_is_valid"]
            fit = {k: torch.where(keep_linear.view(-1, *[1] * (v.dim() - 1)), linear[k], v) for k, v in fit.items()}
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
        index, equilibrium = 0, False
        done, previous_weight = torch.zeros_like(fit["fit_is_valid"]), None    # finished events keep their fit
        for _ in range(self.adaptive_max_iterations):
            temperature = self.annealing_temperatures[index]
            # compatibility at the current vertex -> AnnealingUtility weight (single vertex)
            if competing_chi2 is None:
                vertex_weight = torch.sigmoid((self.annealing_cut_off - fit["track_chi2"]) / (2.0 * temperature)).nan_to_num(0.0)
            else:   # 1 / (1 + exp((chi2 - cutOff)/2T) + sum_k exp((chi2 - chi2_k)/2T))
                chi2 = fit["track_chi2"].unsqueeze(-1)
                vertex_weight = 1.0 / (1.0 + torch.exp((chi2.squeeze(-1) - self.annealing_cut_off) / (2.0 * temperature))
                                       + torch.exp((chi2 - competing_chi2) / (2.0 * temperature)).sum(dim=-1))
                vertex_weight = vertex_weight.nan_to_num(0.0)
            vertex_weight = vertex_weight.masked_fill(vertex_weight <= self.min_track_weight, 0.0).to(track_weight.dtype)
            if previous_weight is not None:
                vertex_weight = torch.where(done.unsqueeze(-1), previous_weight, vertex_weight)
            previous_weight = vertex_weight
            new = self._fit(track_position, track_direction, track_weight * vertex_weight, params,
                            start=fit["vertex_estimate"], single_step=True)
            valid = new["fit_is_valid"]
            dx = (new["vertex_estimate"] - fit["vertex_estimate"]).nan_to_num().unsqueeze(-1)
            # a singular covariance (rare) gives no shift measure: not converged (max iterations still apply)
            shift, singular = torch.linalg.solve_ex(new["vertex_covariance"].nan_to_num(nan=0.0)
                                                    + (~valid).view(-1, 1, 1) * torch.eye(3, dtype=dx.dtype, device=dx.device), dx)
            relative_shift = torch.where(singular == 0, (dx * shift).sum(dim=(-2, -1)), torch.inf)
            update = valid & ~done
            fit = {k: torch.where(update.view(-1, *[1] * (v.dim() - 1)), v, fit[k]) for k, v in new.items()}
            # anneal after each update; after the last temperature each event stops once its shift is small
            if not equilibrium:
                if index < len(self.annealing_temperatures) - 1:
                    index += 1
                else:
                    equilibrium = True
            if equilibrium:
                done = done | (relative_shift <= self.max_relative_shift) | ~valid
                if bool(done.all()):
                    break
        return fit, vertex_weight

    def _find_secondary_vertices(self, pv_fit, track_position, track_direction, track_weight, params, pv_weight=None):
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
                track_position, track_direction, track_weight if pv_weight is None else pv_weight, params,
                competing_chi2=competing, start=pv_fit["vertex_estimate"])
        return result

    def forward(
        self,
        class_probs=None,
        mask_probs=None,
        points=None,
        padding_mask=None,
        track_parameters=None,
        track_weight=None,
        track_covariance=None,
    ):
        """
        Args:
            class_probs: (B, T, 2) P(empty), P(real); optional with track_parameters.
            mask_probs: (B, N, T); points: (B, N, 4) normalized hits; padding_mask: (B, N), True = real.
            track_parameters, track_covariance: (B, T, 6), (B, T, 6, 6) given perigee tracks instead of hits
                (track_parameter_length_scale units); track_weight: (B, T), default 1.

        Returns:
            dict (the PV: the refit where used_refit):
                vertex_estimate (B, 3), chi_square, fit_is_valid (B,), vertex_covariance (B, 3, 3); its quality:
                vertex_n_tracks (B,) tracks with fit weight (annealing x z window) >= 0.5, vertex_ndf (B,) 2 sum(weights) - 3.
                With max_vertices > 1: primary_vertices (B, V, 3), primary_vertex_valid, primary_vertex_n_tracks (B, V),
                pv_index (B,) the PV among them, track_primary_vertex (B, T) each track's vertex (-1: none).
                track_parameters (B, T, 6) cm, track_position, track_direction (B, T, 3), track_weight,
                track_quality_mask, track_sigma (B, T, 2), track_vertex_weight, track_chi2, track_dca (B, T),
                track_closest_approach_point (B, T, 3); acts_failed (B, T) with use_acts: usable by the hit fit, lost by ACTS;
                track_covariance (B, T, 6, 6): the ACTS perigee covariance (mm), with use_acts or given.
                With find_secondary: sv_estimate (B, K, 3), sv_is_valid, sv_chi2, sv_chi2_per_dof, sv_displacement,
                sv_n_tracks (B, K), sv_track_mask, sv_track_dca (B, K, T), displaced_mask, displaced_weight (B, T),
                pv_original (dict), used_refit (B,).
        """
        acts_failed = track_sigma = None
        if track_parameters is None:
            # track model: hits in cm (padding zeroed), hard assignment, each track's circle and z line
            hits, hit_r = _unnormalize_hits(points.masked_fill(~padding_mask.unsqueeze(-1), 0.0), self.normalization)
            innermost_hit, total_hit_weight, is_assigned = _track_position_from_hits(hits, hit_r, mask_probs, padding_mask)
            # with a field map: each track's Bz, the mean over its hits (signed; tracks without hits: Bz)
            track_bz = self.Bz
            if self.field_map is not None:
                hit_bz = _field_map(self.field_map, self.field_map_kind).bz(
                    hits[..., 1], hits[..., 2], hits[..., 3], 1000.0 / self.scaling_factor)
                n_hits = is_assigned.sum(dim=1)
                track_bz = torch.where(n_hits > 0, (is_assigned * hit_bz.unsqueeze(-1)).sum(dim=1) / n_hits.clamp(min=1),
                                       float(self.Bz))
            params = _helix_from_hits(hits, hit_r, is_assigned, self.noise_pt_threshold, Bz=self.Bz,
                                      scaling_factor=self.scaling_factor, valid_radius=self.valid_radius,
                                      min_hits=self.min_hits, beam_position=self.beam_position,
                                      field=track_bz.abs() if torch.is_tensor(track_bz) else None)
            # the helix fit runs in the dtype of the hits
            params.update({k: params[k].to(hits.dtype) for k in ("theta", "circle_centre", "R_s", "zslope", "z0")})
            theta = params["theta"].double()
            q = torch.sign(params["R_s"].double()) * (1.0 if self.Bz >= 0 else -1.0)
            # the hit fit's covariance at a point s_reference along each helix (cm, from the hit errors)
            hit_track, hit_assigned = is_assigned.to(torch.uint8).argmax(dim=-1), is_assigned.any(dim=-1)
            centre, R_s = params["circle_centre"].double(), params["R_s"].double()
            hit_s = _arc_length(hits[..., 1].double(), hits[..., 2].double(), centre[..., 0].gather(1, hit_track),
                                centre[..., 1].gather(1, hit_track), R_s.gather(1, hit_track), self.beam_position)
            fit_theta = torch.atan2(torch.ones_like(theta), params["zslope"].double())
            hit_covariance = lambda s_reference, curvilinear: _track_covariance(  # noqa: E731
                hit_s, hit_track, hit_assigned, R_s.shape[-1], fit_theta, q * torch.sin(fit_theta) / params["pT"],
                s_reference, self.hit_sigma_rphi, self.hit_sigma_z, track_bz, self.scaling_factor, curvilinear,
                self.min_hits)
            if self.use_acts:
                # ACTS propagates the helix state at the innermost hit (mm) back to the beam spot
                centre, R_s = params["circle_centre"].double(), params["R_s"].double()
                n = innermost_hit[..., :2].double() - centre
                n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                start_xy = centre + R_s.abs().unsqueeze(-1) * n
                start_s = _arc_length(start_xy[..., 0], start_xy[..., 1], centre[..., 0], centre[..., 1], R_s,
                                      self.beam_position)
                start_z = start_s * params["zslope"].double() + params["z0"].double()
                theta = torch.atan2(torch.ones_like(theta), params["zslope"].double())
                direction = torch.cat([torch.sin(theta).unsqueeze(-1) * torch.sign(R_s).unsqueeze(-1)
                                       * torch.stack([n[..., 1], -n[..., 0]], dim=-1), torch.cos(theta).unsqueeze(-1)], dim=-1)
                # start covariance (curvilinear, mm; t uncorrelated) for ACTS to transport
                to_mm = q.new_tensor([1.0, 1.0, self.track_parameter_length_scale, self.track_parameter_length_scale,
                                      self.track_parameter_length_scale]) / self.track_parameter_length_scale
                start_covariance = q.new_zeros(*q.shape, 6, 6)
                start_covariance[..., 5, 5] = 1.0
                start_covariance[..., :5, :5] = hit_covariance(start_s, True) * to_mm[:, None] * to_mm[None, :]
                start_covariance = start_covariance.reshape(-1, 6, 6).cpu().numpy()
                acts_out = _acts_worker(self.acts_worker_command(), self.acts_worker_config()).propagate(
                    (torch.cat([start_xy, start_z.unsqueeze(-1)], dim=-1) / self.track_parameter_length_scale
                     ).reshape(-1, 3).cpu().numpy(), direction.reshape(-1, 3).cpu().numpy(),
                    (q * torch.sin(theta) / params["pT"]).reshape(-1).cpu().numpy(), start_covariance)
                track_parameters = torch.from_numpy(acts_out[0]).reshape(*q.shape, 6).to(hits.device, total_hit_weight.dtype)
                acts_failed = params["usable"] & ~torch.from_numpy(acts_out[1]).reshape(q.shape).to(hits.device)
                track_covariance = torch.from_numpy(acts_out[2]).reshape(*q.shape, 6, 6).to(hits.device)
                track_weight = torch.ones_like(total_hit_weight)
            else:
                perigee = _perigee_from_circle(params["circle_centre"].double(), params["R_s"].double(),
                                               params["zslope"].double(), params["z0"].double(),
                                               q * torch.sin(theta) / params["pT"], self.beam_position)
                covariance = hit_covariance(torch.zeros_like(theta), False)
                track_sigma = torch.stack([covariance[..., 0, 0], covariance[..., 1, 1]], dim=-1).sqrt()
                base_weight = class_probs[..., 1].clone()
        if track_parameters is not None:
            perigee = track_parameters.double().clone()
            perigee[..., [0, 1, 5]] = perigee[..., [0, 1, 5]] * self.track_parameter_length_scale
            fit_ok = torch.isfinite(perigee[..., :5]).all(dim=-1)
            # with a field map: Bz at each track's perigee (beam line, z0)
            perigee_bz = self.Bz if self.field_map is None else _field_map(self.field_map, self.field_map_kind).bz(
                torch.full_like(perigee[..., 1], self.beam_position[0]), torch.full_like(perigee[..., 1], self.beam_position[1]),
                perigee[..., 1], 1000.0 / self.scaling_factor)
            circle_centre, R_s, zslope, z0, pT, beam_dca = _circle_from_perigee(
                perigee, perigee_bz, self.scaling_factor, self.beam_position)
            dtype = track_parameters.dtype
            params = {"circle_centre": circle_centre.to(dtype), "R_s": R_s.to(dtype), "zslope": zslope.to(dtype),
                      "z0": z0.to(dtype), "pT": pT, "beam_dca": beam_dca, "fit_ok": fit_ok,
                      "usable": fit_ok & (pT >= self.noise_pt_threshold) & (beam_dca < self.valid_radius)}
            if class_probs is None:
                class_probs = torch.stack([torch.zeros_like(pT), torch.ones_like(pT)], dim=-1).to(dtype)
            base_weight = class_probs[..., 1] * (torch.ones_like(pT).to(dtype) if track_weight is None else track_weight)
            if track_covariance is None:
                raise ValueError("track_parameters need their track_covariance")
            track_sigma = torch.stack([track_covariance[..., 0, 0], track_covariance[..., 1, 1]], dim=-1
                                      ).double().sqrt() * self.track_parameter_length_scale
        dtype = base_weight.dtype
        # tracks without a usable covariance get no weight (and a finite stand-in error)
        has_sigma = torch.isfinite(track_sigma).all(dim=-1) & (track_sigma > 0).all(dim=-1)
        base_weight = base_weight * has_sigma.to(dtype)
        params["track_sigma"] = torch.where(has_sigma.unsqueeze(-1), track_sigma * self.track_error_scale, 1.0).to(dtype)
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
            # straight-through gates: hard cuts forward, sigmoid gradients backward
            pT, beam_dca = params["pT"].nan_to_num(0.0), params["beam_dca"].nan_to_num(1e6)
            soft = (torch.sigmoid((pT.to(track_weight.dtype) - self.pt_threshold) / self.log_gate_width_pt.exp())
                    * torch.sigmoid((self.beam_radius - beam_dca.to(track_weight.dtype)) / self.log_gate_width_r.exp()))
            hard = ((pT >= self.pt_threshold.detach()) & (beam_dca < self.beam_radius.detach())).to(track_weight.dtype)
            usable = hard + soft - soft.detach()
        track_weight = track_weight * usable

        # finite stand-ins for unfitted tracks (weight 0; their outputs are masked below)
        bad = ~params["fit_ok"]
        params["helix"] = (params["circle_centre"].masked_fill(bad.unsqueeze(-1), 0.0), params["R_s"].masked_fill(bad, 1.0),
                           params["zslope"].masked_fill(bad, 0.0), params["z0"].masked_fill(bad, 0.0))
        # primary vertices, as ACTS AMVF iterates: seed, z window and adaptive fit on the tracks not yet used, up to
        # max_vertices; a later vertex needs min_vertex_tracks tracks of fit weight >= 0.5, and the tracks it uses
        # leave the pool (weight x (1 - fit weight))
        candidates, available = [], track_weight
        for k in range(self.max_vertices):
            seed, window = self._seed_and_window(d0, z0, params["track_sigma"], available, dtype)
            vertex_fit, annealing = self._fit_adaptive(track_position, track_direction, available * window, params,
                                                       seed=seed)
            fit_weight = (annealing * window).masked_fill(bad | (track_weight <= 0), 0.0)
            valid = vertex_fit["fit_is_valid"]
            if k > 0:
                valid = valid & ((fit_weight >= 0.5).sum(dim=-1) >= self.min_vertex_tracks)
                valid = valid if seed is None else valid & torch.isfinite(seed).all(dim=-1)
                # a duplicate of an earlier vertex (z within max_merge_significance sigma, ACTS) is not a new one
                for earlier in candidates:
                    dz = vertex_fit["vertex_estimate"][:, 2] - earlier[0]["vertex_estimate"][:, 2]
                    sigma = (vertex_fit["vertex_covariance"][:, 2, 2] + earlier[0]["vertex_covariance"][:, 2, 2]).sqrt()
                    valid = valid & ~(earlier[6] & (dz.abs() < self.max_merge_significance * sigma))
            candidates.append((vertex_fit, annealing, seed, window, available * window, fit_weight, valid))
            # the candidate's tracks leave the pool whether or not it is kept, so its seed is not found again
            used = vertex_fit["fit_is_valid"].unsqueeze(-1).to(fit_weight.dtype)
            available = available * (1 - fit_weight * used).clamp(min=0.0)
        # the PV: one vertex, or the valid one with the highest sum of fit weight x pT^2
        pv_index = torch.zeros(track_weight.shape[0], dtype=torch.long, device=track_weight.device)
        if len(candidates) > 1:
            pT2 = params["pT"].nan_to_num(0.0).to(dtype) ** 2
            score = torch.stack([(c[5] * pT2).sum(dim=-1).masked_fill(~c[6], -torch.inf) for c in candidates], dim=1)
            pv_index = score.argmax(dim=1)
        pick = lambda values: values[0] if len(candidates) == 1 else torch.stack(values, dim=1)[  # noqa: E731
            torch.arange(len(pv_index), device=pv_index.device), pv_index]
        fit = {key: pick([c[0][key] for c in candidates]) for key in candidates[0][0]}
        vertex_weight, window, pv_weight = (pick([c[i] for c in candidates]) for i in (1, 3, 4))
        seed = None if candidates[0][2] is None else pick([c[2] for c in candidates])
        # SV finding keeps track_weight, minus the tracks of the other primary vertices
        sv_weight = track_weight
        if len(candidates) > 1:
            others = torch.stack([c[5] * c[6].unsqueeze(-1) for c in candidates], dim=1)
            others = others.masked_fill(torch.nn.functional.one_hot(pv_index, len(candidates)).bool().unsqueeze(-1), 0.0)
            sv_weight = track_weight * (1 - others.max(dim=1).values)

        result = {
            "track_parameters": perigee.to(dtype),
            "track_position": track_position,
            "track_direction": track_direction,
            "track_weight": track_weight,
            "track_quality_mask": params["usable"],
        }
        if acts_failed is not None:
            result["acts_failed"] = acts_failed
        if track_covariance is not None:
            result["track_covariance"] = track_covariance
        if seed is not None:
            result["pv_seed"] = seed
        result["track_sigma"] = params["track_sigma"]
        # per-track outputs of unfitted tracks (bad) come from their finite stand-ins: NaN in what is reported
        def masked(f):
            return {**f, "track_dca": f["track_dca"].masked_fill(bad, float("nan")),
                    "track_chi2": f["track_chi2"].masked_fill(bad, float("nan")),
                    "track_closest_approach_point": f["track_closest_approach_point"].masked_fill(
                        bad.unsqueeze(-1), float("nan"))}

        if self.find_secondary:
            secondary = self._find_secondary_vertices(fit, track_position, track_direction, sv_weight, params, pv_weight)
            refit = secondary.pop("pv_refit", None)
            refit_vertex_weight = secondary.pop("pv_refit_vertex_weight", None)
            secondary["sv_track_dca"] = secondary["sv_track_dca"].masked_fill(bad.unsqueeze(1), float("nan"))
            result.update(secondary)
            result["pv_original"] = masked(fit)
            # the refit PV where the event has an SV and the refit is valid; else the original PV fit
            used_refit = secondary["sv_is_valid"].any(dim=1)
            if refit is not None:
                used_refit = used_refit & refit["fit_is_valid"]
                fit = {k: torch.where(used_refit.view(-1, *[1] * (v.dim() - 1)), refit[k], v) for k, v in fit.items()}
                vertex_weight = torch.where(used_refit.unsqueeze(-1), refit_vertex_weight, vertex_weight)
            else:
                used_refit = torch.zeros_like(used_refit)
            result["used_refit"] = used_refit
        fit = masked(fit)
        # vertex quality, as ACTS judges it: each fitted track's weight in the fit, its annealing weight x the z window
        # (tracks with track_weight > 0)
        weight = (vertex_weight * window).masked_fill(bad | (track_weight <= 0), 0.0)
        result.update({
            "vertex_estimate": fit["vertex_estimate"],
            "chi_square": fit["chi_square"],
            "fit_is_valid": fit["fit_is_valid"],
            "track_closest_approach_point": fit["track_closest_approach_point"],
            "track_dca": fit["track_dca"],
            "track_vertex_weight": vertex_weight.masked_fill(bad, 0.0),
            "track_chi2": fit["track_chi2"],
            "vertex_covariance": fit["vertex_covariance"],
            "vertex_n_tracks": (weight >= 0.5).sum(dim=-1),
            "vertex_ndf": 2 * weight.sum(dim=-1) - 3,
        })
        if len(candidates) > 1:
            # all primary vertices (pileup), each track's (fit weight >= 0.5, else -1) and which one is the PV
            valid = torch.stack([c[6] for c in candidates], dim=1)
            fit_weight = torch.stack([c[5] for c in candidates], dim=1) * valid.unsqueeze(-1).to(dtype)
            best, index = fit_weight.max(dim=1)
            result.update({
                "primary_vertices": torch.stack([c[0]["vertex_estimate"] for c in candidates], dim=1).masked_fill(
                    ~valid.unsqueeze(-1), float("nan")),
                "primary_vertex_valid": valid,
                "primary_vertex_n_tracks": (fit_weight >= 0.5).sum(dim=-1).masked_fill(~valid, 0),
                "pv_index": pv_index,
                "track_primary_vertex": torch.where(best >= 0.5, index, -1),
            })
        return result

    def _seed_and_window(self, d0, z0, track_sigma, track_weight, dtype):
        """
        The density seed (B, 3) on the beam line (None without density_seed) and each track's z-window factor (B, T)
        (1 without a window or a seed).
        """
        seed, window = None, torch.ones_like(track_weight)
        if self.density_seed:
            z = _density_seed(d0, z0, track_sigma, track_weight, self.seed_d0_significance, self.seed_z0_significance).to(dtype)
            seed = torch.stack([torch.full_like(z, self.beam_position[0]), torch.full_like(z, self.beam_position[1]), z], -1)
            if self.seed_z_window is not None:
                fade = torch.sigmoid((self.seed_z_window - (z0.to(dtype) - seed[:, 2:]).abs()) / self.seed_z_window_softness)
                window = torch.where(torch.isfinite(seed[:, 2:]), fade.nan_to_num(0.0), 1.0)
        return seed, window
