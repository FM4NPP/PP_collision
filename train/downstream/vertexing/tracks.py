"""
tracks.py: TrackBuilder, the tracks for the vertex fit, from the track finder's output.

A helix fitted to each track's hits (Taubin circle, z against arc length), as ACTS perigee parameters
(d0, z0, phi, theta, q/p, t) in cm with a covariance from the hit errors; with use_acts, ACTS propagates the
tracks to the beam line through the detector, refitting their hits first with acts_kalman_refit (acts.py,
scripts/acts).
"""
import functools
import math
import os

import torch
import torch.nn as nn

from .acts import ACTS_WORKER_SCRIPT, _acts_worker
from .common import C_LIGHT, CM_PER_M, MM_PER_CM, _arc_length, _circle_point

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

def _centre_on_tracks(values, track, assigned, n_tracks, min_hits, eps):
    """
    Each hit's values (a list of (B, N), unassigned hits ignored) centred on its own track's means: the centred values
    (0 off the tracks), the means (B, T), each track's hit count (B, T, at least eps) and count >= min_hits.
    """
    mask = assigned.to(values[0].dtype)
    n = _per_track_sum(mask, track, n_tracks).clamp(min=eps)
    values = [v.masked_fill(~assigned, 0.0) for v in values]          # unassigned hits can't add NaN
    means = [_per_track_sum(v, track, n_tracks) / n for v in values]
    return [(v - m.gather(1, track)) * mask for v, m in zip(values, means)], means, n, n >= min_hits

def _line_fit(u, v, track, assigned, n_tracks, min_hits, eps=1e-12):
    """
    Deming (orthogonal-distance) line fit v = slope * u + intercept, per track.

    Args:
        u, v: (B, N) per hit; track: (B, N) each hit's track slot; assigned: (B, N) bool, the hit is on
            that track; n_tracks: T.

    Returns:
        slope, intercept: (B, T); NaN with fewer than min_hits hits.
    """
    (du, dv), (mean_u, mean_v), _, valid = _centre_on_tracks([u, v], track, assigned, n_tracks, min_hits, eps)

    ssd_u  = _per_track_sum(du * du, track, n_tracks)
    ssd_v  = _per_track_sum(dv * dv, track, n_tracks)
    ssd_uv = _per_track_sum(du * dv, track, n_tracks)

    slope = ((ssd_v - ssd_u + torch.sqrt((ssd_v - ssd_u) ** 2 + 4 * ssd_uv ** 2)) / 2.0 /
              torch.where(ssd_uv.abs() < eps, eps, ssd_uv))
    intercept = mean_v - slope * mean_u

    slope = slope.masked_fill(~valid, float("nan"))
    intercept = intercept.masked_fill(~valid, float("nan"))
    return slope, intercept

def _circle_fit(x, y, track, assigned, n_tracks, min_hits, newton_iters=20, eps=1e-12):
    """
    Taubin circle fit per track, in float64.

    Args:
        x, y: (B, N) per hit; track: (B, N) each hit's track slot; assigned: (B, N) bool, the hit is on
            that track; n_tracks: T.

    Returns:
        radius, x_centre, y_centre: (B, T) float64; NaN with fewer than min_hits hits.
    """
    (Xi, Yi), (mean_x, mean_y), n_valid, valid = _centre_on_tracks([x.double(), y.double()], track, assigned, n_tracks,
                                                                  min_hits, eps)
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
    ties, and 0 for a track with none.
    """
    n_events, n_hits = r.shape
    key = torch.where(assigned, r if outermost else -r, -torch.inf)
    best = key.new_full((n_events, n_tracks), -torch.inf).scatter_reduce_(1, track, key, "amax")
    hit = torch.arange(n_hits, device=r.device).expand_as(track)
    first = torch.full((n_events, n_tracks), n_hits, dtype=torch.long, device=r.device).scatter_reduce_(
        1, track, torch.where(assigned & (key == best.gather(1, track)), hit, n_hits), "amin")
    return first.masked_fill(first == n_hits, 0)

def _unnormalize_hits(points, normalization):
    """
    Invert the dataset's hit normalization (z-scored E; eta, phi, r min-max scaled to [0, 1]).

    Args:
        normalization: dict eta_lim, phi_lim, r_lim (cm), E_mean, E_std (e.g. TPC_NORMALIZATION).

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

def _helix_from_hits(points, hit_r, track, assigned, n_tracks, Bz, min_hits, beam_position):
    """
    Each track's helix (circle and z line) from its assigned hits.

    Travel from the innermost to the outermost hit gives the charge and phi at the innermost hit:
    centre = innermost hit + R_s (sin phi, -cos phi), sign(R_s) = sign(q Bz).

    Args:
        points: (B, N, 4) (E, x, y, z) cm; hit_r: (B, N); track: (B, N) each hit's track slot (0 if unassigned);
            assigned: (B, N) the hit is on that track; n_tracks: T.
        Bz: T, or (B, T) each track's field (a field map); its size gives pT, its sign the charge.
        min_hits: fewer assigned hits -> not fitted.
        beam_position: (x, y) of the beam line, cm; arc length and beam_dca are measured from it.

    Returns:
        dict, float64, NaN for tracks without a fit:
            pT, theta, qop, beam_dca (closest approach of the circle to the beam line) (B, T).
            circle_centre (B, T, 2), R_s, zslope, z0 (B, T): the circle and z line z = zslope * s + z0
                (s from _arc_length); hit_s (B, N) each hit's s on its track.
            innermost (B, T): index of each track's innermost hit (0 for a track with none).
    """
    x, y, z = (points[..., k] for k in (1, 2, 3))
    r = hit_r
    Bz = torch.as_tensor(Bz, dtype=torch.float64, device=x.device)

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
    q = sign_Rs * torch.sign(Bz)
    pT = C_LIGHT * Bz.abs() * radius / CM_PER_M
    # cot(theta) = dz/ds, the arc-length slope
    theta = torch.atan2(torch.ones_like(zslope), zslope)
    dx, dy = xc - beam_position[0], yc - beam_position[1]
    beam_dca = (torch.sqrt(dx * dx + dy * dy) - radius).abs()

    fit_ok = torch.isfinite(torch.stack([pT, theta, xc, yc], dim=-1)).all(dim=-1) & (radius > 0)
    nan = torch.full_like(pT, float("nan"))
    pT, theta, beam_dca, R_s, zslope, z0 = (torch.where(fit_ok, v, nan)
                                            for v in (pT, theta, beam_dca, sign_Rs * radius, zslope, z0))

    return {
        "pT": pT, "theta": theta, "qop": q * torch.sin(theta) / pT, "beam_dca": beam_dca,
        "circle_centre": torch.where(fit_ok.unsqueeze(-1), torch.stack([xc, yc], dim=-1), nan.unsqueeze(-1)),
        "R_s": R_s, "zslope": zslope, "z0": z0, "hit_s": s, "innermost": i_in,
    }

def _track_covariance(s, track, assigned, n_tracks, theta, qop, s_reference, sigma_rphi, sigma_z, Bz, curvilinear,
                      min_hits):
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
    c = C_LIGHT * torch.as_tensor(Bz, dtype=s.dtype, device=s.device).expand(s.shape[0], n_tracks).gather(1, track) / CM_PER_M
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

def _perigee_from_circle(circle_centre, R_s, theta, z0, qop, beam_position):
    """
    Perigee parameters (B, T, 6) of the fitted helix (cm, q/p in e/GeV, t = 0).

    Args:
        circle_centre (B, T, 2), R_s (B, T): circle, sign(R_s) = sign(q Bz).
        theta, z0: (B, T) polar angle and perigee z.
        qop: (B, T); beam_position: (x, y), cm.
    """
    bx, by = beam_position
    ux, uy = bx - circle_centre[..., 0], by - circle_centre[..., 1]
    norm = torch.sqrt(ux * ux + uy * uy).clamp(min=1e-12)
    ux, uy = ux / norm, uy / norm
    h = torch.sign(R_s)
    phi = torch.atan2(-h * ux, h * uy)                         # heading at the perigee, along the direction of travel
    d0 = h * (R_s.abs() - norm)                                # the closest point: centre + |R_s| u, u towards the beam
    return torch.stack([d0, z0, phi, theta, qop, torch.zeros_like(d0)], dim=-1)

def _circle_from_perigee(perigee, Bz, beam_position):
    """
    Inverse of _perigee_from_circle: circle_centre, R_s, zslope, z0, pT (GeV), beam_dca = |d0|. Bz: T, or (B, T)
    each track's field at its perigee (a field map).
    """
    d0, z0, phi, theta, qop = perigee[..., :5].unbind(-1)
    sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
    pT = torch.sin(theta) / qop.abs()
    Bz = torch.as_tensor(Bz, dtype=pT.dtype, device=pT.device)
    R_s = torch.sign(qop) * torch.sign(Bz) * pT * CM_PER_M / (C_LIGHT * Bz.abs())
    pca_x, pca_y = beam_position[0] - d0 * sin_phi, beam_position[1] + d0 * cos_phi
    circle_centre = torch.stack([pca_x + R_s * sin_phi, pca_y - R_s * cos_phi], dim=-1)
    return circle_centre, R_s, torch.cos(theta) / torch.sin(theta), z0, pT, d0.abs()

# ------------------------------------------------------------------------------------------------
# Field map: the ACTS text map the detector file names (field.map), for TrackBuilder's own track maths
# ------------------------------------------------------------------------------------------------

class _FieldMap:
    """
    Bz of an ACTS text field map (columns x y z Bx By Bz, or r z Br Bz; mm and T; a full regular grid, as
    convert_field_map.py writes it), interpolated (trilinear, or bilinear for rz) and clamped to the grid.
    """
    def __init__(self, path, kind):
        import numpy as np
        if kind not in ("xyz", "rz"):
            raise ValueError(f"field map kind {kind!r}: xyz or rz")
        table = np.loadtxt(path, comments="#", ndmin=2)
        if table.shape[1] != (6 if kind == "xyz" else 4):
            raise ValueError(f"{path}: {table.shape[1]} columns, a {kind} map has {6 if kind == 'xyz' else 4}")
        # the stored order, slowest axis first: xyz (x, y, z); rz (z, r)
        columns, bz = ((0, 1, 2), 5) if kind == "xyz" else ((1, 0), 3)
        axes = [np.unique(table[:, c]) for c in columns]
        if any(len(a) < 2 or not np.allclose(np.diff(a), a[1] - a[0]) for a in axes) or table.shape[0] != math.prod(
                len(a) for a in axes):
            raise ValueError(f"{path}: not a full regular grid")
        self.kind = kind
        self.axes = [(float(a[0]), float(a[1] - a[0]), len(a)) for a in axes]
        self.values = torch.from_numpy(table[:, bz].reshape(*(len(a) for a in axes)).copy())

    def bz(self, x, y, z, mm_per_unit):
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

@functools.lru_cache(maxsize=None)
def _field_map(path, kind):
    """The field map at path, loaded once per process."""
    return _FieldMap(path, kind)

class TrackBuilder(nn.Module):
    """
    The tracks for the vertex fit: each track's perigee parameters (cm) and errors, its straight line at the perigee,
    usability and weight, from the track finder's hits (a helix fit, optionally through ACTS) or given perigee tracks.
    Detector values are arguments (defaults: sPHENIX).

    Args:
        learn_weights: learnable per-track weight network, soft pT / beam-line gates and the Kalman fallback's
            covariance scale (start at the plain fit); weight_hidden_dim, gate_width_pt (GeV), gate_width_r (cm)
            configure them.
        Bz: uniform field, T; field_map (+ _kind): ACTS text field map (mm, T) instead, for ACTS and the tracks' pT
            (Bz at their hits, and at the perigee for perigee tracks: ACTS's or given); Bz must not be 0.
        noise_pt_threshold (GeV), valid_radius (cm), min_hits: track usability cuts.
        beam_position: (x, y), cm; normalization: the dataset's hit normalization (TPC_NORMALIZATION).
        hit_sigma_rphi, hit_sigma_z: per-hit errors for the track covariance, cm.
        track_error_scale: multiplies the track errors.
        use_acts: ACTS builds each track's perigee parameters and covariance (tracks it fails on are dropped; no
            gradient flows through ACTS: its tracks are constants for training);
            acts_image or acts_command, acts_geometry + acts_geometry_config, acts_material_map, acts_energy_loss,
            acts_max_steps, acts_threads (propagation threads; None: the CPUs it may use), acts_particle_pdg and
            acts_particle_mass (GeV; None: ACTS's mass for the PDG id; the tracks' particle, which sets energy loss and
            multiple scattering in material) configure it. acts_kalman_refit: ACTS's Kalman fitter first refits each
            track's hits (hit_sigma_rphi, hit_sigma_z), from the hit fit's state at the innermost hit with its
            covariance x acts_kalman_start_covariance_scale; where it fails, that state is used as without the refit
            and its perigee covariance x acts_kalman_fallback_covariance_scale (a soft cut).
    """
    def __init__(
        self,
        learn_weights: bool = False,
        weight_hidden_dim: int = 32,
        Bz: float = 1.4,
        noise_pt_threshold: float = 0.25,
        valid_radius: float = 1.0,
        min_hits: int = 3,
        beam_position: tuple = (0.0, 0.0),
        normalization: dict = None,
        use_acts: bool = False,
        acts_image: str = None,
        acts_command: list = None,
        field_map: str = None,
        field_map_kind: str = "xyz",
        acts_max_steps: int = None,
        acts_threads: int = None,
        acts_geometry: str = None,
        acts_geometry_config: str = None,
        acts_material_map: str = None,
        acts_energy_loss: bool = False,
        acts_particle_pdg: int = 211,
        acts_particle_mass: float = None,
        acts_kalman_refit: bool = True,
        acts_kalman_start_covariance_scale: float = 100.0,
        acts_kalman_fallback_covariance_scale: float = 100.0,
        hit_sigma_rphi: float = 0.034,
        hit_sigma_z: float = 0.1,
        track_error_scale: float = 2.5,
        gate_width_pt: float = 0.05,
        gate_width_r: float = 0.2,
    ):
        super().__init__()
        if (acts_geometry is None) != (acts_geometry_config is None) or ((acts_material_map or acts_energy_loss) and not acts_geometry):
            raise ValueError("acts_geometry and acts_geometry_config go together; acts_material_map and "
                             "acts_energy_loss need both")
        if Bz == 0 or acts_kalman_fallback_covariance_scale <= 0 or field_map_kind not in ("xyz", "rz"):
            raise ValueError("Bz must not be 0, acts_kalman_fallback_covariance_scale must be > 0, field_map_kind xyz or rz")
        self.learn_weights = learn_weights
        self.Bz = Bz
        self.noise_pt_threshold = noise_pt_threshold
        self.valid_radius = valid_radius
        self.min_hits = min_hits
        self.beam_position = tuple(beam_position)
        self.normalization = dict(TPC_NORMALIZATION if normalization is None else normalization)
        self.use_acts = use_acts
        self.acts_image = acts_image
        self.acts_command = acts_command
        self.field_map = field_map
        self.field_map_kind = field_map_kind
        self.acts_max_steps = acts_max_steps
        self.acts_threads = acts_threads
        self.acts_geometry = acts_geometry
        self.acts_geometry_config = acts_geometry_config
        self.acts_material_map = acts_material_map
        self.acts_energy_loss = acts_energy_loss
        self.acts_particle_pdg = acts_particle_pdg
        self.acts_particle_mass = acts_particle_mass
        self.acts_kalman_refit = acts_kalman_refit
        self.acts_kalman_start_covariance_scale = acts_kalman_start_covariance_scale
        self.acts_kalman_fallback_covariance_scale = acts_kalman_fallback_covariance_scale
        self.hit_sigma_rphi = hit_sigma_rphi
        self.hit_sigma_z = hit_sigma_z
        self.track_error_scale = track_error_scale
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
            self.log_kalman_fallback_scale = nn.Parameter(torch.tensor(math.log(acts_kalman_fallback_covariance_scale)))

    def start_acts(self):
        """Starts the ACTS worker in the background (with use_acts), so it is ready by the first forward pass."""
        if self.use_acts:
            _acts_worker(self.acts_worker_command(), self.acts_worker_config())

    def acts_worker_command(self):
        """
        The command that runs the ACTS worker: acts_command if set, else singularity exec acts_image with
        scripts/acts and the ACTS input directories bound.
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
        return {"reference": [self.beam_position[0] * MM_PER_CM, self.beam_position[1] * MM_PER_CM, 0.0],
                "bz_tesla": self.Bz, "field_map": self.field_map, "field_map_kind": self.field_map_kind,
                "max_steps": self.acts_max_steps, "threads": self.acts_threads, "geometry": self.acts_geometry,
                "geometry_config": self.acts_geometry_config, "material_map": self.acts_material_map,
                "energy_loss": self.acts_energy_loss, "particle_pdg": self.acts_particle_pdg,
                "particle_mass": self.acts_particle_mass, "kalman_refit": self.acts_kalman_refit,
                "kalman_start_covariance_scale": self.acts_kalman_start_covariance_scale, "hit_sigma": [self.hit_sigma_rphi * MM_PER_CM, self.hit_sigma_z * MM_PER_CM]}

    def _usable(self, pT, beam_dca):
        """The track usability cuts: pT >= noise_pt_threshold and beam_dca < valid_radius (False where NaN)."""
        return (pT >= self.noise_pt_threshold) & (beam_dca < self.valid_radius)

    def track_fit_config(self):
        """The track-model settings."""
        return {k: getattr(self, k) for k in ("Bz", "field_map", "noise_pt_threshold", "valid_radius", "min_hits",
                                              "beam_position", "normalization", "use_acts", "acts_kalman_refit")}

    def forward(self, class_probs=None, mask_probs=None, points=None, padding_mask=None, track_parameters=None,
                track_covariance=None):
        """
        Args:
            class_probs: (B, T, 2) P(empty), P(real); always required (P(real) is the base track weight).
            mask_probs: (B, N, T); points: (B, N, 4) normalized hits; padding_mask: (B, N), True = real: the hits
                (unused with track_parameters).
            track_parameters, track_covariance: (B, T, 6), (B, T, 6, 6) given perigee tracks instead of hits
                (ACTS units: mm).

        Returns:
            dict: track_parameters (B, T, 6) cm, track_position, track_direction (B, T, 3), track_weight,
            track_quality_mask, track_sigma (B, T, 2); acts_failed, acts_refit_failed, track_covariance as
            VertexHead.forward; for VertexFinder: _helix (circle centre, R_s, z slope, z0), _pT, _beam_dca, _bad
            (B, T) tracks without a fit.
        """
        acts_failed = refit_failed = track_sigma = None
        base_weight = class_probs[..., 1]
        if track_parameters is None:
            # track model: hits in cm (padding zeroed), hard assignment, each track's circle and z line
            hits, hit_r = _unnormalize_hits(points.double().masked_fill(~padding_mask.unsqueeze(-1), 0.0), self.normalization)
            # each real hit goes to its most probable track slot (none if all probabilities are 0)
            n_tracks = mask_probs.shape[-1]
            assigned = padding_mask & (mask_probs.max(dim=-1).values > 0)
            track = mask_probs.argmax(dim=-1).masked_fill(~assigned, 0)
            # with a field map: each track's Bz, the mean over its hits (signed; tracks without hits: Bz)
            track_bz = self.Bz
            if self.field_map is not None:
                hit_bz = _field_map(self.field_map, self.field_map_kind).bz(
                    hits[..., 1], hits[..., 2], hits[..., 3], MM_PER_CM)
                n_hits = _per_track_sum(assigned.to(hit_bz.dtype), track, n_tracks)
                track_bz = torch.where(n_hits > 0, _per_track_sum(hit_bz * assigned, track, n_tracks) / n_hits.clamp(min=1),
                                       float(self.Bz))
            helix = _helix_from_hits(hits, hit_r, track, assigned, n_tracks, Bz=track_bz, min_hits=self.min_hits,
                                     beam_position=self.beam_position)
            centre, R_s, zslope, theta, qop = (helix[k] for k in ("circle_centre", "R_s", "zslope", "theta", "qop"))
            # the hit fit's covariance at a point s_reference along each helix (cm, from the hit errors)
            hit_covariance = lambda s_reference, curvilinear: _track_covariance(  # noqa: E731
                helix["hit_s"], track, assigned, n_tracks, theta, qop, s_reference, self.hit_sigma_rphi, self.hit_sigma_z,
                track_bz, curvilinear, self.min_hits)
            if self.use_acts:
                # ACTS propagates the helix state at the innermost hit (mm) back to the beam spot; with the Kalman
                # refit, ACTS first fits the hits from it
                start_xy, t_xy = _circle_point(centre, R_s, hits[..., 1:3].gather(
                    1, helix["innermost"].unsqueeze(-1).expand(-1, -1, 2)))
                start_s = _arc_length(start_xy[..., 0], start_xy[..., 1], centre[..., 0], centre[..., 1], R_s,
                                      self.beam_position)
                start_z = start_s * zslope + helix["z0"]
                direction = torch.cat([torch.sin(theta).unsqueeze(-1) * t_xy, torch.cos(theta).unsqueeze(-1)], dim=-1)
                # start covariance (curvilinear, mm; t uncorrelated) for ACTS to transport
                to_mm = qop.new_tensor([MM_PER_CM, MM_PER_CM, 1.0, 1.0, 1.0])
                start_covariance = qop.new_zeros(*qop.shape, 6, 6)
                start_covariance[..., 5, 5] = 1.0
                start_covariance[..., :5, :5] = hit_covariance(start_s, True) * to_mm[:, None] * to_mm[None, :]
                start_covariance = start_covariance.reshape(-1, 6, 6).cpu().numpy()
                # the refit's hits: position (mm) and the helix direction there, each with its state's index
                # (event x T + slot), in order along the track
                hit_track, hit_xyz = torch.empty(0), torch.empty(0, 6)
                if self.acts_kalman_refit:
                    hit_track = (torch.arange(track.shape[0], device=track.device).unsqueeze(1) * n_tracks + track)[assigned]
                    order = torch.sort(helix["hit_s"][assigned], stable=True).indices
                    order = order[torch.sort(hit_track[order], stable=True).indices]
                    _, hit_t = _circle_point(centre.gather(1, track.unsqueeze(-1).expand(-1, -1, 2)), R_s.gather(1, track),
                                             hits[..., 1:3])
                    hit_theta = theta.gather(1, track).unsqueeze(-1)
                    hit_xyz = torch.cat([hits[..., 1:4] * MM_PER_CM, torch.sin(hit_theta) * hit_t,
                                         torch.cos(hit_theta)], dim=-1)[assigned][order]
                    hit_track = hit_track[order]
                acts_out = _acts_worker(self.acts_worker_command(), self.acts_worker_config()).propagate(
                    (torch.cat([start_xy, start_z.unsqueeze(-1)], dim=-1) * MM_PER_CM
                     ).reshape(-1, 3).cpu().numpy(), direction.reshape(-1, 3).cpu().numpy(),
                    qop.reshape(-1).cpu().numpy(), start_covariance, hit_track.cpu().numpy(), hit_xyz.cpu().numpy())
                track_parameters = torch.from_numpy(acts_out[0]).reshape(*qop.shape, 6).to(hits.device)
                acts_failed = self._usable(helix["pT"], helix["beam_dca"]) & ~torch.from_numpy(acts_out[1]).reshape(
                    qop.shape).to(hits.device)
                track_covariance = torch.from_numpy(acts_out[2]).reshape(*qop.shape, 6, 6).to(hits.device)
                if self.acts_kalman_refit:
                    # tracks the refit failed on (hit fit used) count less: their covariance inflated
                    refit_failed = acts_out[1] & ~acts_out[3]
                    refit_failed = torch.from_numpy(refit_failed).reshape(qop.shape).to(hits.device)
                    scale = (self.log_kalman_fallback_scale.exp().double() if self.learn_weights
                             else self.acts_kalman_fallback_covariance_scale)
                    track_covariance = track_covariance * torch.where(refit_failed, scale, 1.0)[..., None, None]
            else:
                # the hit fit as perigee tracks (cm); its circle comes back with the field its pT and charge came from
                perigee, perigee_bz = _perigee_from_circle(centre, R_s, theta, helix["z0"], qop, self.beam_position), track_bz
                covariance = hit_covariance(torch.zeros_like(theta), False)
                track_sigma = torch.stack([covariance[..., 0, 0], covariance[..., 1, 1]], dim=-1).sqrt()
        if track_parameters is not None:
            perigee = track_parameters.double() / track_parameters.new_tensor(        # d0, z0, t: mm -> cm
                [MM_PER_CM, MM_PER_CM, 1.0, 1.0, 1.0, MM_PER_CM], dtype=torch.float64)
            # with a field map: Bz at each track's perigee (beam line, z0)
            perigee_bz = self.Bz if self.field_map is None else _field_map(self.field_map, self.field_map_kind).bz(
                torch.full_like(perigee[..., 1], self.beam_position[0]), torch.full_like(perigee[..., 1], self.beam_position[1]),
                perigee[..., 1], MM_PER_CM)
            if track_covariance is None:
                raise ValueError("track_parameters need their track_covariance")
            track_sigma = torch.stack([track_covariance[..., 0, 0], track_covariance[..., 1, 1]], dim=-1
                                      ).double().sqrt() / MM_PER_CM
        # every route: perigee tracks (cm) -> circle, z line, pT and usability; the vertex fit in float64
        dtype = torch.float64
        base_weight = base_weight.to(dtype)
        bad = ~torch.isfinite(perigee[..., :5]).all(dim=-1)          # tracks without a fit
        circle_centre, R_s, zslope, z0, pT, beam_dca = _circle_from_perigee(
            perigee, perigee_bz, self.beam_position)
        # tracks without a usable covariance get no weight (and a finite stand-in error) and are not usable
        has_sigma = torch.isfinite(track_sigma).all(dim=-1) & (track_sigma > 0).all(dim=-1)
        # the helix for the fit, with finite stand-ins for tracks without one (weight 0; their outputs are masked below)
        params = {"pT": pT, "beam_dca": beam_dca, "usable": ~bad & has_sigma & self._usable(pT, beam_dca),
                  "helix": (circle_centre.to(dtype).masked_fill(bad.unsqueeze(-1), 0.0), R_s.to(dtype).masked_fill(bad, 1.0),
                            zslope.to(dtype).masked_fill(bad, 0.0), z0.to(dtype).masked_fill(bad, 0.0))}
        base_weight = base_weight * has_sigma.to(dtype)
        params["track_sigma"] = torch.where(has_sigma.unsqueeze(-1), track_sigma * self.track_error_scale, 1.0).to(dtype)
        perigee = perigee.masked_fill(bad.unsqueeze(-1), float("nan"))

        # straight line of each track at its perigee; tracks without a fit get a finite stand-in (weight 0)
        d0, z0, phi, theta = perigee[..., 0], perigee[..., 1], perigee[..., 2], perigee[..., 3]
        track_position = torch.stack([self.beam_position[0] - d0 * torch.sin(phi),
                                      self.beam_position[1] + d0 * torch.cos(phi), z0], dim=-1)
        track_direction = torch.stack([torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi),
                                       torch.cos(theta)], dim=-1)
        track_position = torch.where(bad.unsqueeze(-1), 0.0, track_position).to(dtype)
        track_direction = torch.where(bad.unsqueeze(-1), track_direction.new_tensor([1.0, 0.0, 0.0]), track_direction).to(dtype)

        track_weight, usable = base_weight, params["usable"].to(dtype)
        if self.learn_weights:
            # features: class_probs (2) + (q/(pT+1), theta, sin phi, cos phi) at the perigee; filler without a fit
            pT = params["pT"].nan_to_num(0.0)
            q = torch.sign(perigee[..., 4]).nan_to_num(1.0)
            track_features = torch.stack([q / (pT + 1.0), theta.nan_to_num(torch.pi / 2), torch.sin(phi).nan_to_num(0.0),
                                          torch.cos(phi).nan_to_num(1.0)], dim=-1)
            network_dtype = next(self.track_weight_network.parameters()).dtype
            learned_weight_scale = self.track_weight_network(torch.cat([class_probs.to(network_dtype),
                                                                        track_features.to(network_dtype)], dim=-1)).squeeze(-1).to(dtype)
            track_weight = base_weight * learned_weight_scale
            # straight-through gates: hard cuts forward, sigmoid gradients backward
            pT, beam_dca = params["pT"].nan_to_num(0.0), params["beam_dca"].nan_to_num(1e6)
            soft = (torch.sigmoid((pT.to(track_weight.dtype) - self.pt_threshold) / self.log_gate_width_pt.exp())
                    * torch.sigmoid((self.beam_radius - beam_dca.to(track_weight.dtype)) / self.log_gate_width_r.exp()))
            params["usable"] = ~bad & has_sigma & (pT >= self.pt_threshold.detach()) & (beam_dca < self.beam_radius.detach())  # SVs too
            usable = params["usable"].to(track_weight.dtype) + soft - soft.detach()
        track_weight = track_weight * usable
        outputs = {
            "track_parameters": perigee.to(dtype),
            "track_position": track_position,
            "track_direction": track_direction,
            "track_weight": track_weight,
            "track_quality_mask": params["usable"],
            "track_sigma": params["track_sigma"],
        }
        if acts_failed is not None:
            outputs["acts_failed"] = acts_failed
        if refit_failed is not None:
            outputs["acts_refit_failed"] = refit_failed
        if track_covariance is not None:
            outputs["track_covariance"] = track_covariance
        return {**outputs, "_helix": params["helix"], "_pT": params["pT"], "_beam_dca": params["beam_dca"], "_bad": bad}
