"""
momentumhead.py: track parameters (momentum, charge, direction) fitted from track-finder output.

Each track's hits (argmax of mask_probs) get a Taubin circle fit in x-y (pT, charge, circle) and a
line fit of z against signed arc length s along that circle (against x for cosmics) for theta and z
at the beam line; no truth is used. VertexHead's helix fit steps along the circle and z line.

Inputs: mask_probs (B, N, T), normalized points (B, N, 4), padding_mask (B, N). B events, N hits,
T track slots. Positions in cm.
"""
import torch
import torch.nn as nn

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

def _arc_length(x, y, xc, yc, R_s):
    """
    Signed arc length along a circle (centre xc, yc; signed radius R_s, sign(R_s) = sign(q Bz)) from its
    closest approach to the beam line to the circle point at the angle of (x, y); positive in the
    direction of travel (anticlockwise <=> R_s < 0). All inputs broadcast.
    """
    # in place, so a long per-hit array costs one allocation
    s = torch.atan2(y - yc, x - xc)
    return s.sub_(torch.atan2(-yc, -xc)).add_(torch.pi).remainder_(2 * torch.pi).sub_(torch.pi).mul_(-R_s)

def _unnormalize_hits(points):
    """
    Invert TPCBatchDataset's normalization (z-scored E; eta, phi, r min-max scaled to [0, 1]).

    Returns:
        points (..., 4) as (E, x, y, z) in cm, and the transverse radius r (...).
    """
    # TPCBatchDataset's normalization (fm4npp/datasets/dataset.py, TPCBatchDataset.__init__);
    # keep these in line with it
    eta_lim = (-2.0, 2.0)
    phi_lim = (-torch.pi, torch.pi)
    r_lim = (31.371997833251953, 75.38493347167969)    # transverse radius, cm
    E_mean, E_std = 253.0982, 268.7093

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

def _track_params_from_hits(points, hit_r, is_assigned, noise_pt_threshold, Bz=1.4,
                           scaling_factor=100.0, valid_radius=1.0, min_hits=3, is_cosmics=False):
    """
    Track parameters and quality labels from each track's assigned hits.

    Travel from the innermost to the outermost hit gives the charge and phi at the innermost hit:
    centre = innermost hit + R_s (sin phi, -cos phi), sign(R_s) = sign(q Bz).

    Args:
        points, hit_r, is_assigned: as from _track_position_from_hits.
        noise_pt_threshold: GeV; lower fitted pT -> not usable.
        Bz: Tesla; scaling_factor: converts metres to the units of points.
        valid_radius: the circle must pass within this of the beam line to be usable.
        min_hits: fewer assigned hits -> not fitted; is_cosmics: fit z against x instead of s.

    Returns:
        dict, NaN (or filler / False) where fit_ok is False:
            track_reg_result (B, T, 4): (q/(pT+1), theta, sin phi, cos phi).
            usable (B, T): pT >= noise_pt_threshold and beam_dca < valid_radius.
            pT, radius, beam_dca, fit_ok (B, T).
            circle_centre (B, T, 2), R_s, zslope, z0 (B, T): the circle and z line z = zslope * s + z0
                (s from _arc_length), for the helix fit (dtype of points).
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
                                         (sign_Rs * radius).gather(1, track))
    zslope, z0 = _line_fit(s, z, track, assigned, n_tracks, min_hits=min_hits)
    ux, uy = (xc - xin) / radius, (yc - yin) / radius
    sin_phi, cos_phi = sign_Rs * ux, -sign_Rs * uy
    q = sign_Rs * (1.0 if Bz >= 0 else -1.0)
    pT = 0.299792458 * abs(Bz) * radius / scaling_factor     # pT [GeV] = c [1e9 m/s] * B [T] * R [m]
    # cot(theta) = dz/ds: the arc-length slope, or dz/dx * dx/ds = zslope cos(phi) for cosmics
    cot_theta = zslope * cos_phi if is_cosmics else zslope
    theta = torch.atan2(torch.ones_like(cot_theta), cot_theta)
    beam_dca = (torch.sqrt(xc * xc + yc * yc) - radius).abs()

    params = torch.stack([q / (pT + 1.0), theta, sin_phi, cos_phi], dim=-1)
    fit_ok = torch.isfinite(params).all(dim=-1)
    filler = params.new_tensor([0.5, torch.pi / 2, 0.0, 1.0])
    params = torch.where(fit_ok.unsqueeze(-1), params, filler)
    nan = torch.full_like(pT, float("nan"))
    pT, radius, beam_dca = (torch.where(fit_ok, v, nan) for v in (pT, radius, beam_dca))

    circle_centre = torch.where(fit_ok.unsqueeze(-1), torch.stack([xc, yc], dim=-1), nan.unsqueeze(-1))
    R_s, zslope, z0 = (torch.where(fit_ok, v, nan) for v in (sign_Rs * radius, zslope, z0))

    return {
        "track_reg_result": params.to(points.dtype),
        "usable": (pT >= noise_pt_threshold) & (beam_dca < valid_radius),    # NaN where fit_ok is False -> False
        "pT": pT, "radius": radius, "beam_dca": beam_dca, "fit_ok": fit_ok,
        "circle_centre": circle_centre.to(points.dtype), "R_s": R_s.to(points.dtype),
        "zslope": zslope.to(points.dtype), "z0": z0.to(points.dtype),
    }

class MomentumHead(nn.Module):

    """
    Per-track momentum fit from track-finder output. No learnable parameters yet.

    Args:
        Bz: solenoid field along z, Tesla.
        is_cosmics: fit z against x instead of arc length.
        scaling_factor: converts the radius pT / (0.2998 Bz), in metres, to the units of points.
        noise_pt_threshold: GeV; tracks with lower fitted pT are labelled noise (not usable).
    """
    def __init__(
        self,
        Bz: float = 1.4,
        is_cosmics: bool = False,
        scaling_factor: float = 100.0,
        noise_pt_threshold: float = 0.25,
    ):
        super().__init__()
        self.Bz = Bz
        self.is_cosmics = is_cosmics
        self.scaling_factor = scaling_factor
        self.noise_pt_threshold = noise_pt_threshold

    def config(self):
        """Constructor arguments, as stored in a momentum checkpoint."""
        return {"Bz": self.Bz, "is_cosmics": self.is_cosmics, "scaling_factor": self.scaling_factor,
                "noise_pt_threshold": self.noise_pt_threshold}

    def save_checkpoint(self, path, **extra):
        """Write config() and the weights to path; extra keys are stored alongside."""
        torch.save({"momentum_config": self.config(), "model_state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        """Rebuild a MomentumHead from save_checkpoint's file and load its weights strictly."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        head = cls(**checkpoint["momentum_config"])
        head.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()})
        return head

    def forward(self, mask_probs, points, padding_mask):
        """
        Args:
            mask_probs: (B, N, T); points: (B, N, 4) normalized hits; padding_mask: (B, N).

        Returns:
            dict: _track_position_from_hits' outputs and _track_params_from_hits'.
        """
        # padded hits are -100 in every column, which overflows sinh(eta); any finite value will do
        points, hit_r = _unnormalize_hits(points.masked_fill(~padding_mask.unsqueeze(-1), 0.0))
        track_position, total_hit_weight, is_assigned = _track_position_from_hits(
            points, hit_r, mask_probs, padding_mask)
        params = _track_params_from_hits(points, hit_r, is_assigned, self.noise_pt_threshold,
                                         Bz=self.Bz, scaling_factor=self.scaling_factor,
                                         is_cosmics=self.is_cosmics)
        return {"track_position": track_position, "total_hit_weight": total_hit_weight,
                "is_assigned": is_assigned, **params}
