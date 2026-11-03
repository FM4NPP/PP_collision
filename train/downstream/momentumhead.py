"""
momentumhead.py: track parameters (momentum, charge, direction) fitted from track-finder output.

EXPERIMENTAL (2026-11-03): ACTS computes momentum and track parameters (its Python bindings, run in
the container in scripts/acts). This module is the package's own pure-torch alternative, kept so that
avenue stays open: MomentumHead's hit fit and, at the end, the perigee maths. Not maintained like the
main tasks; neither are momentum_reconstruction_trainer.py, eval_momentum_reconstruction.py and
train_momentum_reconstruction.py.

The track model (circle and z-line fits) lives in vertexhead.py, which does not import this file;
MomentumHead imports it and adds the momentum-only outputs (track_reg_result, radius).

Each track's hits (argmax of mask_probs) get a Taubin circle fit in x-y (pT, charge, circle) and a
line fit of z against signed arc length s along that circle (against x for cosmics) for theta and z
at the beam line; no truth is used. VertexHead's helix fit steps along the circle and z line.

Inputs: mask_probs (B, N, T), normalized points (B, N, 4), padding_mask (B, N). B events, N hits,
T track slots. Positions in cm.

Detector- and dataset-dependent settings are MomentumHead arguments, never inline constants: the field
Bz, the beam position, the hit normalization (TPC_NORMALIZATION by default), the beam-line cut
valid_radius, the noise pT cut and the minimum hits per fit.
"""
import math

import torch
import torch.nn as nn

# the track model lives in vertexhead.py (moved there 2026-11-02 so the vertex fit doesn't depend on this
# experimental file); imported back, including the helpers, so existing imports from here keep working
from vertexhead import (  # noqa: F401
    TPC_NORMALIZATION, _per_track_sum, _line_fit, _circle_fit, _extreme_hit, _arc_length,
    _unnormalize_hits, _track_position_from_hits, _helix_from_hits,
)

def _track_params_from_hits(points, hit_r, is_assigned, noise_pt_threshold, Bz=1.4,
                           scaling_factor=100.0, valid_radius=1.0, min_hits=3, is_cosmics=False,
                           beam_position=(0.0, 0.0)):
    """
    Track parameters and quality labels from each track's assigned hits: vertexhead's _helix_from_hits
    plus the momentum outputs at the innermost hit (centre = innermost hit + R_s (sin phi, -cos phi)).

    Args:
        as _helix_from_hits.

    Returns:
        dict, NaN (or filler / False) where fit_ok is False:
            track_reg_result (B, T, 4): (q/(pT+1), theta, sin phi, cos phi).
            usable (B, T): pT >= noise_pt_threshold and beam_dca < valid_radius.
            pT, radius, beam_dca, fit_ok (B, T).
            circle_centre (B, T, 2), R_s, zslope, z0 (B, T): the circle and z line z = zslope * s + z0
                (s from _arc_length), for the helix fit (dtype of points).
    """
    helix = _helix_from_hits(points, hit_r, is_assigned, noise_pt_threshold, Bz=Bz, scaling_factor=scaling_factor,
                             valid_radius=valid_radius, min_hits=min_hits, is_cosmics=is_cosmics,
                             beam_position=beam_position)
    # phi at each track's innermost assigned hit, from the circle
    assigned = is_assigned.any(dim=-1)
    track = is_assigned.to(torch.uint8).argmax(dim=-1)
    i_in = _extreme_hit(hit_r.double(), track, assigned, is_assigned.shape[-1])
    sign_Rs, radius = torch.sign(helix["R_s"]), helix["R_s"].abs()
    sin_phi = sign_Rs * (helix["circle_centre"][..., 0] - points[..., 1].double().gather(1, i_in)) / radius
    cos_phi = -sign_Rs * (helix["circle_centre"][..., 1] - points[..., 2].double().gather(1, i_in)) / radius
    q = sign_Rs * (1.0 if Bz >= 0 else -1.0)
    params = torch.stack([q / (helix["pT"] + 1.0), helix["theta"], sin_phi, cos_phi], dim=-1)
    params = torch.where(helix["fit_ok"].unsqueeze(-1), params, params.new_tensor([0.5, torch.pi / 2, 0.0, 1.0]))
    return {
        "track_reg_result": params.to(points.dtype),
        "usable": helix["usable"], "pT": helix["pT"], "radius": radius, "beam_dca": helix["beam_dca"],
        "fit_ok": helix["fit_ok"], **{k: helix[k].to(points.dtype) for k in ("circle_centre", "R_s", "zslope", "z0")},
    }

class MomentumHead(nn.Module):

    """
    Per-track momentum fit from track-finder output. No learnable parameters yet. Every detector- or
    dataset-dependent value is an argument; the defaults are sPHENIX's (TPC, 1.4 T solenoid).

    Args:
        Bz: solenoid field along z, Tesla (uniform).
        is_cosmics: fit z against x instead of arc length.
        scaling_factor: converts the radius pT / (0.2998 Bz), in metres, to the units of points (100: cm).
        noise_pt_threshold: GeV; tracks with lower fitted pT are labelled noise (not usable).
        valid_radius: cm; a track is usable only if its circle passes within this of the beam line.
        min_hits: assigned hits needed for the circle and z-line fits.
        beam_position: (x, y) of the beam line, cm.
        normalization: the dataset's hit normalization (see _unnormalize_hits); TPC_NORMALIZATION if None.
    """
    def __init__(
        self,
        Bz: float = 1.4,
        is_cosmics: bool = False,
        scaling_factor: float = 100.0,
        noise_pt_threshold: float = 0.25,
        valid_radius: float = 1.0,
        min_hits: int = 3,
        beam_position: tuple = (0.0, 0.0),
        normalization: dict = None,
    ):
        super().__init__()
        self.Bz = Bz
        self.is_cosmics = is_cosmics
        self.scaling_factor = scaling_factor
        self.noise_pt_threshold = noise_pt_threshold
        self.valid_radius = valid_radius
        self.min_hits = min_hits
        self.beam_position = tuple(beam_position)
        self.normalization = dict(TPC_NORMALIZATION if normalization is None else normalization)

    def config(self):
        """Constructor arguments, as stored in a momentum checkpoint."""
        return {"Bz": self.Bz, "is_cosmics": self.is_cosmics, "scaling_factor": self.scaling_factor,
                "noise_pt_threshold": self.noise_pt_threshold, "valid_radius": self.valid_radius,
                "min_hits": self.min_hits, "beam_position": self.beam_position,
                "normalization": self.normalization}

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
        points, hit_r = _unnormalize_hits(points.masked_fill(~padding_mask.unsqueeze(-1), 0.0), self.normalization)
        track_position, total_hit_weight, is_assigned = _track_position_from_hits(
            points, hit_r, mask_probs, padding_mask)
        params = _track_params_from_hits(points, hit_r, is_assigned, self.noise_pt_threshold,
                                         Bz=self.Bz, scaling_factor=self.scaling_factor,
                                         valid_radius=self.valid_radius, min_hits=self.min_hits,
                                         is_cosmics=self.is_cosmics, beam_position=self.beam_position)
        return {"track_position": track_position, "total_hit_weight": total_hit_weight,
                "is_assigned": is_assigned, **params}


# ------------------------------------------------------------------------------------------------
# Perigee maths (merged from acts_port.py, 2026-11-03): bound perigee parameters <-> free state, and exact
# propagation of a free state to a perigee surface in a uniform solenoid field; batched torch, so it runs
# on the GPU and keeps gradients. Written from the helix equations, independently of ACTS's source; it
# follows ACTS's documented conventions (units, bound-parameter order, perigee d0 sign, time as c t) so
# results compare directly. Checked against ACTS 48.2 (scripts/acts/compare_acts.py, 50k tracks): agrees
# to ACTS's integration accuracy (<= 1e-4 mm), ~100x faster than ACTS on CPU, ~1000x on GPU.
#
# Units: ACTS native here (mm, GeV, e, rad, field GeV/(e mm), time c t in mm), unlike MomentumHead (cm);
# convert at the boundary (r_cm * UnitConstants.cm).
# Not yet: non-uniform fields. Any field object with get_field(position) -> (..., 3) native fits the
# interface; propagate_to_perigee raises for one without uniform_bz until a stepper (Runge-Kutta) is
# written, once a field map exists.
# Track model (|q| = 1, uniform Bz): R = pT / |Bz|; h = sign(q Bz), h = +1 turns clockwise seen from +z;
# centre = position + h R (sin phi, -cos phi); d(phi)/ds_T = -h / R and dz/ds_T = cot(theta), s_T the
# transverse arc length.
# ------------------------------------------------------------------------------------------------

class UnitConstants:
    """ACTS's acts.UnitConstants (native unit = 1)."""
    mm, um, cm, m = 1.0, 1e-3, 10.0, 1000.0
    GeV, MeV, keV = 1.0, 1e-3, 1e-6
    e = 1.0
    rad, mrad, degree = 1.0, 1e-3, math.pi / 180.0
    s = 299792458000.0                  # c t in mm
    ns = 1e-9 * s
    T = 0.000299792458                  # GeV / (e mm)
    Gauss = 1e-4 * T
    kGauss = 1e-1 * T


class BoundIndices:
    """Acts::BoundIndices: the order of bound (perigee) parameters."""
    eBoundLoc0, eBoundLoc1, eBoundPhi, eBoundTheta, eBoundQOverP, eBoundTime = range(6)
    eBoundSize = 6


PION_MASS = 0.13957039 * UnitConstants.GeV     # ACTS's default ParticleHypothesis.pion


class ConstantBField:
    """
    ACTS's ConstantBField: the same field everywhere.

    Args:
        field: (3,) field vector, native units (e.g. (0, 0, 1.4 * UnitConstants.T)).
    """
    def __init__(self, field):
        self.field = torch.as_tensor(field, dtype=torch.float64)

    @property
    def uniform_bz(self):
        """Bz for a solenoid field along z (what the exact helix needs); None if bx or by is non-zero."""
        return None if bool(self.field[:2].abs().max() > 0) else float(self.field[2])

    def get_field(self, position):
        """(..., 3) field at each position (..., 3)."""
        return self.field.to(position).expand_as(position)


def perigee_to_free(params, reference=(0.0, 0.0, 0.0)):
    """
    Free state at the perigee of bound perigee parameters.

    Args:
        params: (..., 6) (d0, z0, phi, theta, q/p, t), BoundIndices order.
        reference: (3,) or (..., 3) the perigee surface's centre (beam spot); its line is parallel to z.

    Returns:
        position (..., 3), direction (..., 3) unit, qop (...), time (...).
    """
    reference = torch.as_tensor(reference).to(params)
    d0, z0, phi, theta, qop, time = params.unbind(-1)
    position = reference + torch.stack([-d0 * torch.sin(phi), d0 * torch.cos(phi), z0], dim=-1)
    direction = torch.stack([torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi),
                             torch.cos(theta)], dim=-1)
    return position, direction, qop, time


def propagate_to_perigee(position, direction, qop, time=None, reference=(0.0, 0.0, 0.0), field=None,
                         navigation="closest", mass=PION_MASS, straight_radius=1e9, surface_tolerance=1e-4):
    """
    Bound perigee parameters of free states, by exact helix propagation to the perigee surface (a line
    through reference parallel to z): the point of closest approach in x-y.

    Args:
        position (..., 3), direction (..., 3) unit, qop (...) e/GeV, time (...) c t (default 0): free states.
        reference: (3,) or (..., 3) perigee surface centre (beam spot).
        field: ConstantBField-like with uniform_bz; None = no field (straight lines).
        navigation: which turn of a helix that loops past the perigee more than once:
            "forward" the first ahead (ACTS's forward propagation), "backward" the first behind (ACTS
            propagating backward, e.g. from the innermost hit), "closest" the nearest along the arc.
        mass: GeV, particle hypothesis for the time (ACTS default: pion).
        straight_radius: mm; tracks with a larger transverse radius (or q/p = 0) are straight lines.
        surface_tolerance: mm of transverse arc; a state this close to its perigee counts as on it (as
            ACTS's s_onSurfaceTolerance), so forward / backward don't go a full turn round.

    Returns:
        (..., 6) (d0, z0, phi, theta, q/p, t) in BoundIndices order; and the signed 3D path length (...)
        from each state to its perigee (negative = backward). Straight lines ignore navigation.
    """
    bz = 0.0 if field is None else field.uniform_bz
    if bz is None:
        raise NotImplementedError("non-uniform field: no Runge-Kutta stepper yet (use ACTS, scripts/acts)")
    if navigation not in ("forward", "backward", "closest"):
        raise ValueError(f"navigation must be forward, backward or closest, not {navigation!r}")
    reference = torch.as_tensor(reference).to(position).expand_as(position)
    time = torch.zeros_like(qop) if time is None else time

    phi = torch.atan2(direction[..., 1], direction[..., 0])
    theta = torch.atan2(torch.hypot(direction[..., 0], direction[..., 1]), direction[..., 2])
    radius = torch.sin(theta) / (qop.abs() * abs(bz)) if bz != 0.0 else torch.full_like(qop, math.inf)
    straight = ~(radius < straight_radius)
    radius = torch.where(straight, torch.ones_like(radius), radius)          # finite stand-in, unused
    h = torch.sign(qop) * (1.0 if bz >= 0 else -1.0)
    h = torch.where(h == 0, torch.ones_like(h), h)

    # circle: closest point to the reference line, the heading there, and the transverse arc to it
    centre = position[..., :2] + (h * radius).unsqueeze(-1) * torch.stack([torch.sin(phi), -torch.cos(phi)], dim=-1)
    u = reference[..., :2] - centre
    u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    phi_circle = torch.atan2(-h * u[..., 0], h * u[..., 1])
    turn = 2 * math.pi * radius
    arc = -h * radius * (phi_circle - phi)                                    # one of the s_T, mod one turn
    # a state on the surface (within surface_tolerance, either side) stays there, not a full turn away
    ahead = torch.remainder(arc, turn)
    arc = {"forward": torch.where(ahead > turn - surface_tolerance, ahead - turn, ahead),
           "backward": torch.where(ahead < surface_tolerance, ahead, ahead - turn),
           "closest": torch.remainder(arc + turn / 2, turn) - turn / 2}[navigation]
    pca_circle = centre + radius.unsqueeze(-1) * u

    # straight line: projection of the reference onto the track in x-y
    arc_line = ((reference[..., :2] - position[..., :2]) * torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)).sum(-1)
    pca_line = position[..., :2] + arc_line.unsqueeze(-1) * torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)

    arc = torch.where(straight, arc_line, arc)
    pca = torch.where(straight.unsqueeze(-1), pca_line, pca_circle) - reference[..., :2]
    phi_pca = torch.where(straight, phi, phi_circle)
    path = arc / torch.sin(theta)
    d0 = -pca[..., 0] * torch.sin(phi_pca) + pca[..., 1] * torch.cos(phi_pca)
    z0 = position[..., 2] + arc * torch.cos(theta) / torch.sin(theta) - reference[..., 2]
    # c dt = ds / beta, beta = p / E with |q| = 1
    t = time + path * torch.sqrt(1.0 + (mass * qop) ** 2)
    return torch.stack([d0, z0, phi_pca, theta, qop, t], dim=-1), path
