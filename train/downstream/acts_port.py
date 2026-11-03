"""
acts_port.py: batched torch port of the ACTS pieces the downstream tasks need, so they run on the GPU,
keep gradients and need no ACTS install. Mirrors ACTS's conventions and names (Acts/Definitions/Units.hpp,
Acts::BoundIndices, LineSurface / PerigeeSurface, ConstantBField); results should match ACTS's.

Units: ACTS native throughout. Length mm, momentum GeV, charge e, angle rad, field GeV/(e mm), time as
c t in mm. Convert at the boundary (e.g. r_cm * UnitConstants.cm; VertexHead works in cm).

Ported so far: bound perigee parameters <-> free state, and exact propagation of a free state to a
perigee surface in a uniform solenoid field.
Not yet: non-uniform fields. Any field object with get_field(position) -> (..., 3) native fits the
interface; propagate_to_perigee raises for one without uniform_bz until a stepper (Runge-Kutta, like
ACTS's EigenStepper) is ported, once a field map exists.

Track model (|q| = 1, uniform Bz): R = pT / |Bz|; h = sign(q Bz), h = +1 turns clockwise seen from +z;
centre = position + h R (sin phi, -cos phi); d(phi)/ds_T = -h / R and dz/ds_T = cot(theta), s_T the
transverse arc length.
"""
import math

import torch


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
        raise NotImplementedError("non-uniform field: the Runge-Kutta stepper is not ported yet")
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
