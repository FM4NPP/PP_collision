"""
vertices.py: VertexFinder, primary and secondary vertices from TrackBuilder's tracks, as ACTS finds them.

Primary vertices: ACTS's adaptive multi-vertex finder (AMVF) with the settings of its shipped example
(AdaptiveMultiVertexFinderAlgorithm, Gaussian track-density seeder): seed, preselect tracks, fit all connected
vertices together (adaptive multi-vertex fit: Kalman vertex updates on linearised helices, annealed weights shared
between vertices), keep or delete the new vertex, repeat on the tracks left. The PV is the first vertex kept.
Secondary vertices: sPHENIX's SecondaryVertexFinder pair selection (circle intersections of displaced tracks), each
seed fitted with the same vertex fit; the PV is then refitted with the SVs sharing tracks. ACTS's hard cuts are soft
(sigmoids of width cut_softness_*); hard_cuts=True restores them (the check against ACTS).

Written from the published maths (R. Fruehwirth et al., adaptive vertex fitting, J. Phys. G 34 (2007) N343;
G. Piacquadio, CERN-THESIS-2010-027, the helix linearisation) and checked against ACTS v48.2.0 (MPL-2.0); all
lengths in cm (ACTS: mm).
"""
import functools
import math

import torch
import torch.nn as nn
from torch.func import jacrev, vmap

from .common import C_LIGHT, CM_PER_M

K_CM = C_LIGHT / CM_PER_M          # pT [GeV] = K_CM x B [T] x R [cm]
_CURVILINEAR_TOLERANCE = 0.999995  # |direction . z| above which the curvilinear frame uses x instead of z


def _wrap(angle):
    """Angle in [-pi, pi)."""
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


def _soft_below(value, cut, width):
    """1 where value < cut: a step (width 0) or sigmoid((cut - value) / width); NaN -> 0 (with a zero gradient)."""
    finite = ~torch.isnan(value)
    value = torch.where(finite, value, torch.as_tensor(cut, dtype=value.dtype, device=value.device))
    soft = (value < cut).to(value.dtype) if width == 0 else torch.sigmoid((cut - value) / width)
    return torch.where(finite, soft, 0.0)


# ------------------------------------------------------------------------------------------------
# One track's helix (perigee p = (d0, z0, phi, theta, q/p) at the beam line, z reference 0), vmapped below
# ------------------------------------------------------------------------------------------------

def _helix_radius(p, bz):
    """Signed radius rho = sin(theta) / (q/p K Bz) (cm; > 0: clockwise seen from +z)."""
    return torch.sin(p[3]) / (p[4] * K_CM * bz)


def _perigee_at(p, bz, beam, point):
    """The helix's perigee parameters (5,) at the line along z through point (3,) (z0 relative to point), and its
    point of closest approach in x-y (3,)."""
    d0, z0, phi, theta = p[0], p[1], p[2], p[3]
    rho = _helix_radius(p, bz)
    centre = beam + d0 * torch.stack([-torch.sin(phi), torch.cos(phi)]) + rho * torch.stack([torch.sin(phi), -torch.cos(phi)])
    u = point[:2] - centre
    u = u / u.norm()
    pca = centre + rho.abs() * u
    phi_new = torch.atan2(-torch.sign(rho) * u[0], torch.sign(rho) * u[1])
    z = z0 - rho * _wrap(phi_new - phi) / torch.tan(theta)          # z0 + arc length x cot(theta)
    d0_new = (pca - point[:2]) @ torch.stack([-torch.sin(phi_new), torch.cos(phi_new)])
    new = torch.stack([d0_new, z - point[2], phi_new, theta, p[4]])
    return new, torch.cat([pca, z.unsqueeze(0)])


def _impact_3d(p, bz, beam, point, iterations=20, guarded=False):
    """The helix's 3D point of closest approach to point (Newton in the turning angle from the perigee, as ACTS), as
    local coordinates (loc0, loc1) on the curvilinear plane there (axes U = z x T / |.|, V = T x U, T the direction),
    and U, V. guarded: also Newton from the transverse closest approach on the three turns nearest the point in z;
    the closest distance minimum wins (far points, low pT)."""
    d0, z0, phi0, theta = p[0], p[1], p[2], p[3]
    rho, cot = _helix_radius(p, bz), 1.0 / torch.tan(p[3])
    centre = torch.stack([beam[0] - (d0 - rho) * torch.sin(phi0), beam[1] + (d0 - rho) * torch.cos(phi0),
                          z0 + rho * phi0 * cot])
    dx, dy, dz = point - centre

    def step(phi):
        first = rho * (dx * torch.cos(phi) + dy * torch.sin(phi) + (dz + rho * phi * cot) * cot)
        second = rho * (-dx * torch.sin(phi) + dy * torch.cos(phi) + rho * cot * cot)
        return phi - first / second

    def root(phi):
        for _ in range(iterations - 1):
            phi = step(phi.detach())
        return phi.detach()
    phi = root(phi0)
    if guarded:
        distance = lambda a: (centre + rho * torch.stack([-torch.sin(a), torch.cos(a), -cot * a]) - point).norm()  # noqa: E731
        minimum = lambda a: rho * (-dx * torch.sin(a) + dy * torch.cos(a) + rho * cot * cot) > 0  # noqa: E731
        sign = torch.sign(rho)
        transverse = torch.atan2(-sign * dx, sign * dy)
        steep = cot.abs() > 1e-6
        along_z = torch.where(steep, -dz / torch.where(steep, rho * cot, 1.0), phi0)
        turn = torch.round((along_z - transverse) / (2 * math.pi))
        for k in (-1, 0, 1):
            other = root(transverse + 2 * math.pi * (turn + k))
            phi = torch.where(minimum(other) & (distance(other) < distance(phi) * (1 - 1e-9)), other, phi)
    # the root without derivatives, then one step with them: at a Newton root that step's derivative is the root's
    phi = step(phi)
    position = centre + rho * torch.stack([-torch.sin(phi), torch.cos(phi), -cot * phi])
    direction = torch.stack([torch.cos(phi) * torch.sin(theta), torch.sin(phi) * torch.sin(theta), torch.cos(theta)])
    axis = torch.where(direction[2].abs() < _CURVILINEAR_TOLERANCE, direction.new_tensor([0.0, 0.0, 1.0]),
                       direction.new_tensor([1.0, 0.0, 0.0]))
    U = torch.linalg.cross(axis, direction)
    U = U / U.norm()
    V = torch.linalg.cross(direction, U)
    offset = position - point
    return torch.stack([offset @ U, offset @ V]), torch.stack([U, V])


def _with_jacobian(function):
    """function(p, bz, beam, point) -> (out, aux) batched over tracks, with d out / d p."""
    def value_and_aux(p, bz, beam, point):
        out, aux = function(p, bz, beam, point)
        return out, (out, aux)
    batched = vmap(jacrev(value_and_aux, has_aux=True), in_dims=(0, 0, None, 0))

    def run(p, bz, beam, point):
        shape = p.shape[:-1]
        jac, (out, aux) = batched(p.reshape(-1, 5), bz.reshape(-1), beam, point.expand(*shape, 3).reshape(-1, 3))
        return (out.reshape(*shape, *out.shape[1:]), aux.reshape(*shape, *aux.shape[1:]),
                jac.reshape(*shape, *jac.shape[1:]))
    return run


_perigee_with_jacobian = _with_jacobian(_perigee_at)
_impact_with_jacobian = _with_jacobian(_impact_3d)
_guarded_impact_with_jacobian = _with_jacobian(functools.partial(_impact_3d, guarded=True))


def _transport(perigee, covariance, bz, beam, point):
    """Perigee parameters (..., 5) and covariance (..., 5, 5) at the line along z through point (..., 3), and the
    x-y point of closest approach (..., 3); perigee, covariance, bz (...,) at the beam line."""
    new, pca, J = _perigee_with_jacobian(perigee, bz, beam, point)
    return new, J @ covariance @ J.transpose(-1, -2), pca


def _linearise(perigee, covariance, bz, beam, point):
    """
    The tracks linearised at point (..., 3) (Piacquadio's helix Jacobians, as ACTS's HelicalTrackLinearizer):
    perigee parameters q(x, p) ~ c + A x + B p of the vertex x and the momentum p = (phi, theta, q/p) there.
    Returns A, B (..., 5, 3), q - c (..., 5) at the linearisation and the weight G = cov^-1 (..., 5, 5).
    """
    q, cov, pca = _transport(perigee, covariance, bz, beam, point)
    d0, phi, theta = q[..., 0], q[..., 2], q[..., 3]
    rho = torch.sin(theta) / (q[..., 4] * K_CM * bz)
    h = torch.where(rho < 0, -1.0, 1.0).to(rho.dtype)
    X = pca[..., 0] - point[..., 0] + rho * torch.sin(phi)
    Y = pca[..., 1] - point[..., 1] - rho * torch.cos(phi)
    S2 = X * X + Y * Y
    S = S2.sqrt()
    rho_cot, abs_rho_over_s = rho / torch.tan(theta), h * rho / S
    zero, one = torch.zeros_like(X), torch.ones_like(X)
    A = torch.stack([torch.stack([-h * X / S, -h * Y / S, zero], -1),
                     torch.stack([rho_cot * Y / S2, -rho_cot * X / S2, one], -1),
                     torch.stack([-Y / S2, X / S2, zero], -1),
                     torch.stack([zero, zero, zero], -1), torch.stack([zero, zero, zero], -1)], -2)
    B = torch.stack([torch.stack([zero, zero, zero], -1), torch.stack([rho_cot * (1 - abs_rho_over_s), zero, zero], -1),
                     torch.stack([abs_rho_over_s, zero, zero], -1), torch.stack([zero, one, zero], -1),
                     torch.stack([zero, zero, one], -1)], -2)
    del d0
    momentum = torch.stack([phi, theta, q[..., 4]], -1)
    # q - c = A x_pca + B p (ACTS's constant term c = q - A x_pca - B p)
    q_minus_c = (A @ pca.unsqueeze(-1) + B @ momentum.unsqueeze(-1)).squeeze(-1)
    return A, B, q_minus_c, torch.linalg.inv(cov)


def _density_seed(d0, z0, covariance, weight, d0_significance, z0_significance):
    """
    z of the highest track density on the beam line (ACTS's Gaussian track density with the d0-z0 correlation:
    each track a Gaussian in (d0, z0 - z) at d0 = 0, within its z0_significance window; searched from each track's
    z0 with two Gaussian-shaped Newton steps, the first maximum in track order kept). NaN without one.

    d0, z0, weight: (B, T); covariance: (B, T, 2, 2) of (d0, z0); weight 0 = not used.
    """
    c_dd, c_zz, c_dz = covariance[..., 0, 0], covariance[..., 1, 1], covariance[..., 0, 1]
    det = c_dd * c_zz - c_dz * c_dz
    use = (weight > 0) & (c_dd > 0) & (d0 * d0 / c_dd <= d0_significance ** 2) & (c_zz > 0) & (det > 0)
    det = torch.where(use, det, 1.0)
    c0 = -(d0 * d0 * c_zz + z0 * z0 * c_dd + 2 * d0 * z0 * c_dz) / (2 * det)
    c1 = (d0 * c_dz + z0 * c_dd) / det
    c2 = -c_dd / (2 * det)
    disc = c1 * c1 - 4 * c2 * (c0 + 2 * z0_significance ** 2)
    use = use & (disc >= 0)
    root = disc.clamp(min=0).sqrt()
    upper, lower = (-c1 - root) / (2 * c2), (-c1 + root) / (2 * c2)
    c0 = c0 - torch.log(2 * math.pi * det.sqrt()) + torch.log(weight.clamp(min=1e-300))
    c0, c1, c2 = (torch.where(use, v, 0.0).unsqueeze(1) for v in (c0, c1, c2))
    lower, upper = (torch.where(use, v, 0.0).unsqueeze(1) for v in (lower, upper))

    def density(z):
        """Density and its first two derivatives at candidates z (B, C)."""
        z = z.unsqueeze(-1)
        inside = (z > lower) & (z < upper)
        delta = torch.where(inside, torch.exp(c0 + z * (c1 + z * c2)), 0.0)
        slope = c1 + 2 * z * c2
        return delta.sum(-1), (delta * slope).sum(-1), (2 * c2 * delta + slope * delta * slope).sum(-1)

    z, alive = z0.nan_to_num(), use.clone()
    values, candidates = [], []
    for step in range(3):
        y, dy, ddy = density(z)
        alive = alive & (ddy < 0) & (y > 0)
        values.append(torch.where(alive, y, -torch.inf))
        candidates.append(z)
        if step < 2:
            z = z + torch.where(alive, y * dy / torch.where(alive, dy * dy - y * ddy, 1.0), 0.0)
    values, candidates = torch.stack(values, -1).flatten(1), torch.stack(candidates, -1).flatten(1)  # track order
    best = values.argmax(dim=-1, keepdim=True)
    return torch.where(torch.isfinite(values.gather(1, best)), candidates.gather(1, best), torch.nan).squeeze(1)


class VertexFinder(nn.Module):
    """
    Primary (ACTS AMVF) and secondary (sPHENIX pair selection, ACTS vertex fit) vertices from TrackBuilder's tracks.
    Detector values are arguments (defaults: ACTS's AdaptiveMultiVertexFinderAlgorithm, sPHENIX's
    SecondaryVertexFinder), in cm.

    Args:
        beam_position: (x, y), cm (as TrackBuilder's); the seeds' x, y.
        annealing_temperatures, annealing_cut_off: the fit's annealing (ACTS example: one temperature, 1).
        fit_min_weight, fit_max_iterations, max_dist_to_lin_point (cm), max_relative_shift: the multi-vertex fit.
        initial_variance (cm^2): the vertex prior around its seed (x, y, z).
        seed_d0_significance, seed_z0_significance: the Gaussian track-density seeder.
        tracks_max_z_interval (cm), tracks_max_significance: a vertex's tracks: |z0 - z_seed| and the impact
            parameter significance at the seed (soft).
        max_vertex_chi2, maximum_vertex_contamination, max_merge_vertex_significance: keeping a vertex (>= 2 of
            its tracks still unused with compatibility below max_vertex_chi2; contamination; z separation).
        max_vertices: vertices kept per event at most (ACTS: no limit).
        pv_fallback, fallback_preselection: events where the finder keeps no vertex (ACTS: none) are searched again
            with the tracks' preselection relaxed, step by step, to each (tracks_max_z_interval cm,
            tracks_max_significance) of fallback_preselection (keeping a vertex as ACTS does); vertex_from_fallback
            marks them, vertex_fallback_step gives the step (0: as ACTS). Off with hard_cuts.
        cut_softness_cm, cut_softness_sigma, cut_softness_hits: sigmoid widths of the soft cuts in cm, in
            significance and in hits; hard_cuts: ACTS's / sPHENIX's hard cuts instead.
        find_secondary, refit_pv: SVs, and the PV refitted together with them (tracks shared).
        sv_min_track_hits: SV track candidates' TPC hits; displaced from the PV: tracks with silicon hits by
            sPHENIX's impact cuts sv_min_track_dca_xy, sv_min_track_dca_z (cm), TPC-only tracks (their resolution
            is too coarse for those) by compatibility with the PV above sv_min_track_pv_chi2; tracks of the other
            primary vertices are not candidates.
        sv_max_radius, sv_max_projected_dz, sv_max_pair_dca, sv_min_path (cm): pair seeds at circle intersections.
        sv_max_track_chi2: further tracks attached to a seed; sv_max_chi2_per_dof; sv_merge_distance (cm): one SV per
            group closer than this; max_sv_per_event.
    """
    def __init__(
        self,
        beam_position: tuple = (0.0, 0.0),
        annealing_temperatures: tuple = (1.0,),
        annealing_cut_off: float = 9.0,
        fit_min_weight: float = 0.001,
        fit_max_iterations: int = 30,
        max_dist_to_lin_point: float = 0.05,
        max_relative_shift: float = 0.01,
        initial_variance: float = 1.0,
        seed_d0_significance: float = 3.5,
        seed_z0_significance: float = 12.0,
        tracks_max_z_interval: float = 0.1,
        tracks_max_significance: float = 5.0,
        max_vertex_chi2: float = 18.42,
        maximum_vertex_contamination: float = 0.5,
        max_merge_vertex_significance: float = 3.0,
        max_vertices: int = 10,
        pv_fallback: bool = True,
        fallback_preselection: tuple = ((0.3, 7.0), (1.0, 10.0)),
        cut_softness_cm: float = 0.005,
        cut_softness_sigma: float = 0.25,
        cut_softness_hits: float = 1.0,
        hard_cuts: bool = False,
        find_secondary: bool = True,
        refit_pv: bool = True,
        sv_min_track_dca_xy: float = 0.02,
        sv_min_track_dca_z: float = 0.02,
        sv_min_track_hits: int = 20,
        sv_min_track_pv_chi2: float = 9.0,
        sv_max_radius: float = 40.0,
        sv_max_projected_dz: float = 1.0,
        sv_max_pair_dca: float = 0.5,
        sv_min_path: float = 0.2,
        sv_max_track_chi2: float = 9.0,
        sv_max_chi2_per_dof: float = 10.0,
        sv_merge_distance: float = 0.1,
        max_sv_per_event: int = 6,
    ):
        super().__init__()
        if fit_max_iterations < 1 or max_vertices < 1 or not annealing_temperatures:
            raise ValueError("fit_max_iterations and max_vertices must be at least 1, annealing_temperatures non-empty")
        self.beam_position = tuple(beam_position)
        self.annealing_temperatures = tuple(float(t) for t in annealing_temperatures)
        self.annealing_cut_off = annealing_cut_off
        self.fit_min_weight = fit_min_weight
        self.fit_max_iterations = fit_max_iterations
        self.max_dist_to_lin_point = max_dist_to_lin_point
        self.max_relative_shift = max_relative_shift
        self.initial_variance = initial_variance
        self.seed_d0_significance = seed_d0_significance
        self.seed_z0_significance = seed_z0_significance
        self.tracks_max_z_interval = tracks_max_z_interval
        self.tracks_max_significance = tracks_max_significance
        self.max_vertex_chi2 = max_vertex_chi2
        self.maximum_vertex_contamination = maximum_vertex_contamination
        self.max_merge_vertex_significance = max_merge_vertex_significance
        self.max_vertices = max_vertices
        self.pv_fallback = pv_fallback
        self.fallback_preselection = tuple(tuple(float(v) for v in step) for step in fallback_preselection)
        self.cut_softness_cm = cut_softness_cm
        self.cut_softness_sigma = cut_softness_sigma
        self.cut_softness_hits = cut_softness_hits
        self.hard_cuts = hard_cuts
        self.find_secondary = find_secondary
        self.refit_pv = refit_pv
        self.sv_min_track_dca_xy = sv_min_track_dca_xy
        self.sv_min_track_dca_z = sv_min_track_dca_z
        self.sv_min_track_hits = sv_min_track_hits
        self.sv_min_track_pv_chi2 = sv_min_track_pv_chi2
        self.sv_max_radius = sv_max_radius
        self.sv_max_projected_dz = sv_max_projected_dz
        self.sv_max_pair_dca = sv_max_pair_dca
        self.sv_min_path = sv_min_path
        self.sv_max_track_chi2 = sv_max_track_chi2
        self.sv_max_chi2_per_dof = sv_max_chi2_per_dof
        self.sv_merge_distance = sv_merge_distance
        self.max_sv_per_event = max_sv_per_event

    def _width(self, kind):
        """A soft cut's width (0 with hard_cuts)."""
        return 0.0 if self.hard_cuts else {"cm": self.cut_softness_cm, "sigma": self.cut_softness_sigma,
                                            "hits": self.cut_softness_hits}[kind]

    # --------------------------------------------------------------------------------------------
    # Vertex states: (B, K) vertices, their tracks (B, K, T); the tracks' perigee etc. in self._tracks
    # --------------------------------------------------------------------------------------------

    def _new_vertices(self, seeds, links, tracks, prior=None, guarded=False):
        """
        Vertex states for seeds (B, K, 3) with link factors (B, K, T) (a track's membership, 0..1): the 3D impact
        parameters at the seed (for the compatibility), the tracks linearised at the seed, position = seed, the
        prior (default: the seed) with initial_variance; guarded: the guarded 3D impact point (_impact_3d).
        """
        perigee, cov, bz = tracks["perigee"], tracks["covariance"], tracks["bz"]
        beam = perigee.new_tensor(self.beam_position)
        point = seeds.nan_to_num().unsqueeze(2)
        expand = lambda v: v.unsqueeze(1).expand(-1, seeds.shape[1], *v.shape[1:])  # noqa: E731
        impact = _guarded_impact_with_jacobian if guarded else _impact_with_jacobian
        loc, axes, J = impact(expand(perigee), expand(bz), beam, point)
        loc_cov = J @ expand(cov) @ J.transpose(-1, -2)
        finite = seeds.nan_to_num()          # unused slots (NaN seeds) get finite stand-ins (weight 0)
        state = {"seed": seeds, "prior": finite if prior is None else prior.nan_to_num(), "position": finite.clone(),
                 "lin_point": finite.clone(), "links": links, "ip_loc": loc, "ip_axes": axes,
                 "ip_weight": torch.linalg.inv(loc_cov),
                 "covariance": torch.diag_embed(seeds.new_full(seeds.shape, self.initial_variance)),
                 "compatibility": torch.full_like(links, torch.inf), "weight": torch.zeros_like(links),
                 "chi2": seeds.new_zeros(seeds.shape[:2]), "ndf": seeds.new_full(seeds.shape[:2], -3.0)}
        state.update(zip(("A", "B", "q_minus_c", "G"), _linearise(expand(perigee), expand(cov), expand(bz), beam, point)))
        return state

    def _empty_vertices(self, n_events, n_vertices, n_tracks, dtype, device):
        """Unused vertex slots (B, K) in the layout of _new_vertices (finite, invertible stand-ins)."""
        zeros = lambda *shape: torch.zeros(n_events, n_vertices, *shape, dtype=dtype, device=device)  # noqa: E731
        eye = lambda n, *shape: torch.eye(n, dtype=dtype, device=device).expand(  # noqa: E731
            n_events, n_vertices, *shape, n, n).clone()
        B = zeros(n_tracks, 5, 3)
        B[..., 2:, :] = torch.eye(3, dtype=dtype, device=device)
        return {"seed": zeros(3) + torch.nan, "prior": zeros(3), "position": zeros(3), "lin_point": zeros(3),
                "links": zeros(n_tracks), "ip_loc": zeros(n_tracks, 2), "ip_axes": zeros(n_tracks, 2, 3),
                "ip_weight": eye(2, n_tracks), "covariance": eye(3) * self.initial_variance,
                "compatibility": zeros(n_tracks) + torch.inf, "weight": zeros(n_tracks), "chi2": zeros(),
                "ndf": zeros() - 3.0, "A": zeros(n_tracks, 5, 3), "B": B, "q_minus_c": zeros(n_tracks, 5),
                "G": eye(5, n_tracks)}

    def _compatibility(self, state, position):
        """Each track's chi2 (B, K, T) to the vertex at position (B, K, 3), on the 2D frame of its 3D impact
        parameters at the seed (ACTS's 3D vertex compatibility)."""
        shift = ((position - state["seed"]).nan_to_num().unsqueeze(2).unsqueeze(-2) * state["ip_axes"]).sum(-1)
        r = state["ip_loc"] - shift
        return (r.unsqueeze(-2) @ state["ip_weight"] @ r.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    def _relinearise(self, state, which, tracks):
        """Relinearise the tracks of vertices which (B, K) at their lin_point."""
        if not which.any():
            return
        perigee, cov, bz = tracks["perigee"], tracks["covariance"], tracks["bz"]
        beam = perigee.new_tensor(self.beam_position)
        b, k = which.nonzero(as_tuple=True)
        new = _linearise(perigee[b], cov[b], bz[b], beam, state["lin_point"][b, k].unsqueeze(1))
        for key, value in zip(("A", "B", "q_minus_c", "G"), new):
            state[key] = state[key].index_put((b, k), value)

    def _kalman(self, state, weight):
        """
        The vertices (B, K) from their prior and the tracks with weights (B, K, T): ACTS's Kalman vertex update (all
        tracks at once; the same as one by one): position, covariance, chi2 and ndf.
        """
        A, B, G = state["A"], state["B"], state["G"]
        GB = G @ B
        G_B = G - GB @ torch.linalg.inv(B.transpose(-1, -2) @ GB) @ GB.transpose(-1, -2)
        AtGB = A.transpose(-1, -2) @ G_B
        w = weight[..., None, None]
        prior_info = torch.eye(3, dtype=A.dtype, device=A.device) / self.initial_variance
        information = prior_info + (w * AtGB @ A).sum(2)
        rhs = (prior_info @ state["prior"].unsqueeze(-1)).squeeze(-1) + (
            w.squeeze(-1) * (AtGB @ state["q_minus_c"].unsqueeze(-1)).squeeze(-1)).sum(2)
        covariance = torch.linalg.inv(information)
        position = (covariance @ rhs.unsqueeze(-1)).squeeze(-1)
        r = state["q_minus_c"] - (A @ position.unsqueeze(2).unsqueeze(-1)).squeeze(-1)
        track_chi2 = (r.unsqueeze(-2) @ G_B @ r.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        d = position - state["prior"]
        chi2 = (d.unsqueeze(-2) @ prior_info @ d.unsqueeze(-1)).squeeze(-1).squeeze(-1) + (weight * track_chi2).sum(-1)
        return position, covariance, chi2, 2 * weight.sum(-1) - 3

    def _fit(self, state, which, track_weight, tracks, shared=True):
        """
        ACTS's adaptive multi-vertex fit of the vertices which (B, K) (the others keep their state; their stored
        compatibilities still share the tracks): annealed weights, Kalman update, relinearisation beyond
        max_dist_to_lin_point, until the last temperature and a small shift (or fit_max_iterations). shared=False:
        each vertex fitted on its own (its tracks not shared with the others).
        """
        temperatures = self.annealing_temperatures
        running = which.any(dim=1)
        for iteration in range(self.fit_max_iterations):
            if not running.any():
                break
            active = which & running.unsqueeze(1)
            old = state["position"]
            moved = (old[..., :2] - state["lin_point"][..., :2]).norm(dim=-1) > self.max_dist_to_lin_point
            state["lin_point"] = torch.where((active & moved).unsqueeze(-1), old, state["lin_point"])
            state["compatibility"] = torch.where(active.unsqueeze(-1), self._compatibility(state, old),
                                                 state["compatibility"])
            self._relinearise(state, active & moved, tracks)
            # weight = f exp(-chi2/2T) / (exp(-cut/2T) + sum over the track's vertices f_v exp(-chi2_v/2T))
            t2 = 2.0 * temperatures[min(iteration, len(temperatures) - 1)]
            f = state["links"]
            # (unlinked: a large negative stand-in, not -inf, so that the gradient stays finite)
            log_terms = torch.where(f > 0, torch.log(f.clamp(min=1e-300)) - state["compatibility"].clamp(max=1e30) / t2,
                                    -1e30)
            log_denominator = torch.logaddexp(torch.logsumexp(log_terms, dim=1, keepdim=True) if shared else log_terms,
                                              log_terms.new_tensor(-self.annealing_cut_off / t2))
            weight = torch.exp(log_terms - log_denominator).nan_to_num(0.0)
            weight = torch.where(active.unsqueeze(-1), weight, state["weight"])
            state["weight"] = weight
            fit_weight = weight * (weight > self.fit_min_weight) * track_weight.unsqueeze(1)
            position, covariance, chi2, ndf = self._kalman(state, fit_weight)
            for key, value in (("position", position), ("covariance", covariance), ("chi2", chi2), ("ndf", ndf)):
                state[key] = torch.where(active.view(*active.shape, *[1] * (value.dim() - 2)), value, state[key])
            d = (old - state["position"]).unsqueeze(-1)
            shift = (d.transpose(-1, -2) @ torch.linalg.solve(state["covariance"], d)).squeeze(-1).squeeze(-1)
            small = ((shift <= self.max_relative_shift) | ~active).all(dim=1)
            running = running & ~(small & (iteration >= len(temperatures) - 1))

    # --------------------------------------------------------------------------------------------
    # Primary vertices: ACTS's adaptive multi-vertex finder
    # --------------------------------------------------------------------------------------------

    def _links(self, seed, tracks, preselection):
        """Each track's membership (B, T) of a vertex seeded at seed (B, 3): |z0 - z_seed| and the impact significance
        at the seed below preselection (z interval cm, significance) (soft)."""
        q, cov, _ = _transport(tracks["perigee"], tracks["covariance"], tracks["bz"],
                               tracks["perigee"].new_tensor(self.beam_position), seed.nan_to_num().unsqueeze(1))
        significance = (q[..., 0] ** 2 / cov[..., 0, 0] + q[..., 1] ** 2 / cov[..., 1, 1]).clamp(min=1e-30).sqrt()
        dz = (tracks["perigee"][..., 1] - seed[:, 2:]).abs()
        return (_soft_below(dz, preselection[0], self._width("cm"))
                * _soft_below(significance, preselection[1], self._width("sigma"))
                * tracks["input"].to(dz.dtype))

    def _find_primary(self, tracks, track_weight, preselection):
        """ACTS's AMVF over each event's input tracks (weight 0: not used) with the tracks' preselection (z interval
        cm, significance): the vertex states (B, K) and kept (B, K) in creation order."""
        n_events, n_tracks = track_weight.shape
        dtype, device = track_weight.dtype, track_weight.device
        K = self.max_vertices + 1                                   # the kept ones and the one being tried
        perigee, cov = tracks["perigee"], tracks["covariance"]
        state = self._empty_vertices(n_events, K, n_tracks, dtype, device)
        kept = torch.zeros(n_events, K, dtype=torch.bool, device=device)
        order = torch.full((n_events, K), 10 ** 9, dtype=torch.long, device=device)
        pool = tracks["input"].clone()
        alive = pool.sum(-1) >= 2
        events = torch.arange(n_events, device=device)
        for iteration in range(1000):
            alive = alive & (kept.sum(dim=1) < self.max_vertices)
            if not alive.any():
                break
            slot = (~kept).to(torch.int8).argmax(dim=1)
            z = _density_seed(perigee[..., 0], perigee[..., 1], cov[..., :2, :2], track_weight * pool,
                              self.seed_d0_significance, self.seed_z0_significance)
            alive = alive & torch.isfinite(z)
            seed = torch.stack([torch.full_like(z, self.beam_position[0]), torch.full_like(z, self.beam_position[1]),
                                z.nan_to_num()], -1)
            prior = seed
            links = self._links(seed, tracks, preselection)
            # no track: move the seed to the nearest unused track's z0 and try again
            empty = alive & ~(links > 0.5).any(dim=1)
            if empty.any():
                dz = torch.where(pool, (perigee[..., 1] - seed[:, 2:]).abs(), torch.inf)
                nearest = dz.argmin(dim=1)
                seed = torch.where(empty.unsqueeze(-1), torch.stack(
                    [seed[:, 0], seed[:, 1], perigee[events, nearest, 1]], -1), seed)
                links = torch.where(empty.unsqueeze(-1), self._links(seed, tracks, preselection), links)
                alive = alive & (links > 0.5).any(dim=1)
            if not alive.any():
                break
            new = torch.zeros_like(kept)
            new[events[alive], slot[alive]] = True
            # the new vertex's state (its prior: the density seed, also where the seed moved)
            fresh = self._new_vertices(seed.unsqueeze(1), links.unsqueeze(1), tracks, prior.unsqueeze(1))
            for key, value in fresh.items():
                state[key] = torch.where(new.view(*new.shape, *[1] * (value.dim() - 2)), value, state[key])
            order = torch.where(new, iteration, order)
            # the vertices to fit: the new one and every kept vertex connected to it through shared tracks
            linked = state["links"] > 0.5
            fit = new.clone()
            for _ in range(K):
                shares = ((linked & fit.unsqueeze(-1)).any(dim=1, keepdim=True) & linked).any(dim=-1)
                fit = fit | (shares & kept)
            self._fit(state, fit, track_weight, tracks)
            # the new vertex: >= 2 of its tracks still unused and compatible; those leave the pool (if none, the
            # least compatible of its unused tracks, or the unused track nearest the fitted vertex in z)
            new_links = (linked & new.unsqueeze(-1)).any(dim=1)
            compatibility = torch.where(new.unsqueeze(-1), state["compatibility"], 0.0).sum(dim=1)
            compatible = new_links & pool & (compatibility < self.max_vertex_chi2)
            good = alive & (compatible.sum(-1) >= 2)
            remove = compatible & alive.unsqueeze(-1)
            none = alive & ~compatible.any(dim=-1)
            worst = torch.where(new_links & pool & (compatibility > 0), compatibility, -torch.inf).argmax(dim=-1)
            has_worst = (new_links & pool & (compatibility > 0)).any(dim=-1)
            fitted_z = (state["position"][..., 2] * new).sum(1, keepdim=True)
            nearest = torch.where(pool, (perigee[..., 1] - fitted_z).abs(), torch.inf).argmin(dim=-1)
            drop = torch.where(has_worst, worst, nearest)
            remove = remove | (torch.nn.functional.one_hot(drop, n_tracks).bool() & none.unsqueeze(-1))
            pool = pool & ~remove
            # kept: good, contamination <= maximum_vertex_contamination, not within max_merge_vertex_significance
            # sigma in z of another vertex
            w = torch.where(new.unsqueeze(-1), state["weight"], 0.0).sum(dim=1) * new_links
            contamination = (w * (1 - w)).sum(-1) / (w * w).sum(-1).clamp(min=1e-300)
            z_new = (state["position"][..., 2] * new).sum(1, keepdim=True)
            var_new = (state["covariance"][..., 2, 2] * new).sum(1, keepdim=True)
            merged = (kept & ((state["position"][..., 2] - z_new).abs()
                              < self.max_merge_vertex_significance * (state["covariance"][..., 2, 2] + var_new).sqrt())
                      ).any(dim=1)
            keep = good & (contamination <= self.maximum_vertex_contamination) & ~merged
            kept = kept | (new & keep.unsqueeze(-1))
            rejected = new & ~keep.unsqueeze(-1)
            if rejected.any():
                # the vertex goes; the others it was fitted with are refitted without it
                state["links"] = state["links"].masked_fill(rejected.unsqueeze(-1), 0.0)
                state["weight"] = state["weight"].masked_fill(rejected.unsqueeze(-1), 0.0)
                state["compatibility"] = state["compatibility"].masked_fill(rejected.unsqueeze(-1), torch.inf)
                state["seed"] = state["seed"].masked_fill(rejected.unsqueeze(-1), torch.nan)
                order = order.masked_fill(rejected, 10 ** 9)
                self._fit(state, fit & ~rejected & rejected.any(dim=1, keepdim=True), track_weight, tracks)
            alive = alive & (pool.sum(-1) >= 2)
        return state, kept, order

    # --------------------------------------------------------------------------------------------
    # Secondary vertices: sPHENIX's pair selection, the same vertex fit
    # --------------------------------------------------------------------------------------------

    def _sv_seeds(self, tracks, candidate, pv):
        """
        Pair seeds (B, P, 3) and their pair DCA (B, P) of candidate tracks (B, T): the x-y intersections of their
        circles within sv_max_radius, the helix z there within sv_max_projected_dz, the helix tangents there closer
        than sv_max_pair_dca, the midpoint further than sv_min_path from the PV (soft; score (B, P)); and the pairs'
        tracks (B, P, 2).
        """
        perigee, bz = tracks["perigee"], tracks["bz"]
        beam = perigee.new_tensor(self.beam_position)
        d0, z0, phi, theta = perigee[..., 0], perigee[..., 1], perigee[..., 2], perigee[..., 3]
        rho = torch.sin(theta) / (perigee[..., 4] * K_CM * bz)
        centre = beam + d0.unsqueeze(-1) * torch.stack([-torch.sin(phi), torch.cos(phi)], -1) \
            + rho.unsqueeze(-1) * torch.stack([torch.sin(phi), -torch.cos(phi)], -1)
        n_tracks = perigee.shape[1]
        i, j = torch.triu_indices(n_tracks, n_tracks, offset=1, device=perigee.device)
        c1, c2, r1, r2 = centre[:, i], centre[:, j], rho[:, i].abs(), rho[:, j].abs()
        d = (c2 - c1).norm(dim=-1)
        meets = (d > 0) & (d <= r1 + r2) & (d >= (r1 - r2).abs()) & candidate[:, i] & candidate[:, j]
        d = torch.where(meets, d, 1.0)
        a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
        h = (r1 * r1 - a * a).clamp(min=0).sqrt()
        along = (c2 - c1) / d.unsqueeze(-1)
        across = torch.stack([-along[..., 1], along[..., 0]], -1)
        base = c1 + a.unsqueeze(-1) * along
        points = torch.stack([base + h.unsqueeze(-1) * across, base - h.unsqueeze(-1) * across], 2)   # (B, pairs, 2, 2)

        def at(track, xy):
            """Each track's helix at the x-y point xy: position (z by arc length) and unit direction."""
            q = perigee[:, track].unsqueeze(2).expand(-1, -1, 2, -1)
            p_rho, p_centre = rho[:, track].unsqueeze(2), centre[:, track].unsqueeze(2)
            u = xy - p_centre
            u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            t = torch.sign(p_rho).unsqueeze(-1) * torch.stack([u[..., 1], -u[..., 0]], -1)
            phi_at = torch.atan2(t[..., 1], t[..., 0])
            z = q[..., 1] - p_rho * _wrap(phi_at - q[..., 2]) / torch.tan(q[..., 3])
            sin_t = torch.sin(q[..., 3]).unsqueeze(-1)
            return (torch.cat([xy, z.unsqueeze(-1)], -1),
                    torch.cat([sin_t * t, torch.cos(q[..., 3]).unsqueeze(-1)], -1))
        x1, u1 = at(i, points)
        x2, u2 = at(j, points)
        # the tangent lines' closest approach
        w0, b = x1 - x2, (u1 * u2).sum(-1)
        d1, d2 = (u1 * w0).sum(-1), (u2 * w0).sum(-1)
        denominator = (1 - b * b).clamp(min=1e-12)
        s1, s2 = (b * d2 - d1) / denominator, (d2 - b * d1) / denominator
        p1, p2 = x1 + s1.unsqueeze(-1) * u1, x2 + s2.unsqueeze(-1) * u2
        pair_dca, midpoint = (p1 - p2).norm(dim=-1), (p1 + p2) / 2
        cm = self._width("cm")
        score = (meets.unsqueeze(-1).to(d.dtype)
                 * _soft_below(points.norm(dim=-1), self.sv_max_radius, cm)
                 * _soft_below((x1[..., 2] - x2[..., 2]).abs(), self.sv_max_projected_dz, cm)
                 * _soft_below(pair_dca, self.sv_max_pair_dca, cm)
                 * _soft_below(-(midpoint - pv.unsqueeze(1).unsqueeze(1)).norm(dim=-1), -self.sv_min_path, cm))
        n_events = perigee.shape[0]
        pairs = torch.stack([i, j], -1).unsqueeze(1).expand(-1, 2, -1).reshape(-1, 2)
        return (midpoint.reshape(n_events, -1, 3), pair_dca.reshape(n_events, -1), score.reshape(n_events, -1),
                pairs.unsqueeze(0).expand(n_events, -1, -1))

    def _find_secondary(self, tracks, track_weight, pv, pv_state, pv_slot):
        """SV finding after the PV (track_weight: without the other primary vertices' tracks): sv_* outputs,
        displaced_mask, displaced_weight; the SV states and which are valid."""
        n_events, n_tracks = track_weight.shape
        dtype, device = track_weight.dtype, track_weight.device
        K = self.max_sv_per_event
        beam = tracks["perigee"].new_tensor(self.beam_position)
        # candidates (soft): at least sv_min_track_hits TPC hits; displaced from the PV: with silicon hits by
        # sPHENIX's impact cuts (sv_min_track_dca_xy / _z), TPC-only by PV compatibility (sv_min_track_pv_chi2)
        q, _, _ = _transport(tracks["perigee"], tracks["covariance"], tracks["bz"], beam, pv.nan_to_num().unsqueeze(1))
        cm, hits, sigma = self._width("cm"), self._width("hits"), self._width("sigma")
        pv_chi2 = self._compatibility(pv_state, pv.nan_to_num().unsqueeze(1)).squeeze(1)
        displaced = torch.where(
            tracks["n_silicon_hits"] > 0,
            _soft_below(-q[..., 0].abs(), -self.sv_min_track_dca_xy, cm) * _soft_below(-q[..., 1].abs(),
                                                                                    -self.sv_min_track_dca_z, cm),
            _soft_below(-pv_chi2, -self.sv_min_track_pv_chi2, sigma * 2 * math.sqrt(self.sv_min_track_pv_chi2)))
        candidate_weight = (displaced * (track_weight > 0) * tracks["input"].to(dtype)
                            * _soft_below(-(tracks["n_hits"] - tracks["n_silicon_hits"]), -(self.sv_min_track_hits - 0.5),
                                          hits)
                            * torch.isfinite(pv).all(-1, keepdim=True))
        candidate = candidate_weight > 0.5
        if n_tracks >= 2:
            seeds, pair_dca, score, pairs = self._sv_seeds(tracks, candidate, pv)
        else:   # no pairs: one empty stand-in
            seeds = torch.zeros(n_events, 1, 3, dtype=dtype, device=device)
            pair_dca, score = seeds[..., 0] + torch.inf, seeds[..., 0]
            pairs = torch.zeros(n_events, 1, 2, dtype=torch.long, device=device)
        # the K best pairs (smallest pair DCA)
        best = pair_dca.masked_fill(score <= 0.5, torch.inf).argsort(dim=-1)[:, :K]
        if best.shape[1] < K:
            best = torch.cat([best, best.new_zeros(n_events, K - best.shape[1])], 1)
            valid_slot = torch.arange(K, device=device) < score.shape[1]
        else:
            valid_slot = torch.ones(K, dtype=torch.bool, device=device)
        seed_ok = (score.gather(1, best) > 0.5) & valid_slot
        seed = torch.where(seed_ok.unsqueeze(-1), seeds.gather(1, best.unsqueeze(-1).expand(-1, -1, 3)), torch.nan)
        seed_pair_dca = pair_dca.gather(1, best).masked_fill(~seed_ok, torch.nan)
        pair = pairs.gather(1, best.unsqueeze(-1).expand(-1, -1, 2))
        # each seed's tracks: its pair, and candidates compatible with the seed (chi2 <= sv_max_track_chi2, soft)
        state = self._new_vertices(seed, torch.zeros(n_events, K, n_tracks, dtype=dtype, device=device), tracks,
                                   guarded=True)
        compatible = _soft_below(self._compatibility(state, seed), self.sv_max_track_chi2,
                                 self._width("sigma") * 2 * math.sqrt(self.sv_max_track_chi2))
        in_pair = torch.zeros_like(compatible, dtype=torch.bool).scatter_(2, pair, True)
        links = torch.where(in_pair, 1.0, compatible) * candidate_weight.unsqueeze(1) * seed_ok.unsqueeze(-1)
        state["links"] = links
        self._fit(state, seed_ok, track_weight, tracks, shared=False)
        linked = links > 0.5
        n_good = (linked & (state["compatibility"] < self.max_vertex_chi2)).sum(-1)
        chi2_per_dof = state["chi2"] / state["ndf"].clamp(min=1e-12)
        path = state["position"] - pv.unsqueeze(1)
        length = path.norm(dim=-1)
        ok = (seed_ok & (n_good >= 2) & (state["ndf"] > 0) & (chi2_per_dof <= self.sv_max_chi2_per_dof)
              & (length > self.sv_min_path))
        # one SV per group within sv_merge_distance: the lowest chi2/ndf (ties: the lower slot)
        distance = (state["position"].unsqueeze(2) - state["position"].unsqueeze(1)).nan_to_num().norm(dim=-1)
        quality = chi2_per_dof.masked_fill(~ok, torch.inf)
        beats = (quality.unsqueeze(1) < quality.unsqueeze(2)) | ((quality.unsqueeze(1) == quality.unsqueeze(2))
                                                                  & torch.ones(K, K, dtype=torch.bool, device=device).tril(-1))
        ok = ok & ~(ok.unsqueeze(1) & (distance < self.sv_merge_distance) & beats).any(dim=2)
        result = self._sv_outputs(state, ok, pv, pv_state["covariance"][torch.arange(n_events), pv_slot])
        result.update({"sv_pair_dca": seed_pair_dca.masked_fill(~ok, torch.nan), "displaced_mask": candidate,
                       "displaced_weight": track_weight * candidate_weight})
        return result, state, ok

    def _sv_outputs(self, state, ok, pv, pv_covariance):
        """The SV outputs of SV states (B, K) kept where ok, measured from the PV pv (B, 3) with covariance (B, 3, 3):
        the fit, the momentum at the SV (its tracks refitted to it) and the KFParticle-style topology (decay length
        significance with the two vertices' covariances added)."""
        linked = state["links"] > 0.5
        momentum = self._vertex_momentum(state, ok)
        path = state["position"] - pv.unsqueeze(1)
        length = path.norm(dim=-1)
        covariance = state["covariance"] + torch.nan_to_num(pv_covariance).unsqueeze(1)
        unit = path / length.unsqueeze(-1).clamp(min=1e-12)
        sigma = (unit.unsqueeze(-2) @ covariance @ unit.unsqueeze(-1)).squeeze(-1).squeeze(-1).sqrt()
        nan = lambda v: v.masked_fill(~ok.view(*ok.shape, *[1] * (v.dim() - 2)), torch.nan)  # noqa: E731
        return {
            "sv_estimate": nan(state["position"]), "sv_is_valid": ok, "sv_covariance": nan(state["covariance"]),
            "sv_chi2": nan(state["chi2"]), "sv_chi2_per_dof": nan(state["chi2"] / state["ndf"].clamp(min=1e-12)),
            "sv_displacement": nan(length), "sv_decay_length_significance": nan(length / sigma),
            "sv_momentum": nan(momentum),
            "sv_dira": nan((unit * momentum).sum(-1) / momentum.norm(dim=-1).clamp(min=1e-12)),
            "sv_n_tracks": ((state["weight"] >= 0.5) & linked).sum(-1).masked_fill(~ok, 0),
            "sv_track_mask": linked & (state["weight"] >= 0.5) & ok.unsqueeze(-1),
            "sv_track_weight": state["weight"] * ok.unsqueeze(-1),
        }

    def _vertex_momentum(self, state, ok):
        """Sum (B, K, 3) of the momenta (GeV) of each vertex's tracks (weight >= 0.5) refitted to it."""
        A, B, G = state["A"], state["B"], state["G"]
        GB = G @ B
        r = state["q_minus_c"] - (A @ state["position"].nan_to_num().unsqueeze(2).unsqueeze(-1)).squeeze(-1)
        p = (torch.linalg.inv(B.transpose(-1, -2) @ GB) @ GB.transpose(-1, -2) @ r.unsqueeze(-1)).squeeze(-1)
        phi, theta, qop = p.unbind(-1)
        vector = torch.stack([torch.cos(phi) * torch.sin(theta), torch.sin(phi) * torch.sin(theta), torch.cos(theta)],
                             -1) / qop.abs().unsqueeze(-1)
        use = ((state["weight"] >= 0.5) & (state["links"] > 0.5) & ok.unsqueeze(-1)).unsqueeze(-1)
        return torch.where(use, vector.nan_to_num(), 0.0).sum(2)

    # --------------------------------------------------------------------------------------------

    def forward(self, tracks):
        """
        tracks: TrackBuilder's output. Returns the vertex outputs of VertexHead.forward.
        """
        dtype = torch.float64
        bad = tracks["_bad"]
        track_weight = tracks["track_weight"].to(dtype)
        inputs = {"perigee": tracks["track_parameters"][..., :5].to(dtype).nan_to_num(),
                  "covariance": tracks["_covariance"].to(dtype), "bz": tracks["_bz"].to(dtype),
                  "n_hits": tracks["_n_hits"].to(dtype), "n_silicon_hits": tracks["_n_silicon_hits"].to(dtype)}
        inputs["input"] = (track_weight > 0) & ~bad & torch.isfinite(inputs["covariance"]).all(dim=(-2, -1))
        eye = torch.eye(5, dtype=dtype, device=track_weight.device)
        inputs["covariance"] = torch.where(inputs["input"][..., None, None], inputs["covariance"], eye)
        inputs["perigee"] = torch.where(inputs["input"].unsqueeze(-1), inputs["perigee"],
                                        inputs["perigee"].new_tensor([0.0, 0.0, 0.0, math.pi / 2, 1.0]))
        track_weight = track_weight * inputs["input"]
        n_events = track_weight.shape[0]
        events = torch.arange(n_events, device=track_weight.device)

        state, kept, order = self._find_primary(inputs, track_weight,
                                                (self.tracks_max_z_interval, self.tracks_max_significance))
        step = torch.where(kept.any(dim=1), 0, -1)
        for k, preselection in enumerate(self.fallback_preselection if self.pv_fallback and not self.hard_cuts else ()):
            # events without a vertex (only those): searched again with a looser preselection
            missing = (step < 0).nonzero().squeeze(1)
            if not len(missing):
                break
            again, again_kept, again_order = self._find_primary({key: v[missing] for key, v in inputs.items()},
                                                                track_weight[missing], preselection)
            use = again_kept.any(dim=1)
            found = (missing[use],)
            state = {key: value.index_put(found, again[key][use]) for key, value in state.items()}
            kept, order = kept.index_put(found, again_kept[use]), order.index_put(found, again_order[use])
            step = step.index_put(found, step.new_tensor(k + 1))
        pv_slot = order.argmin(dim=1)
        slot_state = {k: v[events, pv_slot] for k, v in state.items()}
        valid, from_fallback = step >= 0, step > 0
        pv = slot_state["position"].masked_fill(~valid.unsqueeze(-1), torch.nan)

        result = {}
        if self.find_secondary:
            pv_state = {k: v.unsqueeze(1) for k, v in slot_state.items()}
            # the other primary vertices' tracks are theirs (weight x (1 - their fit weight))
            others = (state["weight"] * (state["links"] > 0.5) * (kept & (order != order.gather(1, pv_slot[:, None]))
                                                                 ).unsqueeze(-1)).amax(dim=1)
            secondary, sv_state, sv_ok = self._find_secondary(inputs, track_weight * (others < 0.5), pv, pv_state,
                                                              torch.zeros_like(pv_slot))
            result.update(secondary)
            if self.refit_pv and sv_ok.any():
                # the PV and its SVs fitted together (tracks shared); used where the event has an SV
                joint = {k: torch.cat([pv_state[k], sv_state[k]], dim=1) for k in pv_state if k in sv_state}
                which = torch.cat([(valid & sv_ok.any(dim=1)).unsqueeze(1), sv_ok], dim=1)
                self._fit(joint, which, track_weight, inputs)
                refit = which[:, 0]
                for key in slot_state:
                    if key in joint:
                        mask = refit.view(-1, *[1] * (slot_state[key].dim() - 1))
                        slot_state[key] = torch.where(mask, joint[key][:, 0], slot_state[key])
                pv = slot_state["position"].masked_fill(~valid.unsqueeze(-1), torch.nan)
                # the SV outputs from the same refit, measured from the refitted PV (the SV selection is unchanged)
                joint_sv = self._sv_outputs({k: v[:, 1:] for k, v in joint.items()}, sv_ok, pv, slot_state["covariance"])
                for key, value in joint_sv.items():
                    result[key] = torch.where(refit.view(-1, *[1] * (value.dim() - 1)), value, result[key])

        # the PV's outputs; per-track ones NaN for tracks without a fit
        weight = slot_state["weight"] * (slot_state["links"] > 0.5)
        track_chi2 = self._compatibility({k: v.unsqueeze(1) for k, v in slot_state.items()}, pv.nan_to_num().unsqueeze(1)
                                         ).squeeze(1)
        loc, axes, _ = _impact_with_jacobian(inputs["perigee"], inputs["bz"], inputs["perigee"].new_tensor(
            self.beam_position), pv.nan_to_num().unsqueeze(1))
        closest = pv.nan_to_num().unsqueeze(1) + (loc.unsqueeze(-1) * axes).sum(-2)
        per_track = lambda v: v.masked_fill((bad | ~valid.unsqueeze(-1)).view(*bad.shape, *[1] * (v.dim() - 2)),  # noqa: E731
                                            torch.nan)
        result.update({
            "vertex_estimate": pv, "fit_is_valid": valid, "vertex_from_fallback": from_fallback,
            "vertex_fallback_step": step,
            "vertex_covariance": slot_state["covariance"].masked_fill(~valid.view(-1, 1, 1), torch.nan),
            "chi_square": slot_state["chi2"].masked_fill(~valid, torch.nan),
            "vertex_ndf": slot_state["ndf"].masked_fill(~valid, torch.nan),
            "track_vertex_weight": weight.masked_fill(bad, 0.0), "vertex_n_tracks": (weight >= 0.5).sum(-1),
            "track_chi2": per_track(track_chi2), "track_dca": per_track(loc.norm(dim=-1)),
            "track_closest_approach_point": per_track(closest),
        })
        # all primary vertices (pileup), in the order found; each track's (weight >= 0.5, else -1)
        rank = torch.where(kept, order, order.new_tensor(10 ** 9)).argsort(dim=1)
        vertices_kept = kept.gather(1, rank)
        fit_weight = (state["weight"] * (state["links"] > 0.5)).gather(1, rank.unsqueeze(-1).expand_as(state["weight"]))
        fit_weight = fit_weight * vertices_kept.unsqueeze(-1)
        best, index = fit_weight.max(dim=1)
        result.update({
            "primary_vertices": state["position"].gather(1, rank.unsqueeze(-1).expand(-1, -1, 3)).masked_fill(
                ~vertices_kept.unsqueeze(-1), torch.nan),
            "primary_vertex_valid": vertices_kept,
            "primary_vertex_n_tracks": (fit_weight >= 0.5).sum(-1),
            "pv_index": torch.zeros_like(pv_slot),
            "track_primary_vertex": torch.where(best >= 0.5, index, -1),
        })
        return result
