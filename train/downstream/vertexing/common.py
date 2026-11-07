"""
common.py: units and the helix geometry both parts use (cm, the hits' unit).
"""
import torch

# units: positions in cm (the hits are in cm); ACTS lengths in mm; pT [GeV] = C_LIGHT x B [T] x R [m]
CM_PER_M, MM_PER_CM, C_LIGHT = 100.0, 10.0, 0.299792458

def _arc_length(x, y, xc, yc, R_s, beam_position):
    """
    Signed arc length along a circle (centre xc, yc; signed radius R_s, sign(R_s) = sign(q Bz)) from its
    closest approach to the beam line (at beam_position in x-y, cm) to the circle point at the angle of
    (x, y); positive in the direction of travel (anticlockwise <=> R_s < 0). All inputs broadcast.
    """
    s = torch.atan2(y - yc, x - xc)
    beam_angle = torch.atan2(-(yc - beam_position[1]), -(xc - beam_position[0]))
    return s.sub_(beam_angle).add_(torch.pi).remainder_(2 * torch.pi).sub_(torch.pi).mul_(-R_s)
def _circle_point(circle_centre, R_s, xy):
    """Each circle's point nearest xy (..., 2) and the unit direction of travel there (anticlockwise <=> R_s < 0)."""
    n = xy - circle_centre
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return circle_centre + R_s.abs().unsqueeze(-1) * n, torch.sign(R_s).unsqueeze(-1) * torch.stack([n[..., 1], -n[..., 0]], -1)
