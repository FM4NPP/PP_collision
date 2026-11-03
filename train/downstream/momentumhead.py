"""
momentumhead.py: track parameters (momentum, charge, direction) fitted from track-finder output.

OBSOLETE (2026-11-02): momentum is to be calculated in ACTS, not in this package. Kept, not
deleted, so the momentum task can be picked up again.
The track model (circle and z-line fits) now lives in vertexhead.py, which no longer imports this file;
MomentumHead imports it from there and adds the momentum-only outputs (track_reg_result, radius) here. Obsolete here: MomentumHead as a momentum output, its checkpoints
(save_checkpoint / from_checkpoint), and momentum_reconstruction_trainer.py,
eval_momentum_reconstruction.py, train_momentum_reconstruction.py.

Each track's hits (argmax of mask_probs) get a Taubin circle fit in x-y (pT, charge, circle) and a
line fit of z against signed arc length s along that circle (against x for cosmics) for theta and z
at the beam line; no truth is used. VertexHead's helix fit steps along the circle and z line.

Inputs: mask_probs (B, N, T), normalized points (B, N, 4), padding_mask (B, N). B events, N hits,
T track slots. Positions in cm.

Detector- and dataset-dependent settings are MomentumHead arguments, never inline constants: the field
Bz, the beam position, the hit normalization (TPC_NORMALIZATION by default), the beam-line cut
valid_radius, the noise pT cut and the minimum hits per fit.
"""
import torch
import torch.nn as nn

# the track model lives in vertexhead.py (moved there 2026-11-02 so the vertex fit doesn't depend on this
# obsolete file); imported back, including the helpers, so existing imports from here keep working
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
