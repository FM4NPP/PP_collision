"""
model.py: VertexHead, vertex reconstruction from track-finder output and the training interface: TrackBuilder
(tracks.py) then VertexFinder (vertices.py).

Inputs: class_probs (B, T, 2), mask_probs (B, N, T), normalized points (B, N, 4), padding_mask (B, N);
B events, N hits, T track slots, K = max_sv_per_event. Positions in cm.
"""
import inspect
import os

import torch.nn as nn

from .tracks import TrackBuilder
from .vertices import VertexFinder


def _setting_names(part):
    return set(inspect.signature(part.__init__).parameters) - {"self"}


class VertexHead(nn.Module):
    """
    TrackBuilder then VertexFinder. Takes both parts' settings as keywords (beam_position goes to both; see
    TrackBuilder and VertexFinder); the parts are self.tracks and self.vertices.
    """
    def __init__(self, **settings):
        super().__init__()
        unknown = sorted(set(settings) - self.setting_names())
        if unknown:
            raise TypeError(f"VertexHead has no settings {unknown}")
        self.tracks = TrackBuilder(**{k: v for k, v in settings.items() if k in _setting_names(TrackBuilder)})
        self.vertices = VertexFinder(**{k: v for k, v in settings.items() if k in _setting_names(VertexFinder)})

    @staticmethod
    def setting_names():
        """The settings VertexHead takes: those of TrackBuilder and VertexFinder."""
        return _setting_names(TrackBuilder) | _setting_names(VertexFinder)

    @classmethod
    def from_detector(cls, path, **overrides):
        """
        A VertexHead from a detector file (scripts/acts/detectors/<name>/<name>.yaml): use_acts when acts.image or
        acts.command is set, its vertex_head block as further settings; keyword arguments override it.
        """
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        base = os.path.dirname(os.path.abspath(path))
        resolve = lambda p: p if p is None else os.path.join(base, os.path.expanduser(p))  # noqa: E731 - absolute stays
        field, acts, hit_errors = d.get("field", {}), d.get("acts", {}) or {}, d.get("hit_errors_cm", {})
        # unset keys (None) take TrackBuilder's / VertexFinder's defaults
        kwargs = {
            "Bz": field.get("bz_tesla"),
            "beam_position": d.get("beam_position_cm"),
            "normalization": d.get("normalization"),
            "hit_sigma_rphi": hit_errors.get("rphi"),
            "hit_sigma_z": hit_errors.get("z"),
            "use_acts": bool(acts.get("image") or acts.get("command")),
            "acts_image": resolve(acts.get("image")),
            "acts_command": acts.get("command"),
            "field_map": resolve(field.get("map")),
            "field_map_kind": field.get("map_kind"),
            "acts_geometry": resolve(acts.get("geometry")),
            "acts_geometry_config": resolve(acts.get("geometry_config")),
            "acts_material_map": resolve(acts.get("material_map")),
            "acts_energy_loss": acts.get("energy_loss"),
            "acts_max_steps": acts.get("max_steps"),
            "acts_threads": acts.get("threads"),
            "acts_particle_pdg": acts.get("particle_pdg"),
            "acts_particle_mass": acts.get("particle_mass"),
            "acts_kalman_refit": acts.get("kalman_refit"),
            "acts_kalman_start_covariance_scale": acts.get("kalman_start_covariance_scale"),
            "acts_kalman_fallback_covariance_scale": acts.get("kalman_fallback_covariance_scale"),
            **(d.get("vertex_head") or {}),
        }
        unknown = sorted(set(d.get("vertex_head") or {}) - cls.setting_names())
        if unknown:
            raise ValueError(f"{path}: vertex_head sets {unknown}, which VertexHead does not take")
        return cls(**{**{k: v for k, v in kwargs.items() if v is not None}, **overrides})

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
            dict (the PV: with find_secondary, refitted with the SV tracks shared where an event has an SV and the
            refit is valid):
                vertex_estimate (B, 3), chi_square, fit_is_valid (B,), vertex_covariance (B, 3, 3); its quality:
                vertex_n_tracks (B,) tracks with fit weight (annealing x z window) >= 0.5, vertex_ndf (B,) 2 sum(weights) - 3.
                With max_vertices > 1: primary_vertices (B, V, 3), primary_vertex_valid, primary_vertex_n_tracks (B, V),
                pv_index (B,) the PV among them, track_primary_vertex (B, T) each track's vertex (-1: none).
                track_parameters (B, T, 6) cm, track_position, track_direction (B, T, 3), track_weight,
                track_quality_mask, track_sigma (B, T, 2) d0, z0 errors (cm) x track_error_scale, track_vertex_weight
                (B, T) each track's annealing weight in the PV fit, track_chi2, track_dca (B, T),
                track_closest_approach_point (B, T, 3); acts_failed (B, T) with use_acts: usable by the hit fit, lost by ACTS;
                acts_refit_failed (B, T) with the Kalman refit: propagated from the hit fit's state (refit failed);
                track_covariance (B, T, 6, 6): the ACTS perigee covariance (mm), with use_acts (x
                acts_kalman_fallback_covariance_scale where the refit failed) or as given.
                With find_secondary: sv_estimate (B, K, 3), sv_is_valid, sv_chi2, sv_chi2_per_dof, sv_displacement,
                sv_n_tracks (B, K), sv_track_mask, sv_track_dca (B, K, T), displaced_mask, displaced_weight (B, T).
        """
        tracks = self.tracks(class_probs, mask_probs, points, padding_mask, track_parameters, track_covariance)
        return {**{k: v for k, v in tracks.items() if not k.startswith("_")}, **self.vertices(tracks)}
