"""Minimal reward helpers required by examples/model_runs."""

from .math_verify_reward_type_boxed import compute_score_from_data_source as default_compute_score

__all__ = ["default_compute_score"]
