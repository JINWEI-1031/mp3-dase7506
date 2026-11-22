"""Pure frozen compiler; no controller, filesystem or label access."""
from .interface import compile_candidates

__all__ = ["compile_candidates"]
