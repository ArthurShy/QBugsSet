"""Backward-compatible shim for the renamed detection evaluator module."""

try:
    from .detection_evaluator import BugDetectionEvaluator
except ImportError:
    from detection_evaluator import BugDetectionEvaluator


__all__ = ["BugDetectionEvaluator"]
