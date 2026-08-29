"""Thin WebUI Adapter for the research pipeline."""

from .adapter import RunSpecification, read_run_state, submit_run

__all__ = ["RunSpecification", "read_run_state", "submit_run"]
