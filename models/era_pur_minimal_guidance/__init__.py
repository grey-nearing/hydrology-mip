"""ERA (Evolving Research Agent) PUR Minimal Guidance winning model (Node 135)."""

from models.era_pur_minimal_guidance.model import DualFiLMLSTM, StatEncoder
from models.era_pur_minimal_guidance.runner import RUNNER_DEFAULTS, run_fold

__all__ = ["DualFiLMLSTM", "StatEncoder", "RUNNER_DEFAULTS", "run_fold"]
