"""Shim module: re-exports model classes from core.sgm_model_primitives."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sgm_model_primitives import (
    NNModel, TransformerModel, SGMBaselineModel,
    SGMWithLockingModel, ModelTask, build_tasks_for_model,
    run_model_scenario
)
