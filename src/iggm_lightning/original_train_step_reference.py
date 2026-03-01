# -*- coding: utf-8 -*-
"""Reference training-flow source imported from currently available IgGM code.

The public repository does not expose a dedicated train loop script with
optimizer/scheduler/DDP/AMP orchestration. The closest authoritative logic is
inference-time denoising in:
- ``IgGM/deploy/ab_design.py::__build_inputs_cm``
- ``IgGM/deploy/ab_design.py::__sample_cm_ss2ss``
- ``IgGM/model/arch/design_model/model.py::forward``

This file mirrors that source flow for Lightning migration traceability.
"""

from __future__ import annotations

from typing import Any, Dict

from IgGM.model import DesignModel


def build_inputs_cm_ref(plm_featurizer, diffuser, prot_data_curr: Dict[str, Any], idx_step: int):
    """Reference copy of the CM input build path (no formula/schedule changes)."""

    prot_data_pert = diffuser.run(prot_data_curr, idx_step)
    inputs = DesignModel.featurize(plm_featurizer, prot_data_pert)
    return inputs
