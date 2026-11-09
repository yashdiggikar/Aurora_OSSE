"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import dataclasses
from typing import Generator

import torch

from aurora.batch import Batch
from aurora.model.aurora import Aurora

__all__ = ["rollout"]


def rollout(model: Aurora, batch: Batch, steps: int) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions.

    Args:
        model (:class:`aurora.Aurora`): The model to roll out.
        batch (:class:`aurora.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.

    Yields:
        :class:`aurora.Batch`: The prediction after every step.
    """
    # Ensure everything is already of the right form.
    batch = model.batch_transform_hook(batch)  # This might modify the available variables.
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    for step in range(steps):
        pred = model.forward(batch)

        # ===== PRINTS AFTER EACH PREDICTION =====
        try:
            t = pred.metadata.time[0]
        except Exception:
            t = None

        try:
            levels = pred.metadata.atmos_levels
            nlev = len(levels)
        except Exception:
            levels = None
            nlev = None

        print(f"\n🟦 Step {step}/{steps-1} predicted"
              + (f" | time={t}" if t is not None else "")
              + (f" | atmos_levels={nlev}" if nlev is not None else ""))

        # surf vars
        surf_items = []
        for k, v in pred.surf_vars.items():
            surf_items.append(f"{k}:{tuple(v.shape)}")
        print("   🌊 surf_vars:", ", ".join(surf_items) if surf_items else "(none)")

        # atmos vars
        atmos_items = []
        for k, v in pred.atmos_vars.items():
            atmos_items.append(f"{k}:{tuple(v.shape)}")
        print("   🌪️  atmos_vars:", ", ".join(atmos_items) if atmos_items else "(none)")

        yield pred

        # Add the appropriate history so the model can be run on the prediction.
        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )
