"""Optional live experiment tracking with Weights & Biases.

Enabled only when ``WANDB_API_KEY`` is set in the environment (e.g. a RunPod secret) and ``wandb`` is installed;
otherwise every call is a no-op, so local, Kaggle and test runs are unaffected. The run id is stored in the
experiment directory, so a resumed training run keeps logging to the same W&B run.

Environment: ``WANDB_PROJECT`` (default ``forknet``), ``WANDB_ENTITY`` (optional).
"""

import os
import uuid
import warnings
from pathlib import Path


class Telemetry:
    def __init__(self, cfg, exp_dir, enabled=True, job_type="train"):
        self.run = None
        if not (enabled and os.environ.get("WANDB_API_KEY")):
            return
        # Telemetry must never stop a run: any wandb failure (API change, network, auth) disables it with a warning.
        try:
            import wandb

            exp_dir = Path(exp_dir)
            exp_dir.mkdir(parents=True, exist_ok=True)
            id_file = exp_dir / f"wandb_{job_type}_id.txt"
            run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:8]
            id_file.write_text(run_id)
            name = cfg.get("name", "forknet")
            pod = os.environ.get("RUNPOD_POD_ID")
            self.run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "forknet"),
                entity=os.environ.get("WANDB_ENTITY"),
                id=run_id,
                resume="allow",
                name=f"{name}-{job_type}",
                group=name,
                job_type=job_type,
                config={**cfg, "runpod_pod_id": pod},
                dir=str(exp_dir),
            )
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"W&B telemetry disabled: {e!r}")
            self.run = None

    def log(self, metrics, step):
        if self.run is not None:
            try:
                self.run.log(metrics, step=step)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"W&B log failed: {e!r}")

    def finish(self):
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"W&B finish failed: {e!r}")
            self.run = None
