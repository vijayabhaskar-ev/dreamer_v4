"""WandB run initialization shared across training entry points.

Centralizes the master-only ``wandb.init(...)`` boilerplate that was previously
duplicated in every ``train_*.py`` script, so each trainer only supplies what
actually differs: its run-name prefix and its config dict.

The distributed seam mirrors ``device_utils``: only the master process opens a
real run; every other process gets a disabled run so ``wandb.log`` calls in
worker code become no-ops.
"""

from __future__ import annotations

import os
from datetime import datetime

import wandb

from device_utils import is_master


def add_wandb_args(parser, *, default_project: str):
    """Register the standard ``--wandb-*`` CLI flags shared by every trainer.

    Centralizes the flag definitions that were previously copy-pasted into each
    trainer's ``build_parser()``. ``default_project`` differs per phase (e.g.
    ``dreamer-v4-tokenizer`` / ``-dynamics`` / ``-agent`` / ``-imagination``), so
    it is a required keyword; every other flag is identical across trainers.

    Note: this gives the tokenizer trainer a ``--wandb-disabled`` flag it did not
    previously have. That is purely additive — the flag defaults to False, so the
    pre-existing offline/online behavior is unchanged unless it is passed.
    """
    group = parser.add_argument_group("wandb")
    group.add_argument("--wandb-project", type=str, default=default_project,
                       help="WandB project name")
    group.add_argument("--wandb-entity", type=str, default=None,
                       help="WandB entity (user/team)")
    group.add_argument("--wandb-name", type=str, default=None,
                       help="WandB run name (auto-generated from the run prefix if unset)")
    group.add_argument("--wandb-offline", action="store_true",
                       help="Run WandB in offline mode")
    group.add_argument("--wandb-disabled", action="store_true",
                       help="Disable WandB logging entirely")
    return parser


def init_wandb(opts, *, run_name_prefix: str, config: dict, resume: bool = False):
    """Initialize a WandB run on the master process (disabled elsewhere).

    Args:
        opts: parsed argparse namespace. Reads ``wandb_project``,
            ``wandb_entity``, ``wandb_name``, ``wandb_offline`` and (optionally)
            ``wandb_disabled``.
        run_name_prefix: prefix for the auto-generated run name; the final name
            is ``f"{run_name_prefix}_{YYYYmmdd_HHMMSS}"`` unless ``--wandb-name``
            is set, in which case that value is used verbatim.
        config: the run config dict logged to WandB (built by the caller, so
            each trainer keeps full control over what it records).
        resume: when True, honor a stable ``WANDB_RUN_ID`` env var so restarts
            append to one continuous run (resume-across-preemptions). When the
            env var is unset this is a no-op, matching the pre-refactor behavior.
    """
    if not is_master():
        wandb.init(mode="disabled")
        return

    run_name = opts.wandb_name or f"{run_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # getattr default keeps the tokenizer trainer working: it has no
    # --wandb-disabled flag, so it falls through to offline/online as before.
    if getattr(opts, "wandb_disabled", False):
        mode = "disabled"
    elif getattr(opts, "wandb_offline", False):
        mode = "offline"
    else:
        mode = "online"

    resume_kwargs = {}
    if resume:
        run_id = os.environ.get("WANDB_RUN_ID")
        if run_id:
            resume_kwargs = {"id": run_id, "resume": "allow"}

    wandb.init(
        project=opts.wandb_project,
        entity=opts.wandb_entity,
        name=run_name,
        config=config,
        mode=mode,
        **resume_kwargs,
    )
