from datetime import datetime


def get_run_name(
    model_id: str,
    learning_rate: float,
    batch_size: int,
    accumulation_steps: int,
    num_epochs: int,
    tags: list[str] | None = None,
    include_timestamp: bool = True,
) -> str:
    """Generate a descriptive run name from training hyperparameters.

    Returns a name like: w2v-bert-2.0_lr1e-04_bs16x2_ep10_frozen_20241021_1530
    """
    parts = [model_id.split("/")[-1]]
    parts.append(f"lr{learning_rate:.0e}")
    if accumulation_steps > 1:
        parts.append(f"bs{batch_size}x{accumulation_steps}")
    else:
        parts.append(f"bs{batch_size}")
    parts.append(f"ep{num_epochs}")
    if tags:
        parts.extend(tags)
    if include_timestamp:
        parts.append(datetime.now().strftime("%Y%m%d_%H%M"))
    return "_".join(parts)
