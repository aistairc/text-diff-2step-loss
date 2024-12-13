from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DiffusionTrainingArguments:
    """
    For Training
    """

    num_diffusion_steps: Optional[int] = field(
        default=1000,
    )
    num_steps_for_loss: Optional[int] = field(
        default=2,
    )
    mask_prob_end: Optional[float] = field(
        default=0.01,
    )
    other_prob_end: Optional[float] = field(
        default=0.1,
    )
    other_prob_scheduler_type: Optional[str] = field(
        default="linear",
    )
    loss_type: Optional[str] = field(
        default="ctc",
    )
    exclude_blank_from_masking: Optional[bool] = field(
        default=True,
    )
    task_name: Optional[str] = field(
        default="Squad",
    )
    freq_noise_drawing: Optional[str] = field(
        default="every",
    )

@dataclass
class DiffusionInferenceArguments:
    """
    For inference
    """

    finetuned_model_path: Optional[str] = field(
        default=None,
    )
    num_inference_steps: Optional[int] = field(
        default=20,
    )
