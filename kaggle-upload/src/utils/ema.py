# src/utils/ema.py
from copy import deepcopy
from typing import Optional, Union

import torch
from lightning import Callback, LightningModule, Trainer
import wandb


class EMACallback(Callback):
    """Exponential Moving Average callback.
    
    Maintains a shadow copy of model weights updated as:
        ema_weights = decay * ema_weights + (1 - decay) * model_weights
    """

    def __init__(
        self,
        decay: float = 0.999,
        apply_ema_every_n_steps: int = 1,
        start_step: int = 0,
        save_ema_weights_in_callback_state: bool = True,
        evaluate_ema_weights_instead: bool = True,
    ):
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {decay}")

        self.decay = decay
        self.apply_ema_every_n_steps = apply_ema_every_n_steps
        self.start_step = start_step
        self.save_ema_weights_in_callback_state = save_ema_weights_in_callback_state
        self.evaluate_ema_weights_instead = evaluate_ema_weights_instead

        self._ema_weights: Optional[dict] = None
        self._original_weights: Optional[dict] = None

    def _init_ema_weights(self, module: LightningModule) -> None:
        self._ema_weights = deepcopy(module.state_dict())

    @torch.no_grad()
    def _update_ema(self, module: LightningModule) -> None:
        if self._ema_weights is None:
            self._init_ema_weights(module)
            return

        current_state = module.state_dict()
        for key, ema_val in self._ema_weights.items():
            if key in current_state:
                current_val = current_state[key].float()
                ema_val.copy_(self.decay * ema_val + (1.0 - self.decay) * current_val)

    def _swap_to_ema_weights(self, module: LightningModule) -> None:
        self._original_weights = deepcopy(module.state_dict())
        module.load_state_dict(self._ema_weights)

    def _restore_original_weights(self, module: LightningModule) -> None:
        if self._original_weights is not None:
            module.load_state_dict(self._original_weights)
            self._original_weights = None

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._init_ema_weights(pl_module)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int
    ) -> None:
        global_step = trainer.global_step
        if (
            global_step >= self.start_step
            and global_step % self.apply_ema_every_n_steps == 0
        ):
            self._update_ema(pl_module)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.evaluate_ema_weights_instead and self._ema_weights is not None:
            if not trainer.sanity_checking:
                self._swap_to_ema_weights(pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (
            self.evaluate_ema_weights_instead
            and self._ema_weights is not None
            and not trainer.sanity_checking
        ):
            epoch = trainer.current_epoch
            step = trainer.global_step
            
            ckpt_path = f"{trainer.default_root_dir}/checkpoints/ema.ckpt"
            trainer.save_checkpoint(ckpt_path)
            
            if trainer.logger:
                wandb.save(
                    ckpt_path,
                    base_path=trainer.default_root_dir,
                    policy="now"
                )

        if self._original_weights is not None:
            self._restore_original_weights(pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.evaluate_ema_weights_instead and self._ema_weights is not None:
            self._swap_to_ema_weights(pl_module)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.evaluate_ema_weights_instead and self._original_weights is not None:
            self._restore_original_weights(pl_module)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._ema_weights is not None:
            self._swap_to_ema_weights(pl_module)
            trainer.save_checkpoint(
                filepath=f"{trainer.default_root_dir}/checkpoints/ema_final.ckpt"
            )
            self._restore_original_weights(pl_module)

    def state_dict(self) -> dict:
        if self.save_ema_weights_in_callback_state:
            return {"ema_weights": self._ema_weights}
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        self._ema_weights = state_dict.get("ema_weights")