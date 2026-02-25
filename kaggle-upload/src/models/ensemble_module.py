import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy
from src.models.cnn_module import CNNLitModule
from pathlib import Path
from src.utils.wandb_utils import download_checkpoints_from_files

class EnsembleLitModule(LightningModule):
    def __init__(
        self,
        run_paths: list,
        num_classes: int = 10
    ):
        super().__init__()
        ckpt_paths = download_checkpoints_from_files(run_paths)
        
        self.models = nn.ModuleList()
        for ckpt_path in ckpt_paths:
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
            if "state_dict" in ckpt_data:
                ckpt_data["state_dict"] = {
                    k.replace("_orig_mod.", ""): v
                    for k, v in ckpt_data["state_dict"].items()
                }
            
            fixed_ckpt_path = str(ckpt_path) + ".fixed.ckpt"
            torch.save(ckpt_data, fixed_ckpt_path)
            
            model = CNNLitModule.load_from_checkpoint(fixed_ckpt_path)
            self.models.append(model)

        for model in self.models:
            model.freeze()
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        all_outputs = torch.stack([model(x) for model in self.models])
        avg_output = torch.mean(all_outputs, dim=0)
        preds = torch.argmax(avg_output, dim=1)
        return preds

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        self.log("test/acc", self.test_acc(preds, y), on_epoch=True, prog_bar=True)