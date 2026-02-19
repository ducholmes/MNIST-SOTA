import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy
from cnn_module import CNNLitModule
import wandb
import os
from pathlib import Path

class EnsembleLitModule(LightningModule):
    def __init__(
        self,
        artifact_paths: list,
        num_classes: int = 10
    ):
        super().__init__()
        ckpt_paths = self._download_checkpoints(artifact_paths)
        self.models = nn.ModuleList([
            CNNLitModule.load_from_checkpoint(ckpt)
            for ckpt in ckpt_paths
        ])
        for model in self.models:
            model.freeze()
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    def _download_checkpoints(self, artifact_paths: list) -> list:
        api = wandb.Api()
        local_paths = []

        for artifact_path in artifact_paths:
            artifact = api.artifact(artifact_path)
            # download về thư mục tạm
            download_dir = artifact.download(
                root=f"checkpoints/ensemble/{artifact_path.split('/')[2]}"
            )
            # tìm file .ckpt trong thư mục vừa download
            ckpt_file = list(Path(download_dir).glob("*.ckpt"))[0]
            local_paths.append(str(ckpt_file))

        return local_paths

    def forward(self, x):
        all_outputs = torch.stack([model(x) for model in self.models])
        avg_output = torch.mean(all_outputs, dim=0)
        preds = torch.argmax(avg_output, dim=1)
        return preds

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        self.log("test/acc", self.test_acc(preds, y), on_epoch=True, prog_bar=True)