from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torchvision.utils import make_grid
import lightning.pytorch as pl
import lightning.pytorch.loggers as pll
from torchmetrics import Accuracy
import wandb

from backbone import iresnet
from head import AdaFace
from metrics.embedding_roc import EmbeddingAccuracy
from utils import distance


class Model(pl.LightningModule):
    def __init__(
        self,
        backbone: str,
        n_classes: int | None = None,
        normalize: bool = True,
        margin: float = 0.4,
        h: float = 0.333,
        s: float = 64.0,
        t_alpha: float = 1.0,
        distance_fn: Literal["euclidean", "cosine"] = "cosine",
        lr: float = 0.1,
        momentum: float = 0.9,
        lr_milestones: list[int] | None = None,
        lr_gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.distance_fn = distance_fn
        self.lr = lr
        self.momentum = momentum
        self.lr_milestones = lr_milestones if lr_milestones is not None else [8, 12, 14]
        self.lr_gamma = lr_gamma

        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if normalize else nn.Identity()
        self.backbone = iresnet(backbone)
        if n_classes is not None:
            self.head = AdaFace(embedding_size=512, n_classes=n_classes, margin=margin, h=h, s=s, t_alpha=t_alpha)
        else:
            self.head = None

        self.loss = nn.CrossEntropyLoss()

        self.embedding_accuracy = EmbeddingAccuracy(n_folds=10, distance_fn=self.distance_fn)
        self.test_embedding_accuracy = EmbeddingAccuracy(n_folds=10, distance_fn=self.distance_fn)
        self.test_accuracy = Accuracy(task="binary")
        self.test_distances, self.test_labels = [], []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        embeddings, _ = self.backbone(x)
        return embeddings

    def on_train_start(self) -> None:
        if isinstance(self.logger, pll.WandbLogger) and self.trainer.global_rank == 0:
            self.logger.watch(self, log="all", log_graph=True)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        x = self.normalize(x)
        embeddings, norms = self.backbone(x)
        cos_thetas = self.head(embeddings, norms, y)
        loss = self.loss(cos_thetas, y)
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if batch_idx == 0 and self.trainer.current_epoch == 0 and isinstance(self.logger, pll.WandbLogger) and self.trainer.global_rank == 0:
            images = make_grid(x, normalize=True, value_range=(-1, 1)) if self.normalize else make_grid(x)
            self.logger.experiment.log({
                "images/train": [
                    wandb.Image(images, caption="Images"),
                ]
            })
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x1, x2, issame = batch
        x1_embeddings = self(x1)
        x2_embeddings = self(x2)
        self.embedding_accuracy.update(x1_embeddings, x2_embeddings, issame)

    def on_validation_epoch_end(self) -> None:
        accuracy, best_threshold, all_distances = self.embedding_accuracy.compute()
        self.log("accuracy/val", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("best_threshold/val", best_threshold, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if isinstance(self.logger, pll.WandbLogger) and self.trainer.global_rank == 0:
            distances_list = [[d] for d in all_distances]
            table = wandb.Table(data=distances_list, columns=["distances"])
            self.logger.experiment.log({
                "distances/val": wandb.plot.histogram(
                    table,
                    "distances",
                    title="Val distances",
                )
            })
        self.embedding_accuracy.reset()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x1, x2, issame = batch
        x1_embeddings = self(x1)
        x2_embeddings = self(x2)
        self.test_embedding_accuracy.update(x1_embeddings, x2_embeddings, issame)
        distances = distance(x1_embeddings, x2_embeddings, self.distance_fn)
        self.test_distances.extend(distances.cpu().tolist())
        self.test_labels.extend(issame.cpu().tolist())

    def on_test_epoch_end(self) -> None:
        accuracy, best_threshold, all_distances = self.test_embedding_accuracy.compute()
        self.log("accuracy/test", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("best_threshold/test", best_threshold, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if isinstance(self.logger, pll.WandbLogger) and self.trainer.global_rank == 0:
            test_distances = torch.tensor(self.test_distances, device=best_threshold.device)
            test_scores = torch.sigmoid(best_threshold - test_distances).tolist()
            test_scores = [[1 - s, s] for s in test_scores]
            distances_list = [[d] for d in all_distances]
            table = wandb.Table(data=distances_list, columns=["distances"])
            self.logger.experiment.log({
                "distances/test": wandb.plot.histogram(
                    table,
                    "distances",
                    title="Test distances",
                )
            })
            self.logger.experiment.log({
                "roc/test": wandb.plot.roc_curve(self.test_labels, test_scores)
            })
        self.test_distances, self.test_labels = [], []

    def configure_optimizers(self):
        paras_wo_bn, paras_only_bn = self.split_parameters(self.backbone)
        optimizer = torch.optim.SGD(
            [
                {
                    "params": paras_wo_bn + [self.head.kernel],
                    "weight_decay": 5e-4
                },
                {
                    "params": paras_only_bn,
                }
            ],
            lr=self.lr,
            momentum=self.momentum,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=self.lr_gamma)
        return [optimizer], [scheduler]

    def split_parameters(self, module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        params_decay = []
        params_no_decay = []
        for m in module.modules():
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                params_no_decay.extend([*m.parameters()])
            elif len(list(m.children())) == 0:
                params_decay.extend([*m.parameters()])
        assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
        return params_decay, params_no_decay
