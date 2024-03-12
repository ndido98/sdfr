from __future__ import annotations

from lightning import LightningModule, Trainer

import torch
import lightning.pytorch.callbacks as plc


class Exporter(plc.Callback):
    def __init__(self, output_file: str) -> None:
        super().__init__()
        self.output_file = output_file

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        example_input = torch.rand((4, 3, 112, 112)).to(pl_module.device)
        torch.onnx.export(
            torch.jit.script(pl_module),
            example_input,
            self.output_file,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )