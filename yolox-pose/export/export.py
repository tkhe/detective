import copy
from types import MethodType

import torch
from mmengine.config import Config
from mmengine.runner.checkpoint import load_checkpoint
from mmyolo.registry import MODELS


def focus_forward(self, x):
    patch_top_left = x[..., ::2, ::2]
    patch_top_right = x[..., ::2, 1::2]
    patch_bot_left = x[..., 1::2, ::2]
    patch_bot_right = x[..., 1::2, 1::2]
    x = torch.cat(
        (
            patch_top_left,
            patch_bot_left,
            patch_top_right,
            patch_bot_right,
        ),
        dim=1,
    )
    return x


def backbone_forward(self, x):
    outs = []
    for i, layer_name in enumerate(self.layers):
        layer = getattr(self, layer_name)
        x = layer(x)
        if i == 0:
            x = self.stem.conv(x)
        if i in self.out_indices:
            outs.append(x)

    return tuple(outs)


def main():
    cfg = Config.fromstring("_base_ = ['mmyolo::yolox/pose/yolox-pose_tiny_8xb32-300e-rtmdet-hyp_coco.py']", file_format=".py")
    model = MODELS.build(cfg.model)
    model.eval()

    load_checkpoint(model, "yolox-pose_tiny_8xb32-300e-rtmdet-hyp_coco_20230427_080351-2117af67.pth", map_location="cpu")

    model.backbone.stem.forward = MethodType(focus_forward, model.backbone.stem)
    model.backbone.forward = MethodType(backbone_forward, model.backbone)

    mod = torch.jit.trace(model, torch.randn(1, 3, 640, 640))
    torch.jit.save(mod, "model.pt")


if __name__ == "__main__":
    main()
