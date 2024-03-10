from types import MethodType

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner.checkpoint import load_checkpoint
from mmdet.models.dense_heads.centernet_head import CenterNetHead
from mmdet.registry import MODELS


def forward_single(self, x):
    center_heatmap_pred = self.heatmap_head(x).sigmoid()
    wh_pred = self.wh_head(x)
    offset_pred = self.offset_head(x)
    heatmap_max = F.max_pool2d(center_heatmap_pred, kernel_size=3, stride=1, padding=1)
    keep = (center_heatmap_pred - heatmap_max).float() + 1e-9
    keep = F.relu(keep)
    keep = keep * 1e9
    center_heatmap_pred = center_heatmap_pred * keep
    return center_heatmap_pred, wh_pred, offset_pred


def main():
    cfg = Config.fromstring("_base_ = ['mmdet::centernet/centernet_r18_8xb16-crop512-140e_coco.py']", file_format=".py")
    model = MODELS.build(cfg.model)
    model.eval()

    load_checkpoint(model, "centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth", map_location="cpu")

    model.bbox_head.forward_single = MethodType(forward_single, model.bbox_head)

    input_names = ["data"]
    output_names = [
        "heatmap",
        "wh",
        "offset",
    ]

    torch.onnx.export(model, torch.randn(1, 3, 640, 640), "model.onnx", input_names=input_names, output_names=output_names)


if __name__ == "__main__":
    main()
