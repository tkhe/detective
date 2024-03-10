from types import MethodType

import torch
from mmengine.config import Config
from mmengine.runner.checkpoint import load_checkpoint
from mmyolo.models.layers import CSPLayerWithTwoConv
from mmyolo.registry import MODELS


def forward_single(self, x, cls_pred, reg_pred):
    b, _, h, w = x.shape
    cls_logit = cls_pred(x)
    bbox_dist_preds = reg_pred(x)
    return cls_logit, bbox_dist_preds.permute(0, 2, 3, 1)


def forward(self, x):
    x_main = self.main_conv(x)
    x_main = [x_main, x_main[:, self.mid_channels:, ...]]
    x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
    x_main.pop(1)
    return self.final_conv(torch.cat(x_main, 1))


def main():
    cfg = Config.fromstring("_base_ = ['mmyolo::yolov8/yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco.py']", file_format=".py")
    model = MODELS.build(cfg.model)
    model.eval()

    load_checkpoint(model, "yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_101206-b975b1cd.pth", map_location="cpu")

    CSPLayerWithTwoConv.forward = forward
    model.bbox_head.head_module.forward_single = MethodType(forward_single, model.bbox_head.head_module)

    input_names = ["data"]
    output_names = [
        "cls_score_s8",
        "cls_score_s16",
        "cls_score_s32",
        "bbox_pred_s8",
        "bbox_pred_s16",
        "bbox_pred_s32",
    ]

    torch.onnx.export(model, torch.randn(1, 3, 640, 640), "model.onnx", input_names=input_names, output_names=output_names)


if __name__ == "__main__":
    main()
