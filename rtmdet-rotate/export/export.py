import torch

from mmengine.config import Config
from mmengine.runner.checkpoint import load_checkpoint
from mmyolo.registry import MODELS


def main():
    cfg = Config.fromstring("_base_ = ['mmyolo::rtmdet/rotated/rtmdet-r_tiny_fast_1xb8-36e_dota.py']", file_format=".py")
    model = MODELS.build(cfg.model)
    model.eval()

    load_checkpoint(model, "rtmdet-r_tiny_fast_1xb8-36e_dota_20230228_162210-e8ccfb1c.pth", map_location="cpu")

    input_names = ["data"]
    output_names = [
        "cls_score_s8",
        "cls_score_s16",
        "cls_score_s32",
        "bbox_pred_s8",
        "bbox_pred_s16",
        "bbox_pred_s32",
        "angle_s8",
        "angle_s16",
        "angle_s32",
    ]

    torch.onnx.export(model, torch.randn(1, 3, 1024, 1024), "model.onnx", input_names=input_names, output_names=output_names)


if __name__ = "__main__":
    main()
