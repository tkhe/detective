import torch

from mmengine.config import Config
from mmengine.runner.checkpoint import load_checkpoint
from mmyolo.registry import MODELS


def main():
    cfg = Config.fromstring("_base_ = ['mmyolo::rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py']", file_format=".py")
    model = MODELS.build(cfg.model)
    model.eval()

    load_checkpoint(model, "rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth", map_location="cpu")

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
