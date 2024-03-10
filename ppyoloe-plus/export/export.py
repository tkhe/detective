from types import MethodType

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner.checkpoint import load_checkpoint
from mmyolo.models.layers import RepVGGBlock
from mmyolo.registry import MODELS


def forward_single(self, x, cls_stem, cls_pred, reg_stem, reg_pred):
    avg_feat = F.adaptive_avg_pool2d(x, (1, 1))
    cls_logit = cls_pred(cls_stem(x, avg_feat) + x)
    bbox_dist_preds = reg_pred(reg_stem(x, avg_feat))
    return cls_logit, bbox_dist_preds.permute(0, 2, 3, 1)


def main():
    cfg = Config.fromstring("_base_ = ['mmyolo::ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco.py']", file_format=".py")
    model = MODELS.build(cfg.model)
    model.eval()

    load_checkpoint(model, "ppyoloe_plus_s_fast_8xb8-80e_coco_20230101_154052-9fee7619.pth", map_location="cpu")

    model.bbox_head.head_module.forward_single = MethodType(forward_single, model.bbox_head.head_module.forward_single)

    for module in model.modules():
        if isinstance(module, RepVGGBlock):
            module.switch_to_deploy()
            assert hasattr(module, "rbr_reparam")

    mod = torch.jit.trace(model, torch.randn(1, 3, 640, 640))
    torch.jit.save(mod, "model.pt")


if __name__ == "__main__":
    main()
