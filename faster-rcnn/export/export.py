import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.model_zoo import get_config


class RPN(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, x):
        features = self.model.backbone(x)
        features = [features[f] for f in self.model.proposal_generator.in_features]
        x = self.model.proposal_generator.rpn_head(features)
        return x, features[:-1]


class RCNN(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, features):
        features = self.model.roi_heads.box_head(features)
        cls_scores, bbox_preds = self.model.roi_heads.box_predictor(features)
        return F.softmax(cls_scores, dim=1), bbox_preds


def main():
    cfg = get_config("common/models/mask_rcnn_fpn.py")

    [cfg.model.roi_heads.pop(x) for x in ["mask_in_features", "mask_pooler", "mask_head"]]

    cfg.model.backbone.bottom_up.stem.norm = "BN"
    cfg.model.backbone.bottom_up.stages.norm = "BN"

    model = instantiate(cfg.model)
    model.eval()

    DetectionCheckpointer(model).load("model_final_280758.pkl")

    rpn = RPN(model)

    mod = torch.jit.trace(rpn, torch.randn(1, 3, 640, 640))
    torch.jit.save(mod, "rpn.pt")

    rcnn = RCNN(model)

    mod = torch.jit.trace(rcnn, torch.randn(1, 256, 7, 7))
    torch.jit.save(mod, "rcnn.pt")


if __name__ == "__main__":
    main()
