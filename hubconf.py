# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torchvision.models.resnet import resnet50 as _resnet50

dependencies = ["torch", "torchvision"]


def resnet50(pretrained=True, **kwargs):
    """
    ResNet-50 pre-trained with SwAV.

    Note that `fc.weight` and `fc.bias` are randomly initialized.

    Achieves 75.3% top-1 accuracy on ImageNet when `fc` is trained.
    """
    model = _resnet50(pretrained=False, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
            map_location="cpu",
        )
        # removes "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # load weights
        model.load_state_dict(state_dict, strict=False)
    return model
