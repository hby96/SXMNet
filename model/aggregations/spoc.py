# -*- coding: utf-8 -*-

import torch
import numpy as np

from typing import Dict


class SPoC():
    r"""
    SPoC with center prior.
    c.f. https://arxiv.org/pdf/1510.07493.pdf
    ------------------------
    Hyper-Params
    """

    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        super(SPoC, self).__init__()
        self.first_show = True
        self.spatial_weight_cache = dict()

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        r"""
        aggregate feature map to flatten feature
        :param features: dict of torch.tensor, each with shape (N, C, H, W).
        :return: each feature is with shape of (N, C')
        """
        ret = dict()
        fea = features
        if fea.ndimension() == 4:
            h, w = fea.shape[2:]
            if (h, w) in self.spatial_weight_cache:
                spatial_weight = self.spatial_weight_cache[(h, w)]
            else:
                sigma = min(h, w) / 2.0 / 3.0
                x = torch.Tensor(range(w))
                y = torch.Tensor(range(h))[:, None]
                spatial_weight = torch.exp(-((x - (w - 1) / 2.0) ** 2 + (y - (h - 1) / 2.0) ** 2) / 2.0 / (sigma ** 2))
                if torch.cuda.is_available():
                    spatial_weight = spatial_weight.cuda()
                spatial_weight = spatial_weight[None, None, :, :]
                self.spatial_weight_cache[(h, w)] = spatial_weight
            fea = (fea * spatial_weight).sum(dim=(2, 3))
        return fea