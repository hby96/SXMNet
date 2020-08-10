# -*- coding: utf-8 -*-

import torch

from typing import Dict


class Crow():
    r"""
    Cross-dimensional Weighting for Aggregated Deep Convolutional Features.
    c.f. https://arxiv.org/pdf/1512.04065.pdf
    ------------------------
    Hyper-Params
    spatial_a: float, hyper-parameter for calculating spatial weight
    spatial_b: float, hyper-parameter for calculating spatial weight
    """

    def __init__(self, hps: Dict or None = None):
        self.first_show = True
        self._hyper_params = {
            "spatial_a": 2.0,
            "spatial_b": 2.0,
        }
        super(Crow, self).__init__()

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        r"""
        aggregate feature map to flatten feature
        :param features: dict of torch.tensor, each with shape (N, C, H, W).
        :return: each feature is with shape of (N, C')
        """
        spatial_a = self._hyper_params["spatial_a"]
        spatial_b = self._hyper_params["spatial_b"]

        fea = features
        if fea.ndimension() == 4:
            spatial_weight = fea.sum(dim=1, keepdim=True)
            z = (spatial_weight ** spatial_a).sum(dim=(2, 3), keepdim=True)
            z = z ** (1.0 / spatial_a)
            print((spatial_weight / z)[0])
            spatial_weight = (spatial_weight / z) ** (1.0 / spatial_b)
            print(spatial_weight[0])

            c, w, h = fea.shape[1:]
            nonzeros = (fea!=0).float().sum(dim=(2, 3)) / 1.0 / (w * h) + 1e-6
            channel_weight = torch.log(nonzeros.sum(dim=1, keepdim=True) / nonzeros)

            print(fea[0, :, :, :])
            fea = fea * spatial_weight
            fea = fea.sum(dim=(2, 3))
            fea = fea * channel_weight
            print(fea[0, :])
            assert False

        return fea
