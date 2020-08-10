# -*- coding: utf-8 -*-

import torch

from typing import Dict


class GeM():
    r"""
    Generalized-mean pooling.
    https://pdfs.semanticscholar.org/a2ca/e0ed91d8a3298b3209fc7ea0a4248b914386.pdf
    ------------------------
    Hyper-Params
    p: float, hyper-parameter for calculating generalized mean. If p = 1, GeM is equal to global average pooling, and
    if p = +infinity, GeM is equal to global max pooling.
    """

    def __init__(self, hps: Dict or None = None):
        self.first_show = True
        self._hyper_params = {
            "p": 2.0,
        }
        super(GeM, self).__init__()

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        r"""
        aggregate feature map to flatten feature
        :param features: dict of torch.tensor, each with shape (N, C, H, W).
        :return: each feature is with shape of (N, C')
        """
        p = self._hyper_params["p"]

        ret = dict()
        fea = features
        if fea.ndimension() == 4:
            fea = fea ** p
            h, w = fea.shape[2:]
            fea = fea.sum(dim=(2, 3)) * 1.0 / w / h
            fea = fea ** (1.0 / p)
        return fea
