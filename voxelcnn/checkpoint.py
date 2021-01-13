#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
from glob import glob
from os import path as osp
from typing import Any, Dict, Optional

import torch
from torch import nn, optim


class Checkpointer(object):
    def __init__(self, root_dir: str):
        """ Save and load checkpoints. Maintain best metrics

        Args:
            root_dir (str): Directory to save the checkpoints
        """
        super().__init__()
        self.root_dir = root_dir
        self.best_metric = -1
        self.best_epoch = None

    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        epoch: int,
        metric: float,
    ):
        if self.best_metric < metric:
            self.best_metric = metric
            self.best_epoch = epoch
            is_best = True
        else:
            is_best = False

        os.makedirs(self.root_dir, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
            },
            osp.join(self.root_dir, f"{epoch:02d}.pth"),
        )

        if is_best:
            shutil.copy(
                osp.join(self.root_dir, f"{epoch:02d}.pth"),
                osp.join(self.root_dir, "best.pth"),
            )

    def save_last_layers(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        epoch: int,
        metric: float,
    ):
        if self.best_metric < metric:
            self.best_metric = metric
            self.best_epoch = epoch
            is_best = True
        else:
            is_best = False

        os.makedirs(self.root_dir, exist_ok=True)
        last_layers = {k:v for k,v in model.state_dict().items() if 'predictor' in k}
        torch.save(
            {
                "model": last_layers,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
            },
            osp.join(self.root_dir, f"{epoch:02d}.pth"),
        )

        if is_best:
            shutil.copy(
                osp.join(self.root_dir, f"{epoch:02d}.pth"),
                osp.join(self.root_dir, "best.pth"),
            )

    def load(
        self,
        load_from: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict[str, Any]:
        if torch.cuda.is_available():
            ckp = torch.load(self._get_path(load_from))
        else:
            ckp = torch.load(
                    self._get_path(load_from),
                    map_location=torch.device('cpu')
                )

        if model is not None:
            model.load_state_dict(ckp["model"])
        if optimizer is not None:
            optimizer.load_state_dict(ckp["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(ckp["scheduler"])
        return ckp

    def load_last_layers(
        self,
        load_from: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict[str, Any]:
        if 'best' not in load_from:
            load_from = load_from + '/best'
        label_path = self._get_path(load_from)
        if torch.cuda.is_available():
            last_layers = torch.load(label_path)
        else:
            last_layers = torch.load(label_path, map_location='cpu')
        model_dict = model.state_dict()
        model_dict.update(last_layers["model"])
        if model is not None:
            model.load_state_dict(model_dict)
        if optimizer is not None:
            optimizer.load_state_dict(last_layers["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(last_layers["scheduler"])

    def resume(
        self,
        resume_from: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ) -> int:
        ckp = self.load(
            resume_from, model=model, optimizer=optimizer, scheduler=scheduler
        )
        self.best_epoch = ckp["best_epoch"]
        self.best_metric = ckp["best_metric"]
        return ckp["epoch"]

    def _get_path(self, load_from: str) -> str:
        if load_from == "best":
            return osp.join(self.root_dir, "best.pth")
        if load_from == "latest":
            return sorted(glob(osp.join(self.root_dir, "[0-9]*.pth")))[-1]
        if load_from.isnumeric():
            return osp.join(self.root_dir, f"{int(load_from):02d}.pth")
        return osp.join(self.root_dir, load_from+'.pth')
