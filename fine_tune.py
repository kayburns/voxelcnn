#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
import warnings
from datetime import datetime
import os
from os import path as osp
from time import time as tic

import numpy as np
import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader
from voxelcnn.checkpoint import Checkpointer
from voxelcnn.criterions import CrossEntropyLoss
from voxelcnn.datasets import Craft3DDataset
from voxelcnn.evaluators import CCA, MTC, Accuracy
from voxelcnn.models import VoxelCNN
from voxelcnn.summary import Summary
from voxelcnn.utils import Section, collate_batches, setup_logger, to_cuda

from main import *

def build_datasets(args, logger):
    datasets = {}
    for split in ("train", "val", "test"):
        datasets[split] = Craft3DDataset(
            args.data_dir,
            split,
            max_samples=args.max_samples,
            next_steps=10,
            logger=logger,
        )
    return datasets

def build_data_loaders_per_label(args, label, datasets, logger):
    data_loaders = {}
    for split in ("train", "val", "test"):
        dataset = datasets[split]
        indices = dataset.label2flattened_indices[label].copy()
        label_dataset = torch.utils.data.Subset(dataset, indices)
        data_loaders[split] = DataLoader(
            label_dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=not args.cpu_only,
        )
    return data_loaders

def freeze_base_layers(model):
    for name, param in model.named_parameters():
        if 'predictor' not in name: 
            param.requires_grad = False

def reset_last_layers(model):
    model.types_predictor.reset_parameters()
    model.coords_predictor.reset_parameters()

def build_optimizer(args, model):
    no_decay = []
    decay = []
    print('Updating params')
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            if param.requires_grad:
                print('\t%s'%name)
                no_decay.append(param)
        else:
            if param.requires_grad:
                print('\t%s'%name)
                decay.append(param)
    params = [{"params": no_decay, "weight_decay": 0}, {"params": decay}]
    return optim.SGD(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=True,
    )

def main(args):
    # Set log file name based on current date and time
    cur_datetime = datetime.now().strftime("%Y%m%d.%H%M%S")
    log_path = osp.join(args.save_dir, f"log.{cur_datetime}.txt")
    logger = setup_logger(save_file=log_path)
    logger.info(f"Save logs to: {log_path}")

    print("Global setup")
    global_setup(args)

    print("Building dataset")
    datasets = build_datasets(args, logger)

    print("Loading model")
    model = build_model(args, logger)
    freeze_base_layers(model)

    print("Building evaluators")
    evaluators = build_evaluators(args)
    checkpointer = Checkpointer(args.save_dir)

    print("Resuming from model: {args.resume}")
    last_epoch = checkpointer.resume(args.resume, model=model)

    print("Loading label categories")
    with open('label1000.txt', 'r') as f:
        labels = f.readlines()
        labels = [l.strip() for l in labels]

    for label in labels:

        print("Building data loaders for {}".format(label))
        data_loaders = build_data_loaders_per_label(args, label, datasets, logger)
        
        print("Resetting predictor weights for {}".format(label))
        reset_last_layers(model)

        print("Building criterions, optimizer, scheduler for {}".format(label))
        criterion = build_criterion(args)
        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer)

        print("Creating label level checkpointer for last layer weights")
        label_dir = osp.join(args.save_dir, label)
        if not osp.exists(label_dir):
            os.makedirs(label_dir)
        label_checkpointer = Checkpointer(label_dir)

        print("wandb setup for {label}")
        if args.log:
            wandb.init(project="step-visprim", reinit=True)
            wandb.config.label = label

        for epoch in range(last_epoch + 1, args.num_epochs + 1):
            with Section("Training epoch {epoch}", logger=logger):
                train(
                    args,
                    epoch,
                    data_loaders["train"],
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    evaluators,
                    logger,
                )
            with Section(f"Validating epoch {epoch} of {label}", logger=logger):
                # Evaluate on the validation set by the lightweight accuracy metrics
                metrics = evaluate(
                    args, epoch, data_loaders["val"], model, evaluators, logger
                )
                # Use acc@10 as the key metric to select best model
                label_checkpointer.save_last_layers(
                        model, optimizer, scheduler, epoch, metrics["acc@1"])
                metrics_str = "  ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
                metrics = {"val_"+k:v for k,v in metrics.items()}
                if args.log:
                    wandb.log(metrics)
                best_mark = "*" if epoch == checkpointer.best_epoch else ""
                logger.info(f"Finish  epoch: {epoch}  {metrics_str} {best_mark}")

            best_epoch = checkpointer.best_epoch
            with Section(f"Final test with best model from epoch of {label}: {best_epoch}", logger=logger):
                # Load the best model and evaluate all the metrics on the test set
                label_checkpointer.load_last_layers("best", model=model)
                metrics = evaluate(
                    args, best_epoch, data_loaders["test"], model, evaluators, logger
                )

                # Additional evaluation metrics. Takes quite long time to evaluate
                dataset = data_loaders["test"].dataset
                params = {
                    "local_size": dataset.dataset.local_size,
                    "global_size": dataset.dataset.global_size,
                    "history": dataset.dataset.history,
                }
                metrics.update(CCA(**params).evaluate(dataset.dataset, model))
                #metrics.update(MTC(**params).evaluate(dataset.dataset, model))
                metrics_dict = {"best_"+k:v for k,v in metrics.items()}
                if args.log:
                    wandb.log(metrics_dict)

                metrics_str = "  ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
                logger.info(f"Final test from best epoch for {label}: {best_epoch}\n{metrics_str}")


if __name__ == "__main__":
    work_dir = osp.dirname(osp.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Train and evaluate VoxelCNN model on 3D-Craft dataset"
    )
    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default=osp.join(work_dir, "data"),
        help="Path to the data directory",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers for preprocessing",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="When debugging, set this option to limit the number of training samples",
    )
    # Optimizer
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="Weight decay"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    # Scheduler
    parser.add_argument("--step_size", type=int, default=5, help="StepLR step size")
    parser.add_argument("--gamma", type=int, default=0.1, help="StepLR gamma")
    parser.add_argument("--num_epochs", type=int, default=16, help="Total train epochs")
    # Misc
    parser.add_argument(
        "--log", action="store_true", help="Enables wandb logging"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=osp.join(work_dir, "logs"),
        help="Path to a directory to save log file and checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default='best',
        help="'latest' | 'best' | '<epoch number>' | '<path to a checkpoint>'. "
        "Default: None, will not resume",
    )
    parser.add_argument("--cpu_only", action="store_true", help="Only using CPU")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    main(parser.parse_args())
