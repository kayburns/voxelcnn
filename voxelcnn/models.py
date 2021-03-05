#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch
from torch import nn


def conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> List[nn.Module]:
    conv = nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size,
        padding=padding,
        stride=stride,
        bias=False,
    )
    bn = nn.BatchNorm3d(out_channels)
    relu = nn.ReLU()
    return [conv, bn, relu]


class VoxelCNN(nn.Module):
    def __init__(
        self,
        local_size: int = 7,
        global_size: int = 21,
        history: int = 3,
        num_block_types: int = 256,
        num_features: int = 16,
    ):
        """ VoxelCNN model

        Args:
            local_size (int): Local context size. Default: 7
            global_size (int): Global context size. Default: 21
            history (int): Number of previous steps considered as inputs. Default: 3
            num_block_types (int): Total number of different block types. Default: 256
            num_features (int): Number of channels output by the encoders. Default: 16
        """
        super().__init__()
        self.local_size = local_size
        self.global_size = global_size
        self.history = history
        self.num_block_types = num_block_types
        self.block_emb_dim = 4
        self.num_features = num_features

        self.block_encoder = nn.Embedding(self.num_block_types, self.block_emb_dim) #EMB
        self.local_encoder = self._build_local_encoder()
        self.global_encoder = self._build_global_encoder()

        self.feature_extractor = nn.Sequential(
            *conv3d(self.num_features * 2, self.num_features, kernel_size=1, padding=0)
        )
        self.coords_predictor = nn.Conv3d(
            self.num_features, 1, kernel_size=1, padding=0
        )
        self.types_predictor = nn.Conv3d(
            #self.num_features, self.num_block_types, kernel_size=1, padding=0
            self.num_features, self.block_emb_dim, kernel_size=1, padding=0 #EMB
        )
        self.block_decoder = nn.Linear(self.block_emb_dim, self.num_block_types) #EMB

        # tie weights of embedding matrices
        self.block_decoder.weight = self.block_encoder.weight #EMB

        self._init_params()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs (dict): A dict of inputs
                ```
                {
                    "local": float tensor of shape (N, C * H, D, D, D),
                    "global": float tensor of shape (N, 1, G, G, G),
                    "center": int tensor of shape (N, 3), the coordinate of the last
                        blocks, optional
                }
                ```
                where N is the batch size, H is the history length, D is the
                local size, and G is the global size.

        Returns:
            A dict of coordinates and types scores
            ```
            {
                "coords": float tensor of shape (N, 1, D, D, D),
                "types": float tensor of shape (N, C, D, D, D),
                "center": int tensor of shape (N, 3), the coordinate of the last blocks.
                    Output only when inputs have "center"
            }
            ```
        """
        # reshape local to work with embedding, EMB
        N, H, D, D, D = inputs["local"].shape
        inputs["local"] = self.block_encoder(inputs["local"])
        inputs["local"] = inputs["local"].reshape((N, -1, D, D, D))
        # for first prediction, local_size is same as global, so we remove pool
        if inputs["local"].shape[-1] == inputs["global"].shape[-1]:
            global_out = self.global_encoder[:6](inputs["global"])
            global_out = self.global_encoder[7:](global_out)
            
            outputs = torch.cat(
                [
                    self.local_encoder(inputs["local"]),
                    global_out,
                ],
                dim=1
            )
        else:
            outputs = torch.cat(
                [
                    self.local_encoder(inputs["local"]),
                    self.global_encoder(inputs["global"]),
                ],
                dim=1,
            )
        outputs = self.feature_extractor(outputs)
        ret = {
            "coords": self.coords_predictor(outputs),
            "types": self.types_predictor(outputs),
        }
        ret["types"] = self.block_decoder(ret["types"].reshape(N, D, D, D, -1)) #EMB
        ret["types"] = ret["types"].reshape(N, -1, D, D, D) #EMB
        if "center" in inputs:
            ret["center"] = inputs["center"]
        return ret

    def _build_local_encoder(self) -> nn.Module:
        layers = conv3d(self.block_emb_dim * self.history, self.num_features) #EMB
        #layers = conv3d(self.num_block_types * self.history, self.num_features)
        for _ in range(3):
            layers.extend(conv3d(self.num_features, self.num_features))
        return nn.Sequential(*layers)

    def _build_global_encoder(self) -> nn.Module:
        layers = conv3d(1, self.num_features)
        layers.extend(conv3d(self.num_features, self.num_features))
        layers.append(
            nn.AdaptiveMaxPool3d((self.local_size, self.local_size, self.local_size))
        )
        layers.extend(conv3d(self.num_features, self.num_features))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if m.bias is None:
                    # Normal Conv3d layers
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                else:
                    # Last layers of coords and types predictions
                    nn.init.normal_(m.weight, mean=0, std=0.001)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
