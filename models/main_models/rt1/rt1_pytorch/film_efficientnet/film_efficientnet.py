"""EfficientNet models modified with added film layers.

Mostly taken from:
https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
"""
import copy
import math
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Union

import torch
from torch import nn
from torchvision.models._api import Weights
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.efficientnet import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
    EfficientNet_V2_L_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_S_Weights,
    FusedMBConv,
    FusedMBConvConfig,
    MBConv,
    MBConvConfig,
    _efficientnet_conf,
    _MBConvConfig,
)
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.utils import _log_api_usage_once

from rt1_pytorch.film_efficientnet.film_conditioning_layer import FilmConditioning


class MBConvFilm(nn.Module):
    """MBConv or FusedMBConv with FiLM context"""

    def __init__(self, embedding_dim: int, mbconv: Union[MBConv, FusedMBConv]):
        super().__init__()
        self.mbconv = mbconv
        num_channels = mbconv.block[-1][1].num_features
        self.film = FilmConditioning(
            embedding_dim=embedding_dim, num_channels=num_channels
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.mbconv(x)
        x = self.film(x, context)
        return x


class _FilmEfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        include_top: bool = False,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
        embedding_dim: Optional[int] = 512,
    ) -> None:
        """
        EfficientNet V1 and V2 main class with additional FiLM context layer

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            include_top (bool): Whether to include the classification head
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
            embedding_dim (int): The dimension of the embedding space
        """
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[MBConvConfig]"
            )

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                )
                stage.append(
                    MBConvFilm(
                        embedding_dim=embedding_dim,
                        mbconv=block_cnf.block(block_cnf, sd_prob, norm_layer),
                    )
                )
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = (
            last_channel if last_channel is not None else 4 * lastconv_input_channels
        )
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(lastconv_output_channels, num_classes),
                nn.Softmax(dim=1),
            )
        else:
            self.avgpool = nn.Identity()
            self.classifier = nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for feature in self.features:
            for layer in feature:
                if isinstance(layer, MBConvFilm):
                    x = layer(x, context)
                else:
                    x = layer(x)

        x = self.avgpool(x)
        x = torch.squeeze(x, dim=(2, 3))  # squeeze if h = w = 1
        x = self.classifier(x)

        return x


def get_weights(arch: str) -> Weights:
    """
    Returns the default weights for the given EfficientNet model.

    Parameters:
        arch (str): The EfficientNet variant to use. Allowed values are:
            - 'efficientnet_b0'
            - 'efficientnet_b1'
            - 'efficientnet_b2'
            - 'efficientnet_b3'
            - 'efficientnet_b4'
            - 'efficientnet_b5'
            - 'efficientnet_b6'
            - 'efficientnet_b7'
            - 'efficientnet_v2_s'
            - 'efficientnet_v2_m'
            - 'efficientnet_v2_l'

    Returns:
        WeightsEnum: The default weights for the given architecture.

    Raises:
        ValueError: If the given architecture is not supported.
    """

    if arch == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
    elif arch == "efficientnet_b1":
        weights = EfficientNet_B1_Weights.DEFAULT
    elif arch == "efficientnet_b2":
        weights = EfficientNet_B2_Weights.DEFAULT
    elif arch == "efficientnet_b3":
        weights = EfficientNet_B3_Weights.DEFAULT
    elif arch == "efficientnet_b4":
        weights = EfficientNet_B4_Weights.DEFAULT
    elif arch == "efficientnet_b5":
        weights = EfficientNet_B5_Weights.DEFAULT
    elif arch == "efficientnet_b6":
        weights = EfficientNet_B6_Weights.DEFAULT
    elif arch == "efficientnet_b7":
        weights = EfficientNet_B7_Weights.DEFAULT
    elif arch == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.DEFAULT
    elif arch == "efficientnet_v2_m":
        weights = EfficientNet_V2_M_Weights.DEFAULT
    elif arch == "efficientnet_v2_l":
        weights = EfficientNet_V2_L_Weights.DEFAULT
    else:
        raise ValueError(f"Unsupported model type `{arch}`")

    return weights


class FilmEfficientNet(nn.Module):
    def __init__(
        self,
        arch: str,
        include_top: bool = False,
        embedding_dim: int = 512,
        pretrained: Optional[bool] = True,
        weights: Optional[Weights] = None,
        progress: Optional[bool] = True,
        device: Optional[Union[str, torch.device]] = "cuda",
        **kwargs,
    ):
        """Builds a FilmEfficientNet model.

        Args:
            arch (str): The EfficientNet variant to use. Allowed values are:
                - 'efficientnet_b0'
                - 'efficientnet_b1'
                - 'efficientnet_b2'
                - 'efficientnet_b3'
                - 'efficientnet_b4'
                - 'efficientnet_b5'
                - 'efficientnet_b6'
                - 'efficientnet_b7'
                - 'efficientnet_v2_s'
                - 'efficientnet_v2_m'
                - 'efficientnet_v2_l'
            include_top (bool, optional): Whether to include the classification head
            embedding_dim (int, optional): The dimensionality of the output embeddings.
            pretrained (bool, optional): Whether to load pretrained EfficientNet weights.
                Defaults to True.
            weights (WeightsEnum, optional): The pretrained weights to use.
                only allowed if `pretrained==False`. Defaults to None.
            progress (bool, optional): If True, displays a progress bar of the
                download to stderr. Default is True.
            device (torch.device, optional): The device on which the model will be
            **kwargs: parameters passed to the `FilmEfficientNet` class.
        """
        super().__init__()
        norm_layer = None
        if arch == "efficientnet_b0":
            inverted_residual_setting, last_channel = _efficientnet_conf(
                arch, width_mult=1.0, depth_mult=1.0
            )
            dropout = 0.2
            self.output_hw = 7
        elif arch == "efficientnet_b1":
            inverted_residual_setting, last_channel = _efficientnet_conf(
                arch, width_mult=1.0, depth_mult=1.1
            )
            dropout = 0.2
            self.output_hw = 8
        elif arch == "efficientnet_b2":
            inverted_residual_setting, last_channel = _efficientnet_conf(
                arch, width_mult=1.1, depth_mult=1.2
            )
            dropout = 0.3
            self.output_hw = 9
        elif arch == "efficientnet_b3":
            inverted_residual_setting, last_channel = _efficientnet_conf(
                arch, width_mult=1.2, depth_mult=1.4
            )
            dropout = 0.3
            self.output_hw = 10
        elif arch == "efficientnet_b4":
            inverted_residual_setting, last_channel = _efficientnet_conf(
                arch, width_mult=1.4, depth_mult=1.8
            )
            dropout = 0.4
            self.output_hw = 12
        elif arch == "efficientnet_b5":
            inverted_residual_setting, last_channel = _efficientnet_conf(
                arch, width_mult=1.6, depth_mult=2.2
            )
            dropout = 0.4
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
            self.output_hw = 15
        elif arch == "efficientnet_b6":
            inverted_residual_setting, last_channel = _efficientnet_conf(
                arch, width_mult=1.8, depth_mult=2.6
            )
            dropout = 0.5
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
            self.output_hw = 17
        elif arch == "efficientnet_b7":
            inverted_residual_setting, last_channel = _efficientnet_conf(
                arch, width_mult=2.0, depth_mult=3.1
            )
            dropout = 0.5
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
            self.output_hw = 20
        elif arch == "efficientnet_v2_s":
            inverted_residual_setting, last_channel = _efficientnet_conf(arch)
            dropout = 0.2
            norm_layer = partial(nn.BatchNorm2d, eps=1e-03)
            self.output_hw = 12
        elif arch == "efficientnet_v2_m":
            inverted_residual_setting, last_channel = _efficientnet_conf(arch)
            dropout = 0.3
            norm_layer = partial(nn.BatchNorm2d, eps=1e-03)
            self.output_hw = 15
        elif arch == "efficientnet_v2_l":
            inverted_residual_setting, last_channel = _efficientnet_conf(arch)
            dropout = 0.4
            norm_layer = partial(nn.BatchNorm2d, eps=1e-03)
            self.output_hw = 15

        assert (
            weights is None or not pretrained
        ), "Cannot pass in custom weights with pretrained=True"
        weights = get_weights(arch) if pretrained else weights

        if weights is not None:
            _ovewrite_named_param(
                kwargs, "num_classes", len(weights.meta["categories"])
            )

        model = _FilmEfficientNet(
            inverted_residual_setting,
            dropout,
            include_top=include_top,
            last_channel=last_channel,
            norm_layer=norm_layer,
            embedding_dim=embedding_dim,
            **kwargs,
        )

        if weights is not None:
            state_dict = weights.get_state_dict(progress=progress)
            new_state_dict = {}
            for k, v in state_dict.items():
                if ".block" in k:
                    new_state_dict[k.replace(".block", ".mbconv.block")] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(
                new_state_dict,
                strict=False,
            )

        self.model = model.to(device)
        self.preprocess = weights.transforms(antialias=True) if weights else lambda x: x

        self.conv1x1 = nn.Conv2d(
            in_channels=self.model.features[-1].out_channels,
            out_channels=embedding_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding="same",
            bias=False,
            device=device,
        )
        nn.init.kaiming_normal_(self.conv1x1.weight)
        self.film_layer = FilmConditioning(embedding_dim, embedding_dim).to(device)
        self.include_top = include_top
        self.embedding_dim = embedding_dim

    def forward(
        self, image: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if len(image.shape) == 3:
            # Add batch dimension
            image = image.unsqueeze(0)
        assert len(image.shape) == 4, f"Unexpected image shape: {image.shape}"
        if image.shape[-1] == 3:
            # (B, H, W, C) -> (B, C, H, W)
            image = image.permute(0, 3, 1, 2)
        if torch.max(image) >= 1.0:
            # Normalize to [0, 1]
            image = image / 255.0
        assert torch.min(image) >= 0.0 and torch.max(image) <= 1.0
        image = self.preprocess(image)

        if context is not None and self.include_top:
            raise ValueError("Context cannot be passed in if include_top=True")
        elif context is None:
            context = torch.zeros(
                image.shape[0], self.embedding_dim, device=image.device
            )

        features = self.model(image, context)
        if not self.include_top:
            features = self.conv1x1(features)
            features = self.film_layer(features, context)
        return features


def decode_predictions(preds: torch.Tensor, top=5):
    preds = preds.detach().cpu().numpy()
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(_IMAGENET_CATEGORIES[i], pred[i]) for i in top_indices]
        results.append(result)
    return results