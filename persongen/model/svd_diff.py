import tqdm.autonotebook as tqdm

import torch
import einops


# noinspection PyPep8Naming
class SVDLinear(torch.nn.Linear):
    def __init__(self, linear: torch.nn.Linear, initialized: bool = True, scale: float = 1.0):
        torch.nn.Linear.__init__(
            self, in_features=linear.in_features, out_features=linear.out_features,
            bias=linear.bias is None, device=linear.weight.device, dtype=linear.weight.dtype
        )

        if not initialized:
            linear.weight = self.weight
            linear.bias = self.bias
        else:
            self.weight = linear.weight
            self.bias = linear.bias

        U, S, Vh = torch.linalg.svd(linear.weight.float(), full_matrices=False)
        self.bias = linear.bias

        self.register_buffer('U', U)
        self.register_buffer('S', S)
        self.register_buffer('Vh', Vh)

        # initialize to 0 for smooth tuning
        self.delta = torch.nn.Parameter(torch.zeros_like(self.S))

        del self.weight
        if self.bias is not None:
            self.bias.requires_grad = False

        self.scale = scale

    def _get_weight(self):
        if hasattr(self, 'weight_updated'):
            return self.weight_updated
        weights = (
            self.U @
            torch.diag(torch.nn.functional.relu(self.S + self.scale * self.delta)) @
            self.Vh
        )
        return weights

    def forward(self, x: torch.Tensor, *args: list, **kwargs: dict):
        weight_updated = self._get_weight()
        return torch.nn.functional.linear(x, weight_updated, bias=self.bias)


# noinspection PyPep8Naming
class SVDConv2d(torch.nn.Conv2d):
    def __init__(self, convolution: torch.nn.Conv2d, initialized: bool = True, scale: float = 1.0):
        # noinspection PyTypeChecker
        torch.nn.Conv2d.__init__(
            self, in_channels=convolution.in_channels, out_channels=convolution.out_channels,
            kernel_size=convolution.kernel_size, stride=convolution.stride, padding=convolution.padding,
            dilation=convolution.dilation, groups=convolution.groups, bias=convolution.bias is None,
            padding_mode=convolution.padding_mode, device=convolution.weight.device, dtype=convolution.weight.dtype,
        )

        if not initialized:
            convolution.weight = self.weight
            convolution.bias = self.bias
        else:
            self.weight = convolution.weight
            self.bias = convolution.bias

        weight_reshaped = einops.rearrange(convolution.weight.float(), 'co cin h w -> co (cin h w)')
        U, S, Vh = torch.linalg.svd(weight_reshaped, full_matrices=False)
        self.bias = convolution.bias

        self.register_buffer('U', U)
        self.register_buffer('S', S)
        self.register_buffer('Vh', Vh)

        # initialize to 0 for smooth tuning
        self.delta = torch.nn.Parameter(torch.zeros_like(self.S))

        self.h, self.w = self.weight.shape[2:]
        del self.weight
        if self.bias is not None:
            self.bias.requires_grad = False

        self.scale = scale

    def _get_weight(self):
        if hasattr(self, 'weight_updated'):
            return self.weight_updated
        weights = (
            self.U @
            torch.diag(torch.nn.functional.relu(self.S + self.scale * self.delta)) @
            self.Vh
        )
        weights = einops.rearrange(
            weights, 'co (cin h w) -> co cin h w',
            cin=self.in_channels, h=self.h, w=self.w
        )
        return weights

    def forward(self, x: torch.Tensor, *args: list, **kwargs: dict):
        weight_updated = self._get_weight()
        return torch.nn.functional.conv2d(
            x, weight_updated, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


# noinspection PyPep8Naming
class SVDConv1d(torch.nn.Conv1d):
    def __init__(self, convolution: torch.nn.Conv1d, initialized: bool = True, scale: float = 1.0):
        # noinspection PyTypeChecker
        torch.nn.Conv1d.__init__(
            self, in_channels=convolution.in_channels, out_channels=convolution.out_channels,
            kernel_size=convolution.kernel_size, stride=convolution.stride, padding=convolution.padding,
            dilation=convolution.dilation, groups=convolution.groups, bias=convolution.bias is None,
            padding_mode=convolution.padding_mode, device=convolution.weight.device, dtype=convolution.weight.dtype,
        )

        if not initialized:
            convolution.weight = self.weight
            convolution.bias = self.bias
        else:
            self.weight = convolution.weight
            self.bias = convolution.bias

        weight_reshaped = einops.rearrange(convolution.weight.float(), 'co cin h w -> co (cin h w)')
        U, S, Vh = torch.linalg.svd(weight_reshaped, full_matrices=False)
        self.bias = convolution.bias

        self.register_buffer('U', U)
        self.register_buffer('S', S)
        self.register_buffer('Vh', Vh)

        # initialize to 0 for smooth tuning
        self.delta = torch.nn.Parameter(torch.zeros_like(self.S))

        self.h, self.w = self.weight.shape[2:]
        del self.weight
        if self.bias is not None:
            self.bias.requires_grad = False

        self.scale = scale

    def _get_weight(self):
        if hasattr(self, 'weight_updated'):
            return self.weight_updated
        weights = (
            self.U @
            torch.diag(torch.nn.functional.relu(self.S + self.scale * self.delta)) @
            self.Vh
        )
        weights = einops.rearrange(
            weights, 'co (cin h w) -> co cin h w',
            cin=self.in_channels, h=self.h, w=self.w
        )
        return weights

    def forward(self, x: torch.Tensor, *args: list, **kwargs: dict):
        weight_updated = self._get_weight()
        return torch.nn.functional.conv1d(
            x, weight_updated, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


# noinspection PyPep8Naming
class SVDEmbedding(torch.nn.Embedding):
    def __init__(self, embedding: torch.nn.Embedding, initialized: bool = True, scale: float = 1.0):
        # noinspection PyTypeChecker
        torch.nn.Embedding.__init__(
            self,
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
            padding_idx=embedding.padding_idx,
            max_norm=embedding.max_norm,
            norm_type=embedding.norm_type,
            scale_grad_by_freq=embedding.scale_grad_by_freq,
            sparse=embedding.sparse,
            _freeze=not embedding.weight.requires_grad,
            device=embedding.weight.device,
            dtype=embedding.weight.dtype
        )

        if not initialized:
            embedding.weight = self.weight
        else:
            self.weight = embedding.weight

        U, S, Vh = torch.linalg.svd(embedding.weight.float(), full_matrices=False)

        self.register_buffer('U', U)
        self.register_buffer('S', S)
        self.register_buffer('Vh', Vh)

        # initialize to 0 for smooth tuning
        self.delta = torch.nn.Parameter(torch.zeros_like(self.S))

        del self.weight

        self.scale = scale

    def _get_weight(self):
        if hasattr(self, 'weight_updated'):
            return self.weight_updated
        weight = (
            self.U @
            torch.diag(torch.nn.functional.relu(self.S + self.scale * self.delta)) @
            self.Vh
        )
        return weight

    def forward(self, x: torch.Tensor, *args: list, **kwargs: dict):
        weight_updated = self._get_weight()
        return torch.nn.functional.embedding(
            x, weight_updated, padding_idx=self.padding_idx, max_norm=self.max_norm,
            norm_type=self.norm_type, scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse
        )


# noinspection PyPep8Naming
class SVDLayerNorm(torch.nn.LayerNorm):
    def __init__(self, layer_norm: torch.nn.LayerNorm, initialized: bool = True, scale: float = 1.0):
        # noinspection PyTypeChecker
        torch.nn.LayerNorm.__init__(
            self,
            normalized_shape=layer_norm.normalized_shape,
            eps=layer_norm.eps,
            elementwise_affine=layer_norm.elementwise_affine,
            device=layer_norm.weight.device,
            dtype=layer_norm.weight.dtype,
        )

        if not initialized:
            layer_norm.weight = self.weight
            layer_norm.bias = self.bias
        else:
            self.weight = layer_norm.weight
            self.bias = layer_norm.bias

        U, S, Vh = torch.linalg.svd(layer_norm.weight.unsqueeze(0).float(), full_matrices=False)

        self.register_buffer('U', U)
        self.register_buffer('S', S)
        self.register_buffer('Vh', Vh)

        # initialize to 0 for smooth tuning
        self.delta = torch.nn.Parameter(torch.zeros_like(self.S))

        del self.weight
        if self.bias is not None:
            self.bias.requires_grad = False

        self.scale = scale

    def _get_weight(self):
        if hasattr(self, 'weight_updated'):
            return self.weight_updated
        weight = (
            self.U @
            torch.diag(torch.nn.functional.relu(self.S + self.scale * self.delta)) @
            self.Vh
        ).squeeze(0)
        return weight

    def forward(self, x: torch.Tensor, *args: list, **kwargs: dict):
        weight_updated = self._get_weight()
        return torch.nn.functional.layer_norm(
            x, normalized_shape=self.normalized_shape, weight=weight_updated, bias=self.bias, eps=self.eps
        )


# noinspection PyPep8Naming
class SVDGroupNorm(torch.nn.GroupNorm):
    def __init__(self, group_norm: torch.nn.GroupNorm, initialized: bool = True, scale: float = 1.0):
        # noinspection PyTypeChecker
        torch.nn.GroupNorm.__init__(
            self,
            num_groups=group_norm.num_groups,
            num_channels=group_norm.num_channels,
            eps=group_norm.eps,
            affine=group_norm.affine,
            device=group_norm.weight.device,
            dtype=group_norm.weight.dtype
        )

        if not initialized:
            group_norm.weight = self.weight
            group_norm.bias = self.bias
        else:
            self.weight = group_norm.weight
            self.bias = group_norm.bias

        U, S, Vh = torch.linalg.svd(group_norm.weight.unsqueeze(0).float(), full_matrices=False)

        self.register_buffer('U', U)
        self.register_buffer('S', S)
        self.register_buffer('Vh', Vh)

        # initialize to 0 for smooth tuning
        self.delta = torch.nn.Parameter(torch.zeros_like(self.S))

        del self.weight
        if self.bias is not None:
            self.bias.requires_grad = False

        self.scale = scale

    def _get_weight(self):
        if hasattr(self, 'weight_updated'):
            return self.weight_updated
        weight = (
            self.U @
            torch.diag(torch.nn.functional.relu(self.S + self.scale * self.delta)) @
            self.Vh
        ).squeeze(0)
        return weight

    def forward(self, x: torch.Tensor, *args: list, **kwargs: dict):
        weight_updated = self._get_weight()
        return torch.nn.functional.group_norm(
            x, num_groups=self.num_groups, weight=weight_updated, bias=self.bias, eps=self.eps
        )


def setup_module_for_svd_diff(module, scale=1.0, deltas_path=None, qkv_only=False, fuse=False):
    module.requires_grad_(False)

    if deltas_path is not None:
        weights_dict = torch.load(deltas_path, map_location='cpu', weights_only=True)
    else:
        weights_dict = None

    for name, submodule in tqdm.tqdm(module.named_modules(), desc='Process modules'):
        attributes_names = dir(submodule)
        if isinstance(submodule, torch.nn.ModuleList):
            attributes_names += [str(idx) for idx in range(len(submodule))]

        for attribute_name in tqdm.tqdm(attributes_names, desc='Process attributes:', total=len(attributes_names), leave=False, disable=True):
            if qkv_only and not (
                'to_q' in attribute_name or
                'to_k' in attribute_name or
                'to_v' in attribute_name or
                'q_proj' in attribute_name or
                'k_proj' in attribute_name or
                'v_proj' in attribute_name
            ):
                continue

            attribute = getattr(submodule, attribute_name)

            if isinstance(attribute, torch.nn.Conv1d):
                setattr(submodule, attribute_name, SVDConv1d(attribute, initialized=True, scale=scale))
            elif isinstance(attribute, torch.nn.Conv2d):
                setattr(submodule, attribute_name, SVDConv2d(attribute, initialized=True, scale=scale))
            elif isinstance(attribute, torch.nn.LayerNorm):
                setattr(submodule, attribute_name, SVDLayerNorm(attribute, initialized=True, scale=scale))
            elif isinstance(attribute, torch.nn.GroupNorm):
                setattr(submodule, attribute_name, SVDGroupNorm(attribute, initialized=True, scale=scale))
            elif isinstance(attribute, torch.nn.Linear):
                setattr(submodule, attribute_name, SVDLinear(attribute, initialized=True, scale=scale))
            elif isinstance(attribute, torch.nn.Embedding):
                setattr(submodule, attribute_name, SVDEmbedding(attribute, initialized=True, scale=scale))

            if isinstance(
                    attribute,
                    (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.Linear, torch.nn.Embedding)
            ):
                if weights_dict is not None or fuse:
                    svd_module = getattr(submodule, attribute_name)

                if weights_dict is not None:
                    # noinspection PyUnboundLocalVariable
                    svd_module.load_state_dict(
                        {'delta': weights_dict['.'.join([name, attribute_name, 'delta']).strip('.')]}, strict=False
                    )
                if fuse:
                    # noinspection PyProtectedMember
                    svd_module.weight_updated = torch.nn.Parameter(svd_module._get_weight())
                    del svd_module.U, svd_module.S, svd_module.Vh, svd_module.delta
