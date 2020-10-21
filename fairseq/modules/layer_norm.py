# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_layernorm = False


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class TiedLayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()

        self.eps = eps

    def forward(self, to_normalize: torch.Tensor, other: torch.Tensor):
        # TODO more efficient implementation
        # TODO detach?
        other = other.clone()

        to_normalize = to_normalize.clone()
        to_normalize.sub_(to_normalize.mean(0))
        denom = to_normalize.var(dim=0, unbiased=False)
        denom.add_(self.eps)
        denom.sqrt_()
        to_normalize.div_(denom)

        to_normalize.mul_(other.view(other.shape[0] * other.shape[1], -1).std(dim=0, unbiased=False))
        to_normalize.add_(other.mean((0, 1)))

        return to_normalize


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)
