# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from . import BaseWrapperDataset


class TensorDataset(BaseWrapperDataset):

    def __init__(self, torch_tensor, sizes=None):
        super().__init__(torch_tensor)
        self._sizes = sizes
        self.dtype = torch_tensor.dtype

    def __iter__(self):
        for x in self.dataset:
            yield x

    def collater(self, samples):
        return torch.tensor(samples, dtype=self.dtype)

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def set_epoch(self, epoch):
        pass
