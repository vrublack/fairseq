# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
import torch.nn.functional as F


@register_criterion('triplet_loss')
class TripletLossCriterion(FairseqCriterion):

    def __init__(self, task, margin):
        super().__init__(task)
        self.margin = margin

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--margin')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        assert (
                hasattr(model, 'sequence_embedding_head')
                and model.sequence_embedding_head
        ), 'model must provide sequence embedding head for --criterion=triplet_loss'

        anchor = model(model(**sample['net_input']), src_tokens=sample['sentence_tokens'],
                       src_lengths=sample['sentence_lengths'])
        positive = model(model(**sample['net_input']), src_tokens=sample['phrase_tokens'],
                         src_lengths=sample['phrase_lengths'])
        negative = model(model(**sample['net_input']), src_tokens=sample['paraphrase_tokens'],
                         src_lengths=sample['paraphrase_lengths'])

        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        loss = F.relu(distance_positive - distance_negative + self.margin).sum()
        sample_size = anchor.size(0)

        logging_output = {
            'loss': loss.detach(),
            'distance_positive': distance_positive.detach().sum(),
            'distance_negative': distance_negative.detach().sum(),
            'ntokens': sample['ntokens'],
            'nsentences': sample['sentence_tokens'].size(0),
            'sample_size': sample_size
        }

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        distance_positive_sum = sum(log.get('distance_positive', 0) for log in logging_outputs)
        distance_negative_sum = sum(log.get('distance_negative', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('distance_positive', distance_positive_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('distance_negative', distance_negative_sum / sample_size / math.log(2), sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
