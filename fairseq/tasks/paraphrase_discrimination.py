# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np

from fairseq import utils
from fairseq.data import (
    ConcatSentencesDataset,
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    TruncateDataset
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import FairseqTask, register_task


logger = logging.getLogger(__name__)


@register_task('paraphrase_discrimination')
class ParaphraseDiscriminationTask(FairseqTask):
    """
    Predict which alternative phrase is the correct one in a sentence

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--no-shuffle', action='store_true')
        parser.add_argument('--shorten-method', default='none',
                            choices=['none', 'truncate', 'random_crop'],
                            help='if not none, shorten sequences that exceed --tokens-per-sample')
        parser.add_argument('--shorten-data-split-list', default='',
                            help='comma-separated list of dataset splits to apply shortening to, '
                                 'e.g., "train,valid" (default: all dataset splits)')
        parser.add_argument('--max-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.criterion == 'triplet_loss', \
            'Must set --criterion=triplet_loss'

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, 'dict.txt'),
            source=True,
        )
        logger.info('dictionary length {}'.format(len(data_dict)))
        return ParaphraseDiscriminationTask(args, data_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(split):
            return os.path.join(self.args.data, split)

        dataset = data_utils.load_indexed_dataset(
            get_path(split),
            self.dictionary,
            'paraphrase',
            combine=combine,
        )

        dataset = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': dataset,
                'src_lengths': NumelDataset(src_tokens, reduce=False),
            },
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
        }

        if not self.args.no_shuffle:
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(dataset))

            dataset = SortDataset(
                dataset,
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        # TODO
        # model.register_triplet_loss_head(
        #     getattr(args, 'ranking_head_name', 'sentence_classification_head'),
        #     num_classes=1,
        # )

        return model

    def max_positions(self):
        return self.args.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
