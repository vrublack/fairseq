# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import os.path as osp

import numpy as np
import torch

from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset
)
from fairseq.data.tensor_dataset import TensorDataset
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
        parser.add_argument('--seq-embedding-reduction', default='max', choices=['mean', 'max'],
                            help='How to combine the seq length dimension of the model output')

    def __init__(self, args, dictionary):
        super().__init__(args)

        self.args = args
        self.dictionary = dictionary

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.criterion == 'triplet_loss' or args.criterion == 'sentence_ranking', \
            'Must set --criterion=triplet_loss or sentence_ranking'

        if args.criterion == 'sentence_ranking':
            args.num_classes = 2

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

        comps = {}
        comp_names = ['sentences', 'phrases', 'paraphrases']
        for component in comp_names:
            comps[component] = data_utils.load_indexed_dataset(
                osp.join(self.args.data, split, component),
                self.dictionary,
                'raw',
                combine=combine,
            )

        if self.args.criterion == 'sentence_ranking':
            dataset = {
                'id': IdDataset(),
                'net_input1': {
                    'src_tokens': RightPadDataset(comps['sentences'], pad_idx=self.source_dictionary.pad()),
                    'src_lengths': NumelDataset(comps['sentences']),
                    'aux_tokens': RightPadDataset(comps['phrases'], pad_idx=self.source_dictionary.pad()),
                    'aux_lengths': NumelDataset(comps['phrases'])
                },
                'net_input2': {
                    'src_tokens': RightPadDataset(comps['sentences'], pad_idx=self.source_dictionary.pad()),
                    'src_lengths': NumelDataset(comps['sentences']),
                    'aux_tokens': RightPadDataset(comps['paraphrases'], pad_idx=self.source_dictionary.pad()),
                    'aux_lengths': NumelDataset(comps['paraphrases'])
                },
                'target': TensorDataset(torch.tensor([0] * len(comps['sentences']), dtype=torch.long)),
                'nsentences': NumSamplesDataset(),
                'ntokens': NumelDataset(comps['sentences'], reduce=True),
            }
        else:
            dataset = {
                'id': IdDataset(),
                'net_input': {
                },
                'anchor_tokens': RightPadDataset(
                    comps['sentences'],
                    pad_idx=self.source_dictionary.pad(),
                ),
                'positive_tokens': RightPadDataset(
                    comps['phrases'],
                    pad_idx=self.source_dictionary.pad(),
                ),
                'negative_tokens': RightPadDataset(
                    comps['paraphrases'],
                    pad_idx=self.source_dictionary.pad(),
                ),
                'anchor_lengths': NumelDataset(comps['sentences']),
                'positive_lengths': NumelDataset(comps['phrases']),
                'negative_lengths': NumelDataset(comps['paraphrases']),
                'nsentences': NumSamplesDataset(),
                'ntokens': NumelDataset(comps['sentences'], reduce=True),
            }

        dataset = NestedDictionaryDataset(
            dataset,
            sizes=np.array([comps[component].sizes for component in comp_names]).max(0),
        )

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

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        dataset = TokenBlockDataset(
            src_tokens,
            src_lengths,
            block_size=None,  # ignored for "eos" break mode
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode="eos",
        )
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": RightPadDataset(dataset, pad_idx=self.source_dictionary.pad()),
                    "src_lengths": NumelDataset(dataset),
                }
            },
            sizes=[np.array(src_lengths)],
        )

    def build_model(self, args):
        from fairseq import models

        assert args.arch in ('lstm_encoder', 'bow_encoder')

        model = models.build_model(args, self)

        model.set_sequence_embedding_head(
            getattr(args, 'seq_embedding_reduction')
        )

        if args.criterion == 'sentence_ranking':
            model.set_classification_head()

        return model

    def max_positions(self):
        return self.args.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def extract_sequence_embeddings_step(self, sample, model):
        model.remove_classification_head()
        return model(**sample['net_input'])
