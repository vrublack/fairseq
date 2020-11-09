import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    register_model,
    register_model_architecture, FairseqEncoderModel,
)
from fairseq.models.lstm import Embedding, LSTMSequenceEmbeddingHead, LSTMClassificationHead
from fairseq.modules import FairseqDropout

DEFAULT_MAX_POSITIONS = 1e5

CLASSIFICATION_HEAD_RANKING = 'ranking'
CLASSIFICATION_HEAD_STANDARD = 'sentence_classification_head'

@register_model('bow_encoder')
class BOWEncoderModel(FairseqEncoderModel):
    """
    Bag of words model (just sums/averages the word embeddings)
    """
    def __init__(self, args, encoder):
        super().__init__(encoder)

        self.args = args
        self.sequence_embedding_head = None
        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained embedding')
        parser.add_argument('--freeze-embed', action='store_true',
                            help='freeze embeddings')

        parser.add_argument('--seq-embedding-reduction', default='max', choices=['mean', 'max'],
                            help='How to combine the seq length dimension of the model output')
        parser.add_argument('--classification-head-hidden', type=int,
                            help='hidden dimension in classification head or -1 to have no inner layer at all')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.freeze_embed:
            pretrained_embed.weight.requires_grad = False

        encoder = BOWEncoder(
            dictionary=task.source_dictionary,
            embed=pretrained_embed,
            dropout=args.dropout
        )

        return cls(args, encoder)

    def set_sequence_embedding_head(self):
        self.sequence_embedding_head = LSTMSequenceEmbeddingHead(self.args.seq_embedding_reduction)

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        if name == CLASSIFICATION_HEAD_RANKING:
            self.classification_heads[name] = \
                LSTMClassificationHead(2 * self.args.encoder_embed_dim, self.args.classification_head_hidden,
                                       1, 'relu', self.args.dropout)
        elif name == CLASSIFICATION_HEAD_STANDARD:
            self.classification_heads[name] = \
                LSTMClassificationHead(self.args.encoder_embed_dim, self.args.classification_head_hidden,
                                       num_classes, 'relu', self.args.dropout)
        else:
            raise ValueError('Unknown classification head: ' + name)

        if self.sequence_embedding_head is None:
            self.set_sequence_embedding_head()

    def remove_classification_head(self):
        self.classification_heads.clear()

    def forward(
            self,
            src_tokens,
            src_lengths,
            aux_tokens=None,
            aux_lengths=None,
            classification_head_name=None,  # ignored
            features_only=False
    ):
        x, padding_mask = self.encoder(src_tokens, src_lengths=src_lengths)

        if self.sequence_embedding_head is not None:
            x = self.sequence_embedding_head(x, padding_mask)

        if aux_tokens is not None and CLASSIFICATION_HEAD_RANKING in self.classification_heads:
            assert self.sequence_embedding_head is not None, "Classification head requires sequence embedding head"

            x_aux, padding_mask_aux = self.encoder(aux_tokens, src_lengths=aux_lengths)
            x_aux = self.sequence_embedding_head(x_aux, padding_mask_aux)
            # combine main and auxiliary embeddings
            x = self.classification_heads[CLASSIFICATION_HEAD_RANKING](torch.cat((x, x_aux), dim=1))
            # add dummy because the sentence ranking loss expects this
            x = x, None
        elif CLASSIFICATION_HEAD_STANDARD in self.classification_heads:
            assert self.sequence_embedding_head is not None, "Classification head requires sequence embedding head"

            x = self.classification_heads[CLASSIFICATION_HEAD_STANDARD](x)

        if features_only:
            x = x, None

        return x


class BOWEncoder(FairseqEncoder):
    def __init__(
            self, dictionary, embed, dropout
    ):
        super().__init__(dictionary)
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.embed_tokens = embed
        self.padding_idx = dictionary.pad()
        self.hidden_size = embed.embedding_dim

    def forward(
            self,
            src_tokens: Tensor,
            src_lengths: Tensor
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of
                shape `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of
                shape `(batch)`
        """

        # do almost nothing, just embedding layer

        x = self.embed_tokens(src_tokens)
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return x, encoder_padding_mask

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return DEFAULT_MAX_POSITIONS


@register_model_architecture('bow_encoder', 'bow_encoder')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.freeze_embed = getattr(args, 'freeze_embed', False)
    args.classification_head_hidden = getattr(args, 'classification_head_hidden', args.encoder_embed_dim)
