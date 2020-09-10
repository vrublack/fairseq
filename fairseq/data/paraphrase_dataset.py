import os

from fairseq.data import FairseqDataset
from functools import lru_cache
import numpy as np


class ParaphraseDataset(FairseqDataset):
    """Dataset that has a sentence with an alternative paraphrase"""

    SEPARATOR = ' ||| '
    PHRASE_TOK = '<PHR>'
    GAP_TOK = '<GAP>'

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False,
                 prepend_phrase_token=True, insert_gap_token=False):
        self.sentence_tokens_list = []
        self.phrase_tokens_list = []
        self.paraphrase_tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order

        for symbol in [self.PHRASE_TOK, self.GAP_TOK]:
            dictionary.add_symbol(symbol)

        self.read_data(path, dictionary)
        self.size = len(self.sentence_tokens_list)
        self.prepend_phrase_token = prepend_phrase_token
        self.insert_gap_token = insert_gap_token

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # "phrase" is the original phrase, "paraphrase" the alternative one
                insert_pos, phrase, paraphrase, sentence = tuple(line.strip().split(self.SEPARATOR))
                insert_pos = int(insert_pos)

                # remove original phrase
                gap_filler = self.GAP_TOK if self.insert_gap_token else ''
                sentence = sentence[:insert_pos] + ' ' + gap_filler + ' ' + sentence[insert_pos + len(phrase):]
                sentence_tokens = dictionary.encode_line(
                    sentence, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()

                if self.prepend_phrase_token:
                    phrase = self.PHRASE_TOK + ' ' + phrase
                    paraphrase = self.PHRASE_TOK + ' ' + paraphrase

                phrase_tokens = dictionary.encode_line(
                    phrase, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                paraphrase_tokens = dictionary.encode_line(
                    paraphrase, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()

                self.sentence_tokens_list.append(sentence_tokens)
                self.phrase_tokens_list.append(phrase_tokens)
                self.paraphrase_tokens_list.append(paraphrase_tokens)
                self.sizes.append(len(sentence_tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.sentence_tokens_list[i], self.phrase_tokens_list[i], self.paraphrase_tokens_list

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)
