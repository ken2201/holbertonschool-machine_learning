#!/usr/bin/env python3
"""
Dataset class for Machine Translation (Portuguese → English)
using TED HRLR dataset and pretrained BERT tokenizers
"""

import tensorflow as tf
# TF2 compatibility: enable_eager_execution is default in TF2, patch for TF1 scripts
if not hasattr(tf, 'enable_eager_execution'):
    tf.enable_eager_execution = lambda: None

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    A class to load and prepare the TED HRLR translation dataset for machine
    translation from Portuguese to English.
    """

    def __init__(self):
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
                self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset using pre-trained models
        """
        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
                'neuralmind/bert-base-portuguese-cased', use_fast=True,
                clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
                'bert-base-uncased', use_fast=True,
                clean_up_tokenization_spaces=True)

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                            vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences,
                                                            vocab_size=2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes Portuguese and English sentences into tokens
        """
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        pt_tokens = self.tokenizer_pt.encode(pt_sentence,
                                             add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_sentence,
                                             add_special_tokens=False)

        pt_tokens = [vocab_size_pt] + pt_tokens + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + en_tokens + [vocab_size_en + 1]

        return pt_tokens, en_tokens
