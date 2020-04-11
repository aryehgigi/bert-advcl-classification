# coding=utf-8
# Copyright 2019 Hao WANG, Shanghai University, KB-NLP team.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import re
import logging
import os
import sys
from io import open
from collections import Counter
import tempfile
import subprocess

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score


logger = logging.getLogger(__name__)

cleaning_map = {'-RRB-': ')', '-LRB-': '(', '-LSB-': '[', '-RSB-': ']', '-LCB-': '{', '-RCB-': '}',
               '&nbsp;': ' ', '&quot;': "'", '--': '-', '---': '-'}

MAX_SEQ_LEN = 128


def clean_tokens(tokens):
    return [cleaning_map.get(x, x) for x in tokens]


class InputExample(object):
    def __init__(self, guid, text, args_char_offset, label=None):
        self.guid = guid
        self.text = text
        self.arg1_char_offset = args_char_offset[0]
        self.arg2_char_offset = args_char_offset[1]
        self.pred_char_offset = args_char_offset[3]  # TODO - this is 3 because we have the main predicate in location 2
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, args_indices, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.args_indices = args_indices
        self.label_id = label_id


def convert_examples_to_features(examples, tokenizer):
    features = []
    
    for example in examples:
        tokenized = tokenizer.encode_plus(example.text, return_offsets_mapping=True, pad_to_max_length=True)
        
        # find the tokens of interesting args
        tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
        for i, (tok, offsets) in enumerate(zip(tokens, tokenized['offset_mapping'])):
            if not offsets:
                assert ((tok == "[CLS]") or (tok == "[SEP]"))
                continue
            c_s, c_e = offsets
            if c_s == example.arg1_char_offset:
                arg1 = i
            elif c_s == example.arg2_char_offset:
                arg2 = i
            elif c_s == example.pred_char_offset:
                pred = i
        
        assert arg1 is not None and arg2 is not None and pred is not None
        
        features.append(InputFeatures(
            input_ids=tokenized['input_ids'] + ([tokenizer.pad_token_id] * (MAX_SEQ_LEN - len(tokenized['input_ids']))),
            input_mask=tokenized['attention_mask'] + ([0] * (MAX_SEQ_LEN - len(tokenized['attention_mask']))),
            segment_ids=tokenized['token_type_ids'] + ([0] * (MAX_SEQ_LEN - len(tokenized['token_type_ids']))),
            args_indices=(arg1, arg2, pred),
            label_id=int(example.label)))
    return features


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            pre_lines = f.readlines()
            post_lines = []
            for line in pre_lines:
                post_lines.append(line.strip().split("\t"))
            return post_lines


def find_char_offsets(text, word_offsets):
    splited_text = text.split()
    ret = []
    for arg_str in word_offsets:
        arg = int(arg_str)
        char_offset = sum(len(t) + len(" ") for t in splited_text[:arg])
        assert(text[char_offset: char_offset + len(splited_text[arg])] == splited_text[arg])
        ret.append(char_offset)
    return ret


class AdvclProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")))

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            logger.info(line)
            guid = int(line[0])
            label = int(line[1])
            text = " ".join(clean_tokens(line[3].split()))
            args_char_offset = find_char_offsets(text, line[2].split("-"))
            examples.append(
                InputExample(guid=guid, text=text, args_char_offset=args_char_offset, label=label))
        return examples


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    mat = matthews_corrcoef(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "mcc": mat
    }


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)

