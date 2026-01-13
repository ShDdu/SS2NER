#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :CNN_Nested.py
# @Time      :2023/5/8 10:39
# @Author    :Ying Hu
# @Desc      :

import json
import torch
import numpy as np
from typing import List
from dataclasses import dataclass
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


def baseline_collate_fn(batch):
    input_ids = []
    attention_mask = []
    loss_mask = []
    word_length = []
    pieces_length = []
    pieces_index = []
    pieces2word = []
    dist_inputs = []
    entity_text = []
    labels = []
    word_ids = []
    char_ids = []
    # pos_ids = []

    for i, example in enumerate(batch):
        input_ids.append(torch.LongTensor(example['input_ids']))
        word_ids.append(torch.LongTensor(example["word_ids"]))
        # pos_ids.append(torch.LongTensor(example["pos_ids"]))
        char_ids.append(example["char_ids"])
        attention_mask.append(torch.LongTensor([1] * len(example['input_ids'])))
        loss_mask.append(torch.LongTensor(example['loss_mask']))
        try:
            example['labels'][:, 0] = i
        except:
            example['labels'] = np.array([[i, 0, 0, 0, 0, 0]])
        labels.append(torch.FloatTensor(example['labels']))
        dist_inputs.append(torch.LongTensor(example['dist_inputs']))
        word_length.extend([example['word_length']])
        pieces2word.append(torch.LongTensor(example['pieces2word']))
        pieces_length.extend([example["pieces_length"]])
        pieces_index.append(torch.LongTensor(example["pieces_index"]))
        entity_text.append(example['entity_text'])

    max_word_length = max(word_length)
    max_pieces_length = max(pieces_length)

    input_ids = pad_sequence(input_ids, batch_first=True)
    word_ids = pad_sequence(word_ids, batch_first=True)
    # pos_ids = pad_sequence(pos_ids, batch_first=True)

    # char_ids = pad_sequence(char_ids, batch_first=True
    char_encoding = []
    char_count = []
    max_char_length = max([max([len(i) for i in char_i]) for char_i in char_ids])
    for char_sent in char_ids:
        sent_count = []
        sent_encoding = []
        for char_token in char_sent:
            sent_count.append(len(char_token))
            if len(char_token) < max_char_length:
                char_token_pad = char_token + [0] * (max_char_length - len(char_token))
            else:
                char_token_pad = char_token[:max_char_length]
            char_token_tensor = torch.LongTensor(char_token_pad)
            sent_encoding.append(char_token_tensor)
        char_encoding.append(torch.stack(sent_encoding))
        char_count.append(torch.LongTensor(sent_count))
    char_encoding = pad_sequence(char_encoding, batch_first=True)
    token_masks_char = (char_encoding != 0)
    char_count = pad_sequence(char_count, batch_first=True)

    attention_mask = pad_sequence(attention_mask, batch_first=True)
    pieces_index = pad_sequence(pieces_index, batch_first=True)
    word_length = torch.LongTensor(word_length)
    pieces_length = torch.LongTensor(pieces_length)

    for i, label in enumerate(labels):
        for j, lab in enumerate(label):
            for k, l in enumerate(lab):
                if k > 1:
                    labels[i][j][k] = l / max_word_length
    labels = torch.cat(labels, 0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((len(batch), max_word_length, max_word_length), dtype=torch.long)
    dist_inputs_matrix = fill(dist_inputs, dis_mat)
    mask2d_mat = torch.zeros((len(batch), max_word_length, max_word_length), dtype=torch.bool)
    loss_mask_matrix = fill(loss_mask, mask2d_mat)
    sub_mat = torch.zeros((len(batch), max_word_length, max_pieces_length), dtype=torch.bool)
    pieces2word_matrix = fill(pieces2word, sub_mat)

    return dict(
        input_ids=input_ids,
        word_ids=word_ids,
        char_ids=char_encoding,
        token_masks_char=token_masks_char,
        char_count=char_count,
        # pos_ids=pos_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask_matrix,
        pieces_index=pieces_index,
        labels=labels,
        dist_inputs=dist_inputs_matrix,
        word_length=word_length,
        pieces_length=pieces_length,
        entity_text=entity_text,
        pieces2word=pieces2word_matrix,
    )


@dataclass
class Entity:
    def __init__(self, start: int, end: int, span: str, label: str, entity_id: str = None):
        self.entity_id = entity_id
        self.start = start
        self.end = end
        self.span = span
        self.label = label


@dataclass
class Example:
    def __init__(self, tokens: List[str], entities: List[Entity], ner_ss: List[List] = None):
        self.tokens = tokens
        self.entities = entities
        self.ner_ss = ner_ss


class BaselineDataset(Dataset):
    def __init__(self, config, logger, set_type="train"):
        self.config = config
        self.logger = logger
        self.tokenizer = config.tokenizer
        self.max_seq_length = config.max_seq_length
        self.set_type = set_type
        self.word2id = None
        self.label2id = {v: k for k, v in enumerate(self.config.label_list, 0)}
        if set_type == "train":
            self.data_path = config.train_file
        elif set_type == "eval":
            self.data_path = config.valid_file
        elif set_type == "test":
            self.data_path = config.test_file
        else:
            print("set_type should be in [train, valid, test]")

        self.examples = self.create_examples(self.data_path)
        if self.config.do_debug:
            self.examples = self.examples[:10]
        else:
            self.count_dataset()
        self.features = self.convert_example_to_features()

    def convert_example_to_features(self):
        features = []
        for example in self.examples:
            if len(example.tokens) == 0:
                continue

            word_ids = []
            char_ids = []
            if self.config.use_word_embedding:
                for w in example.tokens:
                    if w in self.config.word2id:
                        word_ids.append(self.config.word2id[w])
                    else:
                        word_ids.append(self.config.word2id["<UNK>"])

                    if self.config.use_char_embedding:
                        char_id = []
                        for c in w:
                            if c in self.config.char2id:
                                char_id.append(self.config.char2id[c])
                            else:
                                char_id.append(self.config.char2id["<UNK>"])
                        char_id.append(self.config.char2id["<END>"])
                        char_ids.append(char_id)

            encoded_output = self.tokenizer(example.tokens, add_special_tokens=True, padding=False, truncation=True,
                                            max_length=self.max_seq_length, is_split_into_words=True)

            input_ids = encoded_output['input_ids']
            pieces_index = [i + 1 if i is not None else 0 for i in encoded_output.word_ids()]
            word_length = len(example.tokens)
            pieces_length = len(input_ids)

            pieces2word = np.zeros((word_length, len(input_ids)), dtype=np.bool_)
            tokens = [self.tokenizer.tokenize(word) for word in example.tokens]
            if self.tokenizer is not None:
                start = 0
                for i, pieces in enumerate(tokens):
                    if len(pieces) == 0:
                        continue
                    pieces = list(range(start, start + len(pieces)))
                    pieces2word[i, pieces[0] + 1: pieces[-1] + 2] = 1
                    start += len(pieces)

            loss_mask = np.ones((word_length, word_length))
            labels = [[0] + [ex * word_length if i != 0 else ex for i, ex in enumerate(ex_ss)] for ex_ss in example.ner_ss]
            labels = list(filter(None, [lab if lab[-1] < 140 else None for lab in labels]))
            gold_labels = np.around(np.array(labels), 5)

            # 位置Embedding
            dist_inputs = np.zeros((word_length, word_length), dtype=np.int_)
            for k in range(word_length):
                dist_inputs[k, :] += k
                dist_inputs[:, k] -= k

            for i in range(word_length):
                for j in range(word_length):
                    if dist_inputs[i, j] < 0:
                        dist_inputs[i, j] = dis2idx[-dist_inputs[i, j]] + 9
                    else:
                        dist_inputs[i, j] = dis2idx[dist_inputs[i, j]]
            dist_inputs[dist_inputs == 0] = 19

            # 真实的标签字符串 (3, 6, DNA) ---> '3-4-5-#-1'
            def convert_index_to_text(index, type):
                text = "-".join([str(i) for i in index])
                text = text + "-#-{}".format(type)
                return text
            entity_text = set([convert_index_to_text(range(e.start, e.end), self.label2id[e.label]) for e in example.entities])

            features.append(dict(input_ids=input_ids,
                                 word_ids=word_ids,
                                 char_ids=char_ids,
                                 # pos_ids=pos_ids,
                                 word_length=word_length,
                                 pieces_length=pieces_length,
                                 pieces_index=pieces_index,
                                 pieces2word=pieces2word,
                                 labels=gold_labels,
                                 loss_mask=loss_mask,
                                 dist_inputs=dist_inputs,
                                 entity_text=entity_text,
                                 ))

        return features

    @staticmethod
    def create_examples(filename):
        examples = []
        with open(filename, 'r', encoding='utf-8') as f:
            lines = json.load(f)
            for sen_id, line in enumerate(lines):
                entities = []
                sentence = line["sentence"]
                for ent_id, entity in enumerate(line["ner"]):
                    start = entity["index"][0]
                    end = entity["index"][-1] + 1
                    label = entity["type"]
                    if "jnlpba" in filename or "JNLPBA" in filename:
                        label = "GENE"
                        line["ner_ss"] = [[1] + i[1:] for i in line["ner_ss"]]
                    entities.append(Entity(start, end, " ".join(sentence[start: end]), label, str(sen_id) + "-" + str(ent_id)))
                examples.append(Example(sentence, entities, line["ner_ss"]))
        return examples

    def count_dataset(self):
        entity_set = {}
        for example in self.examples:
            for entity in example.entities:
                if entity.label not in entity_set:
                    entity_set[entity.label] = 1
                else:
                    entity_set[entity.label] += 1
        self.logger.info("Data file: {}, Entity count: {}, Total: {}".format(self.set_type, entity_set, sum(entity_set.values())))

    def __getitem__(self, item):

        return self.features[item]

    def __len__(self):
        return len(self.features)


def main():
    print('this message is from main function')


if __name__ == '__main__':
    main()
    print('now __name__ is %s' % __name__)
