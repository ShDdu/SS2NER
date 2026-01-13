#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : baseline.py
@Author  : HuYing
@Time    : 2022/12/27 17:05
@Description: 
"""
from torch.utils.data import Dataset
from dataclasses import dataclass
import json
from typing import List
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")


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
            word_ids = []
            char_ids = []
            pos_ids = []
            if len(example.tokens) == 0:
                continue
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
                        # if len(char_id) < 15:
                        #     char_id = char_id + [0] * (15 - len(char_id))
                        # else:
                        #     char_id = char_id[:15]
                        char_id.append(self.config.char2id["<END>"])
                        char_ids.append(char_id)

            # if self.config.use_pos_embedding:
            #     doc = nlp(" ".join(example.tokens))
            #     t = 0
            #     flag = True
            #     for word in doc:
            #         try:
            #             if str(word) == example.tokens[t]:
            #                 pos_ids.append(self.config.pos2id[word.tag_] if word.tag_ in self.config.pos2id else self.config.pos2id["<UNK>"])
            #                 t += 1
            #                 flag = True
            #             else:
            #                 if flag:
            #                     pos_ids.append(self.config.pos2id[word.tag_] if word.tag_ in self.config.pos2id else self.config.pos2id["<UNK>"])
            #                     flag = False
            #                     t += 1
            #         except:
            #             continue
            #     # pos_ids = pos_ids[:len(example.tokens)]
            #     try:
            #         assert len(pos_ids) == len(example.tokens)
            #     except:
            #         print(" ".join(example.tokens))

            encoded_output = self.tokenizer(example.tokens, add_special_tokens=True, padding=False, truncation=True,
                                            max_length=self.max_seq_length, is_split_into_words=True)

            input_ids = encoded_output['input_ids']
            pieces_index = [i + 1 if i is not None else 0 for i in encoded_output.word_ids()]
            word_length = len(example.tokens)
            pieces_length = len(input_ids)

            pieces2word = np.zeros((word_length, len(input_ids)), dtype=np.bool)
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
            labels = list(filter(None, [lab if lab[-1] < 16 else None for lab in labels]))
            gold_labels = np.around(np.array(labels), 5)

            # 位置Embedding
            dist_inputs = np.zeros((word_length, word_length), dtype=np.int)
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
                    entities.append(Entity(start, end, " ".join(sentence[start: end]), entity["type"], str(sen_id) + "-" + str(ent_id)))
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
