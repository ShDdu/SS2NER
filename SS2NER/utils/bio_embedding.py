#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : 筛选Bio词向量.py
@Author  : HuYing
@Time    : 2023/3/6 21:19
@Description: 
"""
import os
import json


def main():
    data_list = ["genia"]
    data_path = "../data/"
    for dataset in data_list:
        words = []
        for file in ["train.json", "dev.json", "test.json"]:
            data_file = os.path.join(data_path + dataset, file)
            with open(data_file, "r", encoding="utf-8") as fp:
                file_lines = json.load(fp)
                for line in file_lines:
                    for word in line["sentence"]:
                        if word not in words:
                            words.append(word)
        words_embedding = []
        with open("../output/bio/PubMed-shuffle-win-30.txt", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(' ')
                vector = list(map(float, line[1:]))
                if line[0] in words:
                    words_embedding.append([line[0], vector])
        with open("../output/bio/bio2embedding-200.txt", 'w', encoding='utf-8') as fw:
            for i in words_embedding:
                fw.write(" ".join(i))
                fw.write("\n")

    print('this message is from main function')


if __name__ == '__main__':
    main()
    print('now __name__ is %s' % __name__)
