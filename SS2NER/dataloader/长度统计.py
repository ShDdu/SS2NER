#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : 长度统计.py
@Author  : HuYing
@Time    : 2022/11/21 15:56
@Description: 
"""
import os
import json


def main():
    data_list = ["genia"]
    data_path = "../data/raw/"
    for dataset in data_list:

        for file in ["train.json", "dev.json", "test.json"]:
            new_data = {}
            data_length = []
            data_file = os.path.join(data_path + dataset, file)
            with open(data_file, "r", encoding="utf-8") as fp:
                file_lines = json.load(fp)
                for line in file_lines:
                    for ner in line["ner"]:
                        data_length.append(len(ner["index"]))
                        if int(len(ner["index"])) in new_data:
                            new_data[int(len(ner["index"]))] += 1
                        else:
                            new_data[int(len(ner["index"]))] = 1
            data_length.sort()
            print(data_length[int(0.99 * len(data_length))])
            new_data = sorted(new_data.items(), key=lambda d: d[0], reverse=False)
            print(new_data)


if __name__ == '__main__':
    main()
    print('now __name__ is %s' % __name__)
