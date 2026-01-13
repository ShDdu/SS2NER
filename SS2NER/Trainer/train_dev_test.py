#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : train_dev_test.py
@Author  : HuYing
@Time    : 2022/12/29 9:50
@Description: 
"""
import time
import os
from tqdm import tqdm
import prettytable as pt
from utils.baseline import *
from transformers import get_scheduler
from torch.utils.data import DataLoader
from dataloader.W2NER import baseline_collate_fn


class Trainer(object):
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.lr_scheduler = None
        self.collate_fn = baseline_collate_fn
        self.adv_fn = None
        self.optimizer = None

    def create_optimizer(self, updates_total):
        bert_params = set(self.model.embedding.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in self.model.embedding.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.config.bert_learning_rate,
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.embedding.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': self.config.learning_rate,
             'weight_decay': self.config.weight_decay},
        ]
        self.optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.lr_scheduler = get_scheduler(name=self.config.lr_scheduler_type, optimizer=self.optimizer,
                                          num_warmup_steps=self.config.warmup_rate * updates_total, num_training_steps=updates_total)

    def create_dataloader(self, dataset, training=False, num_workers=0):
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.config.train_batch_size if training else self.config.eval_batch_size,
                                shuffle=True if training else False,  #
                                pin_memory=True,
                                num_workers=num_workers,
                                collate_fn=self.collate_fn)
        return dataloader

    def load_model(self, model_path, device="cpu"):
        if os.path.isfile(model_path):
            print(model_path)
            state_dict = torch.load(model_path, map_location="cpu")
            load_result = self.model.load_state_dict(state_dict, strict=False)
            if len(load_result.missing_keys) != 0:
                if hasattr(self.model, "_keys_to_ignore_on_save") and self.model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(self.model._keys_to_ignore_on_save):
                    self.model.tie_weights()
                else:
                    self.config.logger.warning(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
            if len(load_result.unexpected_keys) != 0:
                self.config.logger.warning(f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.")
            del state_dict

    def save_model(self, output_dir=None, save_best_last="best_model.bin"):
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.config.logger.info(f'Saving model to {output_dir}...')

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        # if not self.is_in_train:
        #     if self.opt.save_to_fp16:
        #         model_to_save = model_to_save.to(dtype=torch.float16)
        state_dict = model_to_save.state_dict()

        if hasattr(model_to_save, 'base_model_prefix') and model_to_save.base_model_prefix == 'nezha':
            for name in list(state_dict.keys()):
                if "relative_positions_encoding" in name:
                    del state_dict[name]
        torch.save(state_dict, os.path.join(output_dir, save_best_last))
        if hasattr(model_to_save, "config"):
            model_to_save.config.save_pretrained(output_dir)
        if self.config.tokenizer is not None:
            self.config.tokenizer.save_pretrained(output_dir)
        return output_dir

    def set_device(self, model):
        if model is not None:
            model.to(self.config.device)

    @staticmethod
    def prepare_inputs(dic, device):
        for k, v in dic.items():
            if isinstance(v, torch.Tensor):
                dic[k] = v.to(device)
            else:
                dic[k] = v
        return dic

    def train(self, train_dataset=None, dev_dataset=None, test_dataset=None):
        train_dataloader = self.create_dataloader(train_dataset, training=True)
        num_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        t_total = num_steps_per_epoch * self.config.epochs
        self.create_optimizer(t_total)
        self.model.to(self.config.device)

        # Train!
        self.config.logger.info("***** Running training on {} *****".format(self.config.task_name))
        self.config.logger.info("  Num examples = %d", len(train_dataset))
        self.config.logger.info("  Num Epochs = %d", self.config.epochs)
        self.config.logger.info("  Gradient Accumulation steps = %d", self.config.gradient_accumulation_steps)
        self.config.logger.info("  Total optimization steps = %d", t_total)

        train_iter = tqdm(total=t_total, desc='training')
        best_f1_score, global_step, total_loss = -1, 0, 0

        self.model.train()
        for trained_epoch in range(self.config.epochs):
            for step, train_batch in enumerate(train_dataloader):
                data = self.prepare_inputs(train_batch, self.config.device)
                inputs = {
                    "input_ids": data["input_ids"],
                    "word_ids": data["word_ids"],
                    "char_ids": data["char_ids"],
                    "token_masks_char": data["token_masks_char"],
                    "char_count": data["char_count"],
                    # "pos_ids": data["pos_ids"],
                    "pieces_index": data["pieces_index"],
                    # "attention_mask": data["attention_mask"],
                    "loss_mask": data["loss_mask"],
                    "labels": data["labels"],
                    "dist_inputs": data["dist_inputs"],
                    "word_length": data["word_length"],
                    "pieces_length": data["pieces_length"],
                    # "entity_text": data["entity_text"],
                    "pieces2word": data["pieces2word"],
                }
                outputs = self.model(**inputs)
                loss, output = outputs["loss"], outputs["output"]

                total_loss += float(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                train_iter.set_postfix(epoch=trained_epoch, loss=loss.item())
                train_iter.update(1)
                global_step += 1
                # if global_step % 100 == 0:
                    # self.save_model(os.path.join(self.config.output_dir, str(self.config.task_name)), save_best_last=str(global_step) + "_model.bin")
                    # self.config.logger.info(
                        # " Save Model in %s" % str(os.path.join(self.config.output_dir, self.config.task_name + str(global_step) + "_model.bin")))

            if dev_dataset is not None:
                self.config.logger.info("***** Running Evaluation on {} *****".format(self.config.task_name))
                self.config.logger.info("  Current Epoch = %d" % trained_epoch)
                self.config.logger.info("  Current Step = %d" % global_step)

                ner_p, ner_r, ner_f = self.evaluate(dev_dataset, epoch=trained_epoch)
                self.config.logger.info(" dev f1 score = %s  " % ner_f)
                dev_f1_score = ner_f
                if dev_f1_score > best_f1_score:
                    best_f1_score = dev_f1_score
                    should_save = True
                else:
                    should_save = False
                if should_save:
                    # best_f1_score = best_eval_f
                    self.save_model(os.path.join(self.config.output_dir, str(self.config.task_name)))
                    self.config.logger.info(
                        " Save Model in %s" % str(os.path.join(self.config.output_dir, self.config.task_name + "/best_model.bin")))
                self.config.logger.info(" Best F1 on Evaluation Data: %.5f  " % best_f1_score)

        train_iter.close()

    def evaluate(self, dataset, epoch=0):
        self.config.logger.info('Evaluating %s dataset...' % self.config.task_name)
        data_loader = self.create_dataloader(dataset)
        total_len = len(data_loader)
        self.model.eval()
        eval_iter = tqdm(total=total_len, desc='Evaluating')

        output_result, labels, length = [], [], []
        eval_loss = 0.0
        total_ent_r, total_ent_p, total_ent_c = 0, 0, 0
        for step, data in enumerate(data_loader):
            data = self.prepare_inputs(data, self.config.device)
            # data["mode"] = "Test"
            targets = data["labels"].clone()
            labels += targets[:, 1].tolist()
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= data["loss_mask"].shape[1]  # sentence length

            with torch.no_grad():
                inputs = {
                    "input_ids": data["input_ids"],
                    "word_ids": data["word_ids"],
                    "char_ids": data["char_ids"],
                    "token_masks_char": data["token_masks_char"],
                    "char_count": data["char_count"],
                    # "pos_ids": data["pos_ids"],
                    "pieces_index": data["pieces_index"],
                    # "attention_mask": data["attention_mask"],
                    "loss_mask": data["loss_mask"],
                    "labels": data["labels"],
                    "dist_inputs": data["dist_inputs"],
                    "word_length": data["word_length"],
                    "pieces_length": data["pieces_length"],
                    # "entity_text": data["entity_text"],
                    "pieces2word": data["pieces2word"],
                }
                outputs = self.model(**inputs)
                loss, output = outputs["loss"], outputs["output"]
                eval_loss += float(loss)
            output = non_max_suppression(output, conf_thres=self.config.conf_thres, nms_thres=self.config.nms_thres)
            ent_c, ent_p, ent_r, _ = decode(output, targets, data["entity_text"], data["word_length"])
            total_ent_r += ent_r  # 数据集中实体总数
            total_ent_p += ent_p  # 预测为实体但预测错误的个数
            total_ent_c += ent_c  # 预测正确的个数
            eval_iter.update(1)
        eval_iter.close()
        p, r, f = call_f1(total_ent_c, total_ent_p, total_ent_r)
        table_train = pt.PrettyTable(["Test {}({})/{}".format(self.config.task_name, epoch, self.config.epochs), "Loss", "Precision", "Recall", "F1"])
        table_train.add_row(["Performance", "{:.4f}".format(eval_loss / len(data_loader))] + ["{:3.4f}".format(x) for x in [p, r, f]])
        self.config.logger.info("\n{}".format(table_train))
        self.model.train()

        return p, r, f

    def evaluate_with_search(self, dataset, epoch=0):
        self.config.logger.info('Testing %s dataset with search...' % self.config.task_name)
        data_loader = self.create_dataloader(dataset)
        total_len = len(data_loader)
        self.model.eval()
        eval_iter = tqdm(total=total_len, desc='Evaluating')

        eval_loss = 0.0
        outputs_all, labels_all, length_all = [], [], []
        for step, data in enumerate(data_loader):
            data = self.prepare_inputs(data, self.config.device)
            with torch.no_grad():
                inputs = {
                    "input_ids": data["input_ids"],
                    "word_ids": data["word_ids"],
                    "char_ids": data["char_ids"],
                    "token_masks_char": data["token_masks_char"],
                    "char_count": data["char_count"],
                    # "pos_ids": data["pos_ids"],
                    "pieces_index": data["pieces_index"],
                    # "attention_mask": data["attention_mask"],
                    "loss_mask": data["loss_mask"],
                    "labels": data["labels"],
                    "dist_inputs": data["dist_inputs"],
                    "word_length": data["word_length"],
                    # "pieces_length": data["pieces_length"],
                    # "entity_text": data["entity_text"],
                    "pieces2word": data["pieces2word"],
                }
                outputs = self.model(**inputs)
                loss, output = outputs["loss"], outputs["output"]
                output = non_max_suppression(output, conf_thres=self.config.conf_thres, nms_thres=self.config.nms_thres)
            outputs_all.extend(output)
            labels_all.extend(data["entity_text"])
            length_all.append(data["word_length"])
            eval_iter.update(1)
        eval_iter.close()

        length_all = torch.cat(length_all, dim=-1)
        best_p, best_r, best_f = 0, 0, 0
        # if self.config.show_result:
            # show_results(outputs_all, labels_all, length_all, self.config)
        # else:
        best_p, best_r, best_f = decode_search(outputs_all, labels_all, length_all)

        table_eval = pt.PrettyTable(["Test {}({})/{}".format(self.config.task_name, epoch, self.config.epochs), "Loss", "Precision", "Recall", "F1"])
        table_eval.add_row(["Performance", "{:.4f}".format(eval_loss / len(data_loader))] + ["{:3.4f}".format(x) for x in [best_p, best_r, best_f]])
        self.config.logger.info("\n{}".format(table_eval))
        self.model.train()
        return best_p, best_r, best_f

    def test(self, test_dataset, best_path_dir):
        self.config.logger.info("Begin Testing from {}".format(os.path.join(self.config.output_dir, best_path_dir)))
        best_path_dir = os.path.join(self.config.output_dir, best_path_dir)
        self.load_model(best_path_dir)
        # self.model.load_state_dict(torch.load(path))  # , map_location="cpu"
        # parameters = torch.load(best_path_dir, map_location="cpu")
        # self.model.load_state_dict(parameters["model"])
        self.set_device(self.model)
        # self.model.eval()

        # ner_p, ner_r, ner_f = self.evaluate(test_dataset)
        best_eval_p, best_eval_r, best_eval_f = self.evaluate_with_search(test_dataset)
        # self.config.show_result = False
        # self.config.logger.info("ner_p: %.5f, ner_r: %.5f, ner_f: %.5f" % (ner_p, ner_r, ner_f))
        self.config.logger.info("ner_search_p: {}, ner_search_r: {}, ner_search_f: {}".format(best_eval_p, best_eval_r, best_eval_f))
        return best_eval_p, best_eval_r, best_eval_f


def main():
    print('this message is from main function')


if __name__ == '__main__':
    main()
    print('now __name__ is %s' % __name__)
