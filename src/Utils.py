import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import time
import logging
import os
import copy
import pandas as pd
from torch import Tensor
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import zipfile
import json
from io import BytesIO


#######################################################################################################
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def d2s(dt, time=False):
    if time is False:
        return dt.strftime("%Y_%m_%d")
    else:
        return dt.strftime("%Y_%m_%d_%H_%M_%S")


def timer(func):
    def deco(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        logging.info("Function {} run {:.2f}s.".format(func.__name__, end_time - start_time))
        return res

    return deco


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.6, emb_name='word_embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and "video_embeddings.word_embeddings" not in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and "video_embeddings.word_embeddings" not in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        adv_step=1,
        device=None,
        scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.device = device
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, batch, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save() 
        for i in range(self.adv_step):
            self._attack_step()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            ###
            video_input = batch["video_input"].to(self.device)
            video_mask = batch["video_mask"].to(self.device)
            ###
            labels = batch["labels"].to(self.device)
            _, adv_loss = self.model(input_ids=input_ids, attention_mask=attention_mask, video_input=video_input, \
                video_mask=video_mask, labels=labels)
            self.optimizer.zero_grad()
            adv_loss.backward()
            
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class PGD():
    def __init__(self, model, emb_name="word_embeddings.", epsilon=1.0, alpha=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, num_labels=2, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.num_labels = num_labels
        
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.num_labels).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()

# f1_loss = F1_Loss().cuda()

#######################################################################################################
from category_id_map import category_id_to_lv2id, lv2id_to_lv1id


@timer
def prepare_training_data(data_path, tokenizer, fix_length=256):
    with open(data_path, 'r', encoding='utf8') as f:
        anns = json.load(f)
    training_samples = []
    for item in anns:
        vid = item["id"]
        title = item["title"]
        asr = item["asr"]
        ocr = ""
        for o in item["ocr"]:
            ocr += o["text"]
        input_t = tokenizer.encode_plus(
            title,
            add_special_tokens=False,
        ).input_ids
        input_a = tokenizer.encode_plus(
            asr,
            add_special_tokens=False,
        ).input_ids
        input_o = tokenizer.encode_plus(
            ocr,
            add_special_tokens=False,
        ).input_ids
        while len(input_t) + len(input_a) + len(input_o) + 5 >= fix_length:
            if len(input_t) >= len(input_a) and len(input_t) >= len(input_o):
                input_t.pop()
            elif len(input_a) >= len(input_t) and len(input_a) >= len(input_o):
                input_a.pop()
                # del(input_a[0])
            else:
                input_o.pop()
        input_ids = [tokenizer.cls_token_id] + [tokenizer.convert_tokens_to_ids("[T]")] + input_t + [tokenizer.convert_tokens_to_ids("[A]")] \
            + input_a + [tokenizer.convert_tokens_to_ids("[O]")] + input_o + [tokenizer.sep_token_id]
        # logging.info(len(input_ids))
        attention_mask = [1] * len(input_ids)
        labels = category_id_to_lv2id(item["category_id"])
        training_samples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "vid": vid,
            "labels": labels,
            "fold": int(item["fold"]),
        })
        # assert len(input_ids) == fix_length
    return training_samples


def fetch_score(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}
    return eval_results


@timer
def prepare_testing_data(data_path, tokenizer, fix_length=256):
    with open(data_path, 'r', encoding='utf8') as f:
        anns = json.load(f)
    training_samples = []
    for item in anns:
        vid = item["id"]
        title = item["title"]
        asr = item["asr"]
        ocr = ""
        for o in item["ocr"]:
            ocr += o["text"]
        input_t = tokenizer.encode_plus(
            title,
            add_special_tokens=False,
        ).input_ids
        input_a = tokenizer.encode_plus(
            asr,
            add_special_tokens=False,
        ).input_ids
        input_o = tokenizer.encode_plus(
            ocr,
            add_special_tokens=False,
        ).input_ids
        while len(input_t) + len(input_a) + len(input_o) + 5 >= fix_length:
            if len(input_t) >= len(input_a) and len(input_t) >= len(input_o):
                input_t.pop()
            elif len(input_a) >= len(input_t) and len(input_a) >= len(input_o):
                input_a.pop()
                # del(input_a[0])
            else:
                input_o.pop()
        input_ids = [tokenizer.cls_token_id] + [tokenizer.convert_tokens_to_ids("[T]")] + input_t + [tokenizer.convert_tokens_to_ids("[A]")] \
            + input_a + [tokenizer.convert_tokens_to_ids("[O]")] + input_o + [tokenizer.sep_token_id]
        # logging.info(len(input_ids))
        attention_mask = [1] * len(input_ids)
        training_samples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "vid": vid,
            "fold": int(item["fold"]),
        })
        # assert len(input_ids) == fix_length
    return training_samples


@timer
def prepare_predict_data(data_path, tokenizer, fix_length=256):
    with open(data_path, 'r', encoding='utf8') as f:
        anns = json.load(f)
    training_samples = []
    for item in anns:
        vid = item["id"]
        title = item["title"]
        asr = item["asr"]
        ocr = ""
        for o in item["ocr"]:
            ocr += o["text"]
        input_t = tokenizer.encode_plus(
            title,
            add_special_tokens=False,
        ).input_ids
        input_a = tokenizer.encode_plus(
            asr,
            add_special_tokens=False,
        ).input_ids
        input_o = tokenizer.encode_plus(
            ocr,
            add_special_tokens=False,
        ).input_ids
        while len(input_t) + len(input_a) + len(input_o) + 5 >= fix_length:
            if len(input_t) >= len(input_a) and len(input_t) >= len(input_o):
                input_t.pop()
            elif len(input_a) >= len(input_t) and len(input_a) >= len(input_o):
                input_a.pop()
            else:
                input_o.pop()
        input_ids = [tokenizer.cls_token_id] + [tokenizer.convert_tokens_to_ids("[T]")] + input_t + [tokenizer.convert_tokens_to_ids("[A]")] \
            + input_a + [tokenizer.convert_tokens_to_ids("[O]")] + input_o + [tokenizer.sep_token_id]
        # logging.info(len(input_ids))
        attention_mask = [1] * len(input_ids)
        training_samples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "vid": vid,
        })
        # assert len(input_ids) == fix_length
    return training_samples