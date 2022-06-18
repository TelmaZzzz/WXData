import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import Model, Utils, Datasets
from transformers import AdamW, get_cosine_schedule_with_warmup, AutoConfig
import logging
import gc
import torch.cuda.amp as AMP
from apex import amp
from tqdm import tqdm
import numpy as np


class TrainerConfig(object):
    def __init__(self, args):
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epoch = args.epoch
        self.opt_step = args.opt_step
        self.eval_step = args.eval_step
        self.Tmax = args.Tmax
        self.min_lr = args.min_lr
        self.scheduler = args.scheduler
        self.max_norm = args.max_norm
        self.model_save = args.model_save
        self.model_load = args.model_load
        self.debug = args.debug
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.fp16 = args.fp16
        self.fgm = args.fgm
        self.fgm_up = args.fgm_up
        self.fix_length = args.fix_length
        self.ema = args.ema
        self.awp = args.awp
        self.awp_up = args.awp_up
        self.awp_lr = args.awp_lr
        self.awp_eps = args.awp_eps
        self.pgd = args.pgd
        self.pgd_k = args.pgd_k
        self.swa = args.swa
        self.swa_start_step = args.swa_start_step
        self.swa_update_step = args.swa_update_step
        self.swa_lr = args.swa_lr
        self.warmup_step = args.warmup_step
        self.warmup_rate = args.warmup_rate
        self.patience_maxn = args.patience_maxn
        self.freeze_nums = args.freeze_nums


class BaseTrainer(object):
    def __init__(self, args):
        self.predict_loss = 0
        self.trainer_config = TrainerConfig(args)
        self.model_config = Model.ModelConfig(args)
        self.device = args.device

    def trainer_init(self, training_size, valid_datasets, valid_collate=None, sz=0):
        self.model_init()
        if sz > 0:
            self.resize_token_embeddings(sz)
        self.set_training_size(training_size)
        self.optimizer_init()
        self.set_valid_datasets(valid_datasets, valid_collate)
        if self.trainer_config.model_load is not None:
            self.model_load(self.trainer_config.model_load)

    def build_model(self):
        self.model = None

    def model_init(self):
        self.build_model()
        self.model.to(self.device)
        self.model.train()

    def optimizer_init(self):
        optimizer_grouped_parameters = self._get_optimizer_grouped_parameters()
        self.optimizer = AdamW(optimizer_grouped_parameters)
        training_epoch = self.training_size // self.trainer_config.epoch
        scheduler_map = {
            "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.trainer_config.Tmax, eta_min=self.trainer_config.min_lr),
            "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=self.trainer_config.Tmax, T_mult=1, eta_min=self.trainer_config.min_lr),
            "get_cosine_schedule_with_warmup": get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=int(self.trainer_config.warmup_rate * self.training_size) if self.trainer_config.warmup_step == -1 else self.trainer_config.warmup_step, 
                num_training_steps=self.training_size,
                num_cycles=1,
                last_epoch=-1,
            ),
            "MultiStepLR": lr_scheduler.MultiStepLR(self.optimizer, [training_epoch * 2, training_epoch * 6], gamma=0.1),
        }
        if self.trainer_config.fp16:
            self.model, self.optmizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        if self.trainer_config.fgm:
            self.fgm = Utils.FGM(self.model)
        if self.trainer_config.ema:
            self.ema = Utils.EMA(self.model, 0.999)
            self.ema.register()
        if self.trainer_config.awp:
            self.awp = Utils.AWP(self.model, self.optimizer, adv_lr=self.trainer_config.awp_lr, device=self.device, \
                adv_eps=self.trainer_config.awp_eps, start_epoch=self.training_size // self.trainer_config.epoch)
        if self.trainer_config.pgd:
            self.pgd = Utils.PGD(self.model)
        if self.trainer_config.swa:
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
            self.swa_model.eval()
            self.swa_scheduler = torch.optim.swa_utils.SWALR(self.optimizer, swa_lr=self.trainer_config.swa_lr)
            self.swa_flag = False
        self.scheduler = scheduler_map[self.trainer_config.scheduler]
        self.score_maxn = 0
        self.num_step = 0
        self.patience = 0

    def _get_optimizer_grouped_parameters(self):
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n in no_decay
                ],
                "weight_decay": 0,
                "lr": self.trainer_config.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in no_decay
                ],
                "weight_decay": self.trainer_config.weight_decay,
                "lr": self.trainer_config.lr,
            },
        ]
        return optimizer_grouped_parameters

    def get_logits(self, batch, return_loss=False, use_swa=False):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        video_input = batch["video_input"].to(self.device)
        video_mask = batch["video_mask"].to(self.device)
        if return_loss:
            labels = batch["labels"].to(self.device)
            logits, loss = self.model(input_ids, attention_mask, video_input, video_mask, labels=labels)
            return logits, loss
        else:
            if use_swa:
                logits, _ = self.swa_model(input_ids, attention_mask, video_input, video_mask)
            else:
                logits, _ = self.model(input_ids, attention_mask, video_input, video_mask)
            return logits
    
    def get_loss(self, batch):
        _, loss = self.get_logits(batch, return_loss=True)
        return loss
    
    def step(self, batch):
        if self.patience > self.trainer_config.patience_maxn:
            return 0
        loss = self.get_loss(batch)
        loss /= self.trainer_config.opt_step
        self.num_step += 1
        if self.trainer_config.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if self.trainer_config.fgm and self.score_maxn >= self.trainer_config.fgm_up:
            self.fgm.attack()
            loss_fgm = self.get_loss(batch)
            if self.trainer_config.fp16:
                with amp.scale_loss(loss_fgm, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_fgm.backward()
            self.fgm.restore()
        if self.trainer_config.pgd:
            self.pgd.backup_grad()
            for t in range(self.trainer_config.pgd_k):
                self.pgd.attack(is_first_attack=(t==0))
                if t != self.trainer_config.pgd_k-1:
                    self.model.zero_grad()
                else:
                    self.pgd.restore_grad()
                with AMP.autocast(enabled=self.trainer_config.fp16):
                    loss_pgd = self.get_loss(batch)
                if self.trainer_config.fp16:
                    with amp.scale_loss(loss_pgd, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_pgd.backward()
            self.pgd.restore()
        if self.trainer_config.awp and self.score_maxn > self.trainer_config.awp_up:
            self.awp.attack_backward(batch, self.num_step)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_config.max_norm)
        if self.num_step % self.trainer_config.opt_step == 0:
            self.optimizer.step()
            if self.trainer_config.ema:
                self.ema.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            if self.num_step % self.trainer_config.eval_step == 0:
                self.eval()
        if self.trainer_config.swa and self.num_step >= self.trainer_config.swa_start_step:
            self.swa_flag = True
            self.scheduler = self.swa_scheduler
        if self.trainer_config.swa and self.swa_flag and self.num_step % self.trainer_config.swa_update_step == 0:
            self.swa_model.update_parameters(self.model)
        return loss.cpu()

    @torch.no_grad()
    def eval(self, valid_datasets=None, valid_collate=None):
        if self.trainer_config.ema:
            self.ema.apply_shadow()
        if self.trainer_config.swa and self.swa_flag:
            self.swa_model.eval()
        else:
            self.model.eval()
        if valid_datasets is None:
            valid_datasets = self.valid_datasets
        if valid_collate is None:
            valid_collate = self.valid_collate
        if valid_collate is None:
            valid_iter = torch.utils.data.DataLoader(valid_datasets, batch_size=self.trainer_config.valid_batch_size)
        else:
            valid_iter = torch.utils.data.DataLoader(valid_datasets, batch_size=self.trainer_config.valid_batch_size, collate_fn=valid_collate)
        preds = self.eval_step(valid_iter)
        score = self.metrics(preds.numpy(), valid_datasets.samples)
        logging.info("Valid Score: {:.6f}".format(score))
        if self.score_maxn < score:
            self.score_maxn = score
            self.save()
            self.patience = 0
        else:
            self.patience += 1
        del valid_iter
        gc.collect()
        self.model.train()
        if self.trainer_config.ema:
            self.ema.restore()
    
    @Utils.timer
    @torch.no_grad()
    def eval_step(self, valid_iter):
        preds = []
        for batch in valid_iter:
            logits = self.get_logits(batch).cpu()
            preds.append(logits)
        preds = torch.cat(preds, dim=0)
        return preds

    def metrics(self, preds, valid_samples):
        golds = [item["labels"] for item in valid_samples]
        result = Utils.fetch_score(np.argmax(preds, axis=1), golds)
        logging.info(result)
        return result["mean_f1"]

    def save(self):
        if self.trainer_config.debug:
            return
        if self.trainer_config.swa and self.swa_flag:
            torch.save(self.swa_model.state_dict(), self.trainer_config.model_save)
        else:
            torch.save(self.model.state_dict(), self.trainer_config.model_save)

    def model_load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.train()

    @torch.no_grad()
    def predict(self, test_iter):
        self.model.eval()
        logits_list = []
        for batch in tqdm(test_iter):
            logits_list.append(self.get_logits(batch).cpu())
        logits = torch.cat(logits_list, dim=0)
        return logits.numpy()

    def set_training_size(self, sz):
        self.training_size = self.trainer_config.epoch * sz // self.trainer_config.opt_step
    
    def set_valid_datasets(self, valid_datasets, valid_collate=None):
        self.valid_datasets = valid_datasets
        self.valid_collate = valid_collate

    def resize_token_embeddings(self, sz):
        self.model.resize_token_embeddings(sz)


class WXTrainer(BaseTrainer):
    def __init__(self, args):
        super(WXTrainer, self).__init__(args)
    
    def build_model(self):
        self.model = Model.WXDataModel(self.model_config)


class PretrainTrainer(BaseTrainer):
    def __init__(self, args):
        super(PretrainTrainer, self).__init__(args)
    
    def build_model(self):
        config = AutoConfig.from_pretrained(self.model_config.pretrain_path)
        # from Model import WXDataPretrainModel
        self.model = Model.WXDataPretrainModel.from_pretrained(self.model_config.pretrain_path, config=config)
        # self.model = WXDataPretrainModel.from_pretrained(self.model_config.pretrain_path, config=config)

    def trainer_init(self, training_size, valid_datasets, valid_collate=None, sz=0):
        super().trainer_init(training_size, valid_datasets, valid_collate, sz)
        self.score_maxn = 100.0

    def get_logits(self, batch, return_loss=False, use_swa=False):
        input_ids = batch["input_ids"].to(self.device)
        input_ids_r = batch["input_ids_r"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        video_input = batch["video_input"].to(self.device)
        video_mask = batch["video_mask"].to(self.device)
        if return_loss:
            labels = batch["labels"].to(self.device)
            logits, loss = self.model(input_ids, attention_mask, video_input, video_mask, input_ids_r, labels=labels)
            return logits, loss
        else:
            if use_swa:
                logits, _ = self.swa_model(input_ids, attention_mask, video_input, video_mask, input_ids_r)
            else:
                logits, _ = self.model(input_ids, attention_mask, video_input, video_mask, input_ids_r)
            return logits

    @Utils.timer
    @torch.no_grad()
    def eval_step(self, valid_iter):
        loss = 0
        for batch in valid_iter:
            loss += self.get_loss(batch).cpu() 
        loss /= len(valid_iter)
        return loss
    
    @torch.no_grad()
    def eval(self, valid_datasets=None, valid_collate=None):
        self.model.eval()
        if valid_datasets is None:
            valid_datasets = self.valid_datasets
        if valid_collate is None:
            valid_collate = self.valid_collate
        if valid_collate is None:
            valid_iter = torch.utils.data.DataLoader(valid_datasets, batch_size=self.trainer_config.valid_batch_size)
        else:
            valid_iter = torch.utils.data.DataLoader(valid_datasets, batch_size=self.trainer_config.valid_batch_size, collate_fn=valid_collate)
        loss = self.eval_step(valid_iter)
        logging.info("Valid loss: {:.6f}".format(loss))
        if self.score_maxn > loss:
            self.score_maxn = loss
            self.save()
            self.patience = 0
        else:
            self.patience += 1
        del valid_iter
        gc.collect()
        self.model.train()
    
    def save(self):
        self.model.save_pretrained(self.trainer_config.model_save)


class WXPredicter(WXTrainer):
    def __init__(self, args):
        super(WXPredicter, self).__init__(args)
    
    def trainer_init(self, valid_datasets, valid_collate=None, sz=0):
        self.model_init()
        if sz > 0:
            self.resize_token_embeddings(sz)
        self.set_valid_datasets(valid_datasets, valid_collate)
        if self.trainer_config.model_load is not None:
            self.model_load(self.trainer_config.model_load)


def check(name, ls):
    for l in ls:
        if l in name:
            return True
    return False

class WXTrainerV2(BaseTrainer):
    def __init__(self, args):
        super(WXTrainerV2, self).__init__(args)
    
    def build_model(self):
        self.model = Model.WXDataModelV2(self.model_config)
    
    def trainer_init(self, training_size, valid_datasets, valid_collate=None, sz=0):
        super().trainer_init(training_size, valid_datasets, valid_collate, sz)
        # self.freeze()
        # for n, p in self.model.named_parameters():
        #     logging.info(f"{n} : {p.requires_grad}")

    def freeze(self):
        freeze_list = [f"layer.{i}." for i in range(24)]
        if self.trainer_config.freeze_nums == 0:
            freeze_list = []
        else:
            freeze_list = freeze_list[:min(len(freeze_list), self.trainer_config.freeze_nums)]
        logging.info(freeze_list)
        for n, p in self.model.named_parameters():
            if check(n, freeze_list):
                p.requires_grad = False
            else:
                p.requires_grad = True


class WXPredicterV2(WXTrainerV2):
    def __init__(self, args):
        super(WXPredicterV2, self).__init__(args)
    
    def trainer_init(self, valid_datasets, valid_collate=None, sz=0):
        self.model_init()
        if sz > 0:
            self.resize_token_embeddings(sz)
        self.set_valid_datasets(valid_datasets, valid_collate)
        if self.trainer_config.model_load is not None:
            self.model_load(self.trainer_config.model_load)


class WXTrainerV3(BaseTrainer):
    def __init__(self, args):
        super(WXTrainerV3, self).__init__(args)
    
    def build_model(self):
        self.model = Model.WXDataModelV3(self.model_config)


class WXPredicterV3(WXTrainerV3):
    def __init__(self, args):
        super(WXPredicterV3, self).__init__(args)
    
    def trainer_init(self, valid_datasets, valid_collate=None, sz=0):
        self.model_init()
        if sz > 0:
            self.resize_token_embeddings(sz)
        self.set_valid_datasets(valid_datasets, valid_collate)
        if self.trainer_config.model_load is not None:
            self.model_load(self.trainer_config.model_load)