import os
import logging
import datetime
import torch
import Config, Datasets, Model, Trainer, Utils
from transformers import AutoTokenizer, BertTokenizerFast
import pandas as pd
import time
import gc
from tqdm import tqdm
import copy
import numpy as np
import json
from category_id_map import CATEGORY_ID_LIST, lv2id_to_category_id
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


@Utils.timer
def WXBase(args):
    args.num_labels = len(CATEGORY_ID_LIST)
    logging.debug(args.tokenizer_path)
    if "nezha" in args.tokenizer_path:
        tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path, trim_offsets=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trim_offsets=False)
    special_token = {"additional_special_tokens": ["[T]", "[A]", "[O]"]}
    tokenizer.add_special_tokens(special_token)
    samples = Utils.prepare_training_data(args.train_path, tokenizer, args.fix_length)
    train_samples, valid_samples = [], []
    for item in samples:
        if item["fold"] != args.fold:
            train_samples.append(item)
        else:
            valid_samples.append(item)
    logging.info(f"Train size: {len(train_samples)}")
    logging.info(f"Valid size: {len(valid_samples)}")
    if args.debug:
        train_samples = train_samples[:100]
        valid_samples = train_samples
    train_datasets = Datasets.BaseDataset(train_samples, args.zip_path, tokenizer, args.mask_prob, args.mask_ratio, fix_length=args.fix_length)
    valid_datasets = Datasets.BaseDataset(valid_samples, args.zip_path, tokenizer, is_test=True, fix_length=args.fix_length)
    train_iter = torch.utils.data.DataLoader(train_datasets, batch_size=args.train_batch_size, shuffle=True)
    if args.model_name == "v1":
        trainer = Trainer.WXTrainer(args)
    elif args.model_name == "v2":
        trainer = Trainer.WXTrainerV2(args)
    elif args.model_name == "v3":
        trainer = Trainer.WXTrainerV3(args)
    elif args.model_name == "db_v1":
        trainer = Trainer.DoubleWXTrainer(args)
    trainer.trainer_init(len(train_iter), valid_datasets, sz=len(tokenizer))
    logging.info(f"Train Size: {len(train_iter)}")
    for epoch in range(args.epoch):
        logging.info(f"Start Epoch: {epoch}")
        t_s = time.time()
        loss = 0
        if args.debug:
            for batch in tqdm(train_iter):
                loss += trainer.step(batch)
        else:
            for batch in train_iter:
                loss += trainer.step(batch)
        logging.info("Train Loss: {:.4f}".format(loss / len(train_iter)))
        t_e = time.time()
        logging.info("Cost {:.2f} s.".format(t_e - t_s))
    score_maxn = trainer.score_maxn
    logging.info("Best Score: {:.6f}".format(score_maxn))
    del train_samples, valid_samples
    del train_datasets, valid_datasets
    del train_iter, trainer
    gc.collect()
    return score_maxn


@Utils.timer
def main(args):
    if args.mode not in ["base"]:
        raise
    model_save = "/".join([args.model_save, Utils.d2s(datetime.datetime.now(), time=True)])
    if not args.debug:
        if os.path.exists(model_save):
            logging.warning("save path exists, sleep 60s")
            raise
        else:
            os.mkdir(model_save)
            args.model_save = model_save
    MODEL_PREFIX = args.model_save
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f"device: {args.device}")
    if args.train_all:
        num = args.fold
        for fold in range(num):
            args.fold = fold
            args.model_save = "/".join([MODEL_PREFIX, f"Fold_{args.fold}.bin"])
            logging.info(f"model save path: {args.model_save}")
            if args.mode == "base":
                WXBase(args)
    else:
        args.model_save = "/".join([MODEL_PREFIX, f"Fold_{args.fold}.bin"])
        logging.info(f"model save path: {args.model_save}")
        if args.mode == "base":
            WXBase(args)


@Utils.timer
def predict(args):
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.num_labels = len(CATEGORY_ID_LIST)
    model_list = [
        # {
        #     "model_path": [
        #         ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_07_20_03_25/Fold_0.bin", 1),
        #         ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_07_22_31_45/Fold_1.bin", 1),
        #         ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_07_22_29_14/Fold_2.bin", 1),
        #         ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_07_22_29_40/Fold_3.bin", 1),
        #         ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_07_22_30_02/Fold_4.bin", 1),
        #     ],
        #     "pretrain_path": "/users10/lyzhang/opt/tiger/WXData/model/Pretrain/2022_06_06_21_02_38",
        #     "version": "v1",
        # },
        # {
        #     "model_path": [
        #         ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_08_22_23_04/Fold_0.bin", 1),
        #         ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_09_19_05_52/Fold_1.bin", 1),
        #         ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_09_19_06_20/Fold_2.bin", 1),
        #         ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_09_19_06_51/Fold_3.bin", 1),
        #         ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_09_19_07_20/Fold_4.bin", 1),
        #     ],
        #     "pretrain_path": "/users10/lyzhang/opt/tiger/WXData/model/Pretrain/2022_06_07_21_50_43",
        #     "version": "v1",
        # },

        {
            "model_path": [
                ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_13_19_26_33/Fold_0.bin", 1),
                # ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_13_20_43_30/Fold_1.bin", 1),
                # ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_13_20_43_52/Fold_2.bin", 1),
                # ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_13_20_44_21/Fold_3.bin", 1),
                # ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_13_20_41_58/Fold_4.bin", 1),
            ],
            "pretrain_path": "/users10/lyzhang/opt/tiger/WXData/model/Pretrain/2022_06_06_21_02_38",
            "version": "v2",
        },
        # {
        #     "model_path": [
        #         ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_13_20_44_41/Fold_5.bin", 1),
        #         ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_13_20_44_56/Fold_6.bin", 1),
        #         ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_13_23_11_59/Fold_7.bin", 1),
        #         ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_13_23_11_45/Fold_8.bin", 1),
        #         ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_13_23_11_31/Fold_9.bin", 1),
        #     ],
        #     "pretrain_path": "/users10/lyzhang/opt/tiger/WXData/model/Pretrain/2022_06_07_21_50_43",
        #     "version": "v2",
        # },
        {
            "model_path": [
                ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_15_18_43_58/Fold_0.bin", 1),
                # ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_15_20_04_19/Fold_1.bin", 2),
                # ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_15_20_10_47/Fold_2.bin", 2),
                # ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_15_20_11_16/Fold_3.bin", 2),
                # ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_15_20_19_33/Fold_4.bin", 2),
            ],
            "pretrain_path": "/users10/lyzhang/opt/tiger/WXData/model/Pretrain/2022_06_14_20_25_33",
            "version": "v3",
        },
        {
            "model_path": [
                ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_19_14_13_25/Fold_0.bin", 1),
                # ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_15_20_04_19/Fold_1.bin", 2),
                # ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_15_20_10_47/Fold_2.bin", 2),
                # ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_15_20_11_16/Fold_3.bin", 2),
                # ("/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_15_20_19_33/Fold_4.bin", 2),
            ],
            "pretrain_path": "/users10/lyzhang/opt/tiger/WXData/model/Pretrain/2022_06_18_23_00_36",
            "version": "db_v1",
        },
    ]
    SUM = 0
    for model in model_list:
        for m in model["model_path"]:
            SUM += m[1]
    result = None
    for model_config in model_list:
        args.pretrain_path = model_config["pretrain_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_config["pretrain_path"], trim_offsets=False)
        samples = Utils.prepare_training_data(args.train_path, tokenizer, args.fix_length)
        # samples = Utils.prepare_predict_data(args.train_path, tokenizer, args.fix_length)

        train_samples, valid_samples = [], []
        for item in samples:
            if item["fold"] != args.fold:
                train_samples.append(item)
            else:
                valid_samples.append(item)
        samples = valid_samples

        valid_datasets = Datasets.BaseDatasetsValid(samples, args.zip_path, tokenizer, is_test=True, fix_length=args.fix_length)
        # valid_datasets = Datasets.BaseDataset(samples, args.zip_path, tokenizer, is_test=True, fix_length=args.fix_length)
        if model_config["version"] == "v1":
            predicter = Trainer.WXPredicter(args)
        elif model_config["version"] == "v2":
            predicter = Trainer.WXPredicterV2(args)
        elif model_config["version"] == "v3":
            predicter = Trainer.WXPredicterV3(args)
        elif model_config["version"] == "db_v1":
            predicter = Trainer.DoubleWXPredicter(args)
        predicter.trainer_init(valid_datasets)
        valid_iter = torch.utils.data.DataLoader(valid_datasets, batch_size=args.valid_batch_size)
        for model_path in model_config["model_path"]:
            predicter.model_load(model_path[0])
            # predicter.eval()
            if result is None:
                result = predicter.predict(valid_iter) * (model_path[1] / SUM)
            else:
                result += predicter.predict(valid_iter) * (model_path[1] / SUM)
    id_list = [item["vid"] for item in samples]
    golds = [item["labels"] for item in samples]
    result = np.argmax(result, axis=1)
    result = Utils.fetch_score(result, golds)
    logging.info(result)
    # with open("/users10/lyzhang/opt/tiger/WXData/output/ans_2.csv", 'w') as f:
    #     for pred_label_id, ann in zip(result, id_list):
    #         video_id = ann
    #         category_id = lv2id_to_category_id(pred_label_id)
    #         f.write(f'{video_id},{category_id}\n')


if __name__ == "__main__":
    args = Config.BaseConfig()
    Utils.set_seed(args.seed)
    if not args.debug:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.train:
        logging.info(f"args: {args}".replace(" ", "\n"))
        main(args)
    elif args.predict:
        predict(args)
