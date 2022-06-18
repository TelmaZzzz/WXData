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
from category_id_map import CATEGORY_ID_LIST
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


args = Config.BaseConfig()
Utils.set_seed(args.seed)
logging.getLogger().setLevel(logging.INFO)
args.num_labels = len(CATEGORY_ID_LIST)
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_save = "/".join([args.model_save, Utils.d2s(datetime.datetime.now(), time=True)])
if not args.debug:
    if os.path.exists(model_save):
        logging.warning("save path exists, sleep 60s")
        raise
    else:
        os.mkdir(model_save)
        args.model_save = model_save
# args.model_save += "/pytorch_model.bin"
logging.info(f"model_save: {args.model_save}")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
special_token = {"additional_special_tokens": ["[T]", "[A]", "[O]"]}
tokenizer.add_special_tokens(special_token)
samples = Utils.prepare_testing_data(args.train_path, tokenizer, args.fix_length)
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
train_datasets = Datasets.PretrainDatasets(train_samples, args.zip_path, tokenizer, args.mask_ratio, fix_length=args.fix_length)
valid_datasets = Datasets.PretrainDatasets(valid_samples, args.zip_path, tokenizer, args.mask_ratio, is_test=True, fix_length=args.fix_length)
train_iter = torch.utils.data.DataLoader(train_datasets, batch_size=args.train_batch_size, shuffle=True)
trainer = Trainer.PretrainTrainer(args)
tokenizer.save_pretrained(args.model_save)
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