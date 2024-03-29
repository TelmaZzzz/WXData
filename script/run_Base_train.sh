#!/bin/sh
# source ~/.bashrc
# source activate WXData

ROOT="/users10/lyzhang/opt/tiger/WXData"
export PYTHONPATH="$HOME/opt/tiger/WXData"


PRETRAIN="chinese_roberta_wwm_ext"
# PRETRAIN="chinese_roberta_wwm_ext_large"


python ../src/Base.py \
--train \
--debug \
--train_path="$ROOT/data/labeled_fold_10.json" \
--zip_path="$ROOT/data/data/zip_feats/labeled.zip" \
--pretrain_path="$HOME/model/$PRETRAIN" \
--tokenizer_path="$HOME/model/$PRETRAIN" \
--model_save="$ROOT/model/Base" \
--fold=0 \
--epoch=50 \
--lr=3e-5 \
--eval_step=50 \
--valid_batch_size=32 \
--train_batch_size=24 \
--fix_length=310 \
--scheduler="get_cosine_schedule_with_warmup" \
--mask_prob=0.0 \
--mask_ratio=0.0 \
--mode="base" \
--model_name="db_v1" \
> ../log/Base_train.log 2>&1 &