#!/bin/sh
source ~/.bashrc
source activate WXData

ROOT="/users10/lyzhang/opt/tiger/WXData"
export PYTHONPATH="$HOME/opt/tiger/WXData"


# PRETRAIN="chinese_roberta_wwm_ext"
PRETRAIN="chinese_roberta_wwm_ext_large"
# PRETRAIN="chinese_macbert_base"
# PRETRAIN="chinese_pert_base"

# --train_path="$ROOT/data/unlabeled_fold_20.json" \
# --zip_path="$ROOT/data/data/zip_feats/unlabeled.zip" \

# eval_step base 7000 large 21000
# train_batch_sze base 64 large 24

python ../src/Pretrain.py \
--train \
--train_path="$ROOT/data/unlabeled_fold_20.json" \
--zip_path="$ROOT/data/data/zip_feats/unlabeled.zip" \
--pretrain_path="$HOME/model/$PRETRAIN" \
--tokenizer_path="$HOME/model/$PRETRAIN" \
--model_save="$ROOT/model/Pretrain" \
--fold=0 \
--epoch=8 \
--lr=3e-5 \
--eval_step=56000 \
--valid_batch_size=30 \
--train_batch_size=8 \
--fix_length=310 \
--scheduler="get_cosine_schedule_with_warmup" \
--mask_prob=1.0 \
--mask_ratio=0.15 \
--mode="base" \
--patience_maxn=3 \
# > ../log/Base_train.log 2>&1 &
# > ../log/Base_train.log 2>&1 &
# --warmup_step=4000 \
# > ../log/Base_train.log 2>&1 &
# --awp \
# --awp_up=0.6 \
# --awp_lr=1.0 \
# --awp_eps=1e-4 \
# --ema \
# --fgm \