#!/bin/sh
source ~/.bashrc
source activate WXData

ROOT="/users10/lyzhang/opt/tiger/WXData"
export PYTHONPATH="$HOME/opt/tiger/WXData"


PRETRAIN="chinese_roberta_wwm_ext"
# PRETRAIN="/users10/lyzhang/opt/tiger/WXData/model/Pretrain/roberta_test"
# PRETRAIN="chinese_roberta_wwm_ext_large"
# PRETRAIN="chinese_macbert_base"
# PRETRAIN="chinese_pert_base"
# $HOME/model/$PRETRAIN
# PRETRAIN="/users10/lyzhang/opt/tiger/WXData/model/Pretrain/2022_06_06_21_02_38" # base
# PRETRAIN="/users10/lyzhang/opt/tiger/WXData/model/Pretrain/2022_06_07_21_50_43" # large
# PRETRAIN="/users10/lyzhang/opt/tiger/WXData/model/Pretrain/2022_06_14_20_25_33" # base itm itc mlm
PRETRAIN="/users10/lyzhang/opt/tiger/WXData/model/Pretrain/2022_06_18_23_00_36" # db_v1 base itm itc mlm

python ../src/Base.py \
--train \
--train_path="$ROOT/data/labeled_fold_10.json" \
--zip_path="$ROOT/data/data/zip_feats/labeled.zip" \
--pretrain_path="$PRETRAIN" \
--tokenizer_path="$PRETRAIN" \
--model_save="$ROOT/model/Base" \
--fold=0 \
--epoch=8 \
--lr=7e-5 \
--eval_step=1250 \
--valid_batch_size=32 \
--train_batch_size=32 \
--fix_length=310 \
--scheduler="get_cosine_schedule_with_warmup" \
--mask_prob=0.0 \
--mask_ratio=0.0 \
--mode="base" \
--patience_maxn=5 \
--model_name="db_v1" \
--fgm \
# --ema \
# --awp \
# --awp_up=0.61 \
# --awp_lr=5e-3 \
# --awp_eps=1e-4 \
# --model_load="/users10/lyzhang/opt/tiger/WXData/model/Base/2022_06_11_22_08_36/Fold_0.bin" \
# --pgd \
# --pgd_k=5 \
# --fgm \
# --freeze_nums=12 \