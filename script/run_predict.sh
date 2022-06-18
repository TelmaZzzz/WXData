#!/bin/sh
# source ~/.bashrc
# source activate WXData

ROOT="/users10/lyzhang/opt/tiger/WXData"
export PYTHONPATH="$HOME/opt/tiger/WXData"


PRETRAIN="chinese_roberta_wwm_ext"
# PRETRAIN="/users10/lyzhang/opt/tiger/WXData/model/Pretrain/roberta_test"
# PRETRAIN="chinese_roberta_wwm_ext_large"
# PRETRAIN="chinese_macbert_base"
# PRETRAIN="chinese_pert_base"
# $HOME/model/$PRETRAIN

# --train_path="$ROOT/data/data/annotations/test_a.json" \
# --zip_path="$ROOT/data/data/zip_feats/test_a.zip" \

python ../src/Base.py \
--predict \
--train_path="$ROOT/data/data/annotations/test_a.json" \
--zip_path="$ROOT/data/data/zip_feats/test_a.zip" \
--pretrain_path="/users10/lyzhang/opt/tiger/WXData/model/Pretrain/2022_06_06_21_02_38" \
--tokenizer_path="/users10/lyzhang/opt/tiger/WXData/model/Pretrain/2022_06_06_21_02_38" \
--valid_batch_size=64 \
--fix_length=310 \
# --fold=0 \