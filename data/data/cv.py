import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold
import random


sgcv = StratifiedKFold(n_splits=10, shuffle=True, random_state=959)
with open("/users10/lyzhang/opt/tiger/WXData/data/data/annotations/labeled.json", "r", encoding="utf-8") as f:
    data = json.load(f)
category_id = [int(item["category_id"]) for item in data]
# index = list(range(len(data)))
for fold, (train_ids, valid_ids) in enumerate(sgcv.split(data, category_id)):
    for id in valid_ids:
        data[id]["fold"] = fold
print(data[:5])
with open("/users10/lyzhang/opt/tiger/WXData/data/labeled_fold_10.json", "w", encoding="utf-8") as f:
    json.dump(data, f)


# with open("/users10/lyzhang/opt/tiger/WXData/data/data/annotations/unlabeled.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
# index = list(range(len(data)))
# random.shuffle(index)
# for i, idx in enumerate(index):
#     if i > int(len(index) * 0.95):
#         data[idx]["fold"] = 0
#     else:
#         data[idx]["fold"] = 1
# print(data[:5])
# with open("/users10/lyzhang/opt/tiger/WXData/data/unlabeled_fold_20.json", "w", encoding="utf-8") as f:
#     json.dump(data, f)