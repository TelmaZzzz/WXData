import json
import pandas as pd


with open("/users10/lyzhang/opt/tiger/WXData/data/data/annotations/labeled.json", "r", encoding="utf-8") as f:
    data = json.load(f)
asrs = []
ocrs = []
titles = []
print(len(data))
df_data = []
for item in data:
    titles.append(len(item["title"]))
    asrs.append(len(item["asr"]))
    ocr = ""
    for o in item["ocr"]:
        ocr += o["text"]
    ocrs.append(len(ocr))
    df_data.append((item["title"], item["asr"], ocr, item["category_id"]))
df = pd.DataFrame(df_data, columns=["title", "asr", "ocr", "category"])
df.to_csv("/users10/lyzhang/opt/tiger/WXData/data/tmp.csv", index=False)
# asrs = sorted(asrs)
# ocrs = sorted(ocrs)
# titles = sorted(titles)
# print(asrs[int(len(asrs) * 0.8)])
# print(ocrs[int(len(ocrs) * 0.8)])
# print(titles[int(len(titles) * 0.8)])
# print("-------")
# print(asrs[-1])
# print(ocrs[-1])
# print(titles[-1])