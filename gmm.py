import os
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# === 1. 讀取 /dataset/ 下的 CSV 檔案 ===
data_dir = "./dataset/"
categories = ['game', 'health', 'politics']
dfs = []

for cat in categories:
    file_path = os.path.join(data_dir, f"{cat}.csv")
    df = pd.read_csv(file_path)
    df['label'] = cat  # 加入標籤
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print("資料集大小:", data.shape)

# 將文字標籤轉換為數值（方便計算 ARI、NMI）
label_map = {cat: idx for idx, cat in enumerate(categories)}
true_labels = data['label'].map(label_map).values

# === 2. 使用 BERT 產生嵌入向量 ===

# 使用 BERT 中文模型 (bert-base-chinese)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese").to(device)

# 定義一個函式來將標題轉換成 BERT 向量
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# 轉換整個數據集
embeddings = np.array([get_bert_embedding(title) for title in data['title']])
X = np.squeeze(embeddings, axis=1)
print("BERT 嵌入完成，向量形狀:", X.shape)

# === 3. 使用 GMM 進行軟性聚類 ===
num_clusters = len(categories)  # 聚類數 = 類別數
gmm = GaussianMixture(n_components=num_clusters, random_state=42)
clusters = gmm.fit_predict(X)

# === 4. 評估模型效果 (ARI、NMI) ===
ari = adjusted_rand_score(true_labels, clusters)
nmi = normalized_mutual_info_score(true_labels, clusters)

print("\n=== GMM 聚類評估 ===")
print(f"Adjusted Rand Index: {ari}")
print(f"Normalized Mutual Information: {nmi}")

