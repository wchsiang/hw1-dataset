import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

# === 1. 讀取數據集 ===
data_dir = "./dataset/"
test_dir = "./test/"
categories = ['game', 'health', 'politics']
dfs = []

for cat in categories:
    file_path = os.path.join(data_dir, f"{cat}.csv")
    df = pd.read_csv(file_path)
    df['label'] = cat  # 加入標籤
    dfs.append(df)

# 合併所有類別數據
data = pd.concat(dfs, ignore_index=True)
print("資料集大小:", data.shape)

test_dfs = []
for cat in categories:
    file_path = os.path.join(test_dir, f"{cat}.csv")
    df = pd.read_csv(file_path)
    df['label'] = cat  # 添加標籤
    test_dfs.append(df)

# 合併測試數據
test_data = pd.concat(test_dfs, ignore_index=True)
print("測試資料集大小:", test_data.shape)

# === 2. 特徵工程：TF-IDF ===
# 增強特徵表達能力 (使用 n-grams, smooth_idf)
chinese_stopwords = ["的", "是", "了", "我", "和", "就", "都", "要", "你", "不", "也", "在", "會", "說", "快訊", "獨家", "專訪"]
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), smooth_idf=True, sublinear_tf=True, stop_words=chinese_stopwords)
X_train = vectorizer.fit_transform(data['title'])
y_train = data['label']
X_test = vectorizer.transform(test_data['title'])  # 注意這裡是 transform
y_test = test_data['label']

# === 4. 處理不平衡資料 (SMOTE + Tomek Links) ===
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 使用 Tomek Links 清理多餘邊界樣本
tomek = TomekLinks()
X_train, y_train = tomek.fit_resample(X_train, y_train)

# === 5. 定義 Logistic Regression 模型 (加入 class_weight, 調整 C 值) ===
model_lr = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced', C=0.5)

# === 6. 使用 StratifiedKFold 進行交叉驗證預測 ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(model_lr, X_train, y_train, cv=cv)

# 交叉驗證結果
print("=== Logistic Regression 交叉驗證結果 ===")
print(classification_report(y_train, y_pred_cv))
print("混淆矩陣:\n", confusion_matrix(y_train, y_pred_cv))

# === 7. 使用訓練集擬合模型，並在測試集上評估 ===
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)

# 測試集結果
print("\n=== 測試集評估 ===")
print("準確率:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("混淆矩陣:\n", confusion_matrix(y_test, y_pred))
