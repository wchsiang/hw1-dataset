import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

# 讀取 /dataset/ 下的 CSV 檔案，每個檔案代表一個分類
data_dir = "./dataset/"
categories = ['game', 'health', 'politics']
dfs = []
for cat in categories:
    file_path = os.path.join(data_dir, f"{cat}.csv")
    df = pd.read_csv(file_path)
    df['label'] = cat
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)
print("資料集大小:", data.shape)

test_dir = "./test/"
test_dfs = []
for cat in categories:
    file_path = os.path.join(test_dir, f"{cat}.csv")
    df = pd.read_csv(file_path)
    df['label'] = cat  # 添加標籤
    test_dfs.append(df)

# 合併測試數據
test_data = pd.concat(test_dfs, ignore_index=True)
print("測試資料集大小:", test_data.shape)

# TF-IDF 特徵萃取
chinese_stopwords = ["的", "是", "了", "我", "和", "就", "都", "要", "你", "不", "也", "在", "會", "說", "快訊", "獨家", "專訪"]
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), smooth_idf=True, sublinear_tf=True, stop_words=chinese_stopwords)
X_train = vectorizer.fit_transform(data['title'])
y_train = data['label']
X_test = vectorizer.transform(test_data['title'])  # 注意這裡是 transform
y_test = test_data['label']


smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 使用 Tomek Links 清理多餘邊界樣本
tomek = TomekLinks()
X_train, y_train = tomek.fit_resample(X_train, y_train)

# 定義 Random Forest 模型
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 5 折交叉驗證預測
cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(model_rf, X_train, y_train, cv=cv)
print("=== Random Forest 交叉驗證結果 ===")
print(classification_report(y_train, y_pred_cv))
print("混淆矩陣:\n", confusion_matrix(y_train, y_pred_cv))

# 在訓練集上訓練並評估測試集
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
print("\n=== 測試集評估 ===")
print("準確率:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("混淆矩陣:\n", confusion_matrix(y_test, y_pred))
