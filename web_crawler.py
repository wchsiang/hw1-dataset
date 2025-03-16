# 載入python 套件
import requests
import os
from bs4 import BeautifulSoup  
import pandas as pd

# 抓 title
topics = {
    24: "game",
    21: "health",
    1: "politics"
}
save_path = "./dataset/"
os.makedirs(save_path, exist_ok=True)
for tt, topic in topics.items():
    title = []
    for month in range(1,13):
        for day in range(1,32):
            u = "https://www.ettoday.net/news/news-list-2024-"+str(month)+"-"+str(day)+"-"+str(tt)+".htm"
            res = requests.get(u)
            soup = BeautifulSoup(res.content, "lxml")
            soup = soup.find("div", class_="part_list_2")
            domian = "https://www.ettoday.net"
            for a in soup.find_all("h3"):
                p = a.a.string
                title.append(p)

    file_path = os.path.join(save_path, f"{topic}.csv")
    df = pd.DataFrame(title, columns=["title"])
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"已將 {topic} 類別的新聞標題儲存至 {file_path}")

save_path = "./test/"
os.makedirs(save_path, exist_ok=True)
for tt, topic in topics.items():
    title = []
    for month in range(1,4):
        for day in range(1,32):
            u = "https://www.ettoday.net/news/news-list-2025-"+str(month)+"-"+str(day)+"-"+str(tt)+".htm"
            res = requests.get(u)
            soup = BeautifulSoup(res.content, "lxml")
            soup = soup.find("div", class_="part_list_2")
            domian = "https://www.ettoday.net"
            for a in soup.find_all("h3"):
                p = a.find("a")
                if p and p.string:
                    p = p.string
                title.append(p)

    file_path = os.path.join(save_path, f"{topic}.csv")
    df = pd.DataFrame(title, columns=["title"])
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"已將 {topic} 類別的新聞標題儲存至 {file_path}")