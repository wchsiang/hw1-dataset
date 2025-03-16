# hw1-dataset
data type: plain text in .csv file

training data(2024 news):

  /dataset/
  
  ├── game.csv      (遊戲新聞) amount: 14420
  
  ├── health.csv    (健康新聞) amount: 28813
  
  └── politics.csv  (政治新聞) amount: 32300

testing data(2025 news):

  /test/
  
  ├── game.csv      (遊戲新聞) amount: 2811
  
  ├── health.csv    (健康新聞) amount: 6389
  
  └── politics.csv  (政治新聞) amount: 7800

data collection:

  web_crawler.py
  
  reference - https://www.ettoday.net/news/news-list.htm
            
            - [wutienyang] (https://gist.github.com/wutienyang/6bf22fdb1f7e704ae7b7fd280f16beda)
