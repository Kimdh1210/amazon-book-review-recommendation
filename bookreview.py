import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings('ignore')

# n개 이상의 데이터 리뷰를 남긴 유저, 맥주를 걸러내는 함수
def preprocessing(data, n):
    min_id = data['아이디'].value_counts() >= n
    min_id = min_id[min_id].index.to_list()
    data = data[data['아이디'].isin(min_id)]

    min_beer = data['맥주'].value_counts() >= n
    min_beer = min_beer[min_beer].index.to_list()
    data = data[data['맥주'].isin(min_beer)]

    return data

temp=tmp.copy()

# 10번 반복합니다.
for i in range(1,10):
    temp = preprocessing(temp, 10)
    print(temp.shape)

temp.to_csv('정제된데이터.csv', encoding='utf-8')