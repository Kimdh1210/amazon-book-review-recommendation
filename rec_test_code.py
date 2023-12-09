import numpy as np
import pandas as pd
from surprise import BaselineOnly, KNNWithMeans, SVD, SVDpp, Dataset, accuracy, Reader
from surprise.model_selection import cross_validate, train_test_split
import matplotlib.pyplot as plt

# 데이터 리드
ratings = pd.read_csv('Books_rating.csv')

#############################################################
# 책 기준 평가 수
min_book_ratings = 10
filter_books = ratings['Id'].value_counts() > min_book_ratings
filter_books = filter_books[filter_books].index.tolist()

# 유저 기준 평가 수
min_user_ratings = 3
filter_users = ratings['User_id'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()

# 평가 수 기준으로 컷
ratings_new = ratings[(ratings['Id'].isin(filter_books)) & (ratings['User_id'].isin(filter_users))]
print(f'The original data shape: {ratings.shape}')
print(f'The new data shape: {ratings_new.shape}')
#############################################################

# surprise 형식으로 데이터 변환 유저/상품/평점 순으로 적용해야 함
ratings = ratings_new[['User_id','Id','review/score']].dropna(axis=0)
reader = Reader(rating_scale=(1,5)) # 1-5점 척도 지정
data = Dataset.load_from_df(ratings[['User_id','Id','review/score']],reader)

# train test 분할 
trainset, testset = train_test_split(data, test_size=0.2)

from surprise.model_selection import GridSearchCV

# param_grid = {
#     'k':[10,20,30,40,50,60], # 군집수 만약에 끝값(10,60등)에 걸리면 범위 조정 후 다시 하는 게 좋음
#     'sim_options':{
#         'names':['pearson_baseline', 'cosine'],
#         'user_based':[False], # True로 할 시 user_based CF, False item_based CF
#         'min_k': [1, 2, 3],  # 추가적인 하이퍼파라미터
#         'shrinkage': [0, 0.1, 0.5],  # 추가적인 하이퍼파라미터
#     }
# }

param_grid = {
    'n_epochs': [10, 20, 30],  # 예측을 위한 에포크 수
    'lr_all': [0.002, 0.005, 0.01],  # 학습률
    'reg_all': [0.02, 0.1, 0.5],  # 정규화 상수
}

gs = GridSearchCV(#KNNWithMeans,
                  SVDpp,
                  param_grid,
                  measures = ['rmse'],
                  cv=4) # cross-validation

gs.fit(data)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])