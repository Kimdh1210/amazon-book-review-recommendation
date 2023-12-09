import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms import KNNWithMeans
from surprise.prediction_algorithms.slope_one import SlopeOne
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.prediction_algorithms.co_clustering import CoClustering
# 데이터 로드
ratings = pd.read_csv('./re_data/cleansed_data.csv')
ratings = ratings[['User_id', 'Id', 'total_score']]
 
# Reader 객체 생성
reader = Reader(rating_scale=(1, 5))  # 1-5점 척도 지정
 
# Surprise Dataset으로 변환
data = Dataset.load_from_df(ratings, reader)
 
# 알고리즘 객체들과 파라미터 그리드 정의
param_grid = {
    'k': [8,10],
    'sim_options': {'name': ['pearson_baseline'], 'user_based': [False]}
}
 
algos = {
    'SlopeOne': SlopeOne,
    'NMF': NMF,
    'CoClustering': CoClustering
}
 
# 그리드 서치 수행
for algo_name, algo_class in algos.items():
    print(f"Running GridSearchCV for {algo_name}")
    algo = algo_class()
    gs = GridSearchCV(algo, param_grid, measures=['rmse'], cv=3)
    gs.fit(data)
 
    # 최적 하이퍼파라미터와 최적 성능 출력
    print(f"Best RMSE for {algo_name}: {gs.best_score['rmse']}")
    print(f"Best params for {algo_name}: {gs.best_params['rmse']}")
    print("\n")