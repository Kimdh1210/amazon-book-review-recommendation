# 전처리 주요항목
# Books_rating_sample.csv 에서
# 텍스트 분석후 리뷰점수가 0-5점으로 추가된다는 전제하에.
# review/score = review/add_point 일시적으로 점수 생성.
# 이 부분은 차후 텍스트분석으로 생성되는 [리뷰점수]로 대체함
# 이거 테스트 코드는 review/score 필드의 점수만 사용함.
# 여기는 유클리디안 유사도함수 사용

import pandas as pd
from scipy.spatial.distance import pdist, squareform

# 데이터 로드
books_df = pd.read_csv('books_data_sample.csv', encoding='ISO-8859-1')
ratings_df = pd.read_csv('Books_rating_sample.csv', encoding='ISO-8859-1')

# 데이터 전처리
# 'Title' 컬럼을 기준으로 두 데이터프레임을 병합
merged_df = pd.merge(books_df, ratings_df, on='Title', how='inner')

# 사용자-책 평점 행렬 생성
user_book_rating = pd.pivot_table(merged_df, index='User_id', columns='Title', values='review/score').fillna(0)

# 유클리디안 거리 행렬 계산
euclidean_dist = squareform(pdist(user_book_rating, metric='euclidean'))

# 유클리디안 거리를 유사도 점수로 변환 (1 / (1 + 거리))
euclidean_similarity = 1 / (1 + euclidean_dist)

# 유사도 행렬을 데이터프레임으로 변환
user_sim_df = pd.DataFrame(euclidean_similarity, index=user_book_rating.index, columns=user_book_rating.index)

# 추천 함수 정의
def recommend_books(user_id, num_recommendations):
    if user_id not in user_sim_df.index:
        return (f"사용자 ID {user_id}를 찾을 수 없습니다.")

    # 사용자의 유사도 점수 가져오기
    user_similarity_scores = user_sim_df.loc[user_id]

    # 유사도 점수에 따라 사용자 정렬
    user_similarity_scores = user_similarity_scores.sort_values(ascending=False)

    # 가장 유사한 사용자들 추출
    top_users = user_similarity_scores.iloc[1:num_recommendations + 1].index

    # 이 사용자들의 평점 가져오기
    top_user_ratings = user_book_rating.loc[top_users]

    # 상위 사용자들의 평균 평점을 통해 추천 책 점수 계산
    recommended_books_scores = top_user_ratings.mean(axis=0)
    recommended_books_scores = recommended_books_scores.sort_values(ascending=False)

    # 추천 책 제목 가져오기
    recommended_titles = recommended_books_scores.index.tolist()

    return recommended_titles[:num_recommendations]

# 모든 유저 아이디에 대해 책 5권 추천
all_user_recommendations = {}

# 모든 유저 아이디를 순회하면서 각각에 대한 추천 실행
for user_id in ratings_df['User_id'].unique():
    recommended_books_for_user = recommend_books(user_id, 5)
    all_user_recommendations[user_id] = recommended_books_for_user

# 추천 결과 출력
for user_id, recommendations in all_user_recommendations.items():
    print(f"User ID: {user_id}")
    print(f"추천된 책 목록: {recommendations}")
    print()  # 빈 줄로 구분
