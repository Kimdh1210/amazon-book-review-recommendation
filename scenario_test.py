
import pandas as pd
from surprise import Dataset, accuracy, Reader, KNNBaseline
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from joblib import dump, load
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def data_read():
    ratings = pd.read_csv('./book/cleansed_data.csv')
    books = pd.read_csv('./book/books_data.csv')
    books= pd.merge(ratings[['Title','Id']], books, on='Title', how='inner')

    books = books.drop_duplicates(subset=['Title'])
    books = books[['Title', 'Id', 'authors', 'publishedDate', 'categories']]
    
    ratings = ratings[['User_id','Id','total_score']]

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
    ratings = ratings_new

    algo = load('best_algo.joblib')
    return ratings, books, algo

def unseen_book(ratings, books, User_id):
    seen_books = ratings[ratings['User_id']== User_id]['Id'].tolist()
    
    # 모든 책명을 list 객체로 만듬 
    total_books = books['Id'].tolist()
      
    unseen_books_set = set(total_books) - set(seen_books)
    unseen_books = list(unseen_books_set)
    
    # 일부 정보 출력
    total_movie_cnt = len(total_books)
    seen_cnt = len(seen_books)
    unseen_cnt = len(unseen_books)
    
    print(f"전체 책 수: {total_movie_cnt}, 평점 매긴 책 수: {seen_cnt}, 추천 대상 책 수: {unseen_cnt}")
    
    return seen_books, unseen_books

def recsys_random_book(algo, User_id, unseen_book, top_n=5):
    
    # 예측 평점: prediction 객체 생성
    predictions = []    
    for Id in unseen_book:
        predictions.append(algo.predict(str(User_id), str(Id)))
    
    # 리스트 내의 prediction 객체의 est를 기준으로 내림차순 정렬
    def sortkey_est(pred):
        return pred.est

    predictions.sort(key=sortkey_est, reverse=True) # key에 리스트 내 객체의 정렬 기준을 입력
    
    # 예상 평점 상위 100권에서 n권 추출
    top100_predictions = predictions[:top_n*10]
    random_N = random.sample(top100_predictions, top_n)
    # 아이디, 제목, 예측 평점 출력
    print(f"How about This {top_n} BOOKs !")
    
    for pred in random_N:
        
        book_id = str(pred.iid)
        book_title = books[books["Id"] == book_id]["Title"].tolist()
        book_rating = pred.est
        book_authors = books[books["Id"] == book_id]["authors"].tolist()
        book_date = books[books["Id"] == book_id]["publishedDate"].tolist()
        book_categories = books[books["Id"] == book_id]["categories"].tolist()

        print(f"{book_title}:{book_authors}({book_date})({book_categories})")
        print(f"expected score is...{book_rating:.2f}")

#######################################################################################################
# Make new DF for data appending() 
def adding_new_book(User_id):
    item_name, Id = input_book_name()
    rating = input_rating()
    text = input_text()
    new_DF_for_merge = pd.read_csv('new_DF_for_merge.csv')
    
    new_row_data = {'User_id': User_id, 'Title': item_name, 'review/score': rating, 'review/text':text, 'Id':Id}
    
    
    new_DF_for_merge = new_DF_for_merge._append(new_row_data, ignore_index=True)
    new_DF_for_merge = new_DF_for_merge.drop_duplicates(['User_id', 'Title'], keep='last')
    length_new_DF_for_merge = len(new_DF_for_merge)
    print(new_DF_for_merge)
    new_DF_for_merge.to_csv('new_DF_for_merge.csv', index=False)
    return new_DF_for_merge, length_new_DF_for_merge
# Book Only in ourt list
def input_book_name():
    while True:
        try:
            book_name = input('Enter Book Title:')
            if not books[books['Title'] == book_name].empty:
                Id = books[books['Title'] == book_name].iloc[0]['Id']
                return book_name, Id
            else:
                print('Sorry. This book is not in our list!')
        except ValueError:
            print("retry")

# Score Only int type between 1-5 
def input_rating():
    while True:
        try:
            rating = int(input("Enter your score (1-5): "))
            if 1 <= rating <= 5:
                return rating
            else:
                print("Please enter between 1-5 point!")
        except ValueError:
            print("retry")

def input_text():
    while True:
        try:
            text = input("Enter your text up to 512 characters: ")
            if len(text) <= 512:
                return text
            else:
                print("Please enter up to 512!")
        except ValueError:
            print("retry")
#######################################################################################################
# Appending extra_df to original_df if meet the condition
def merge_to_original_ReLearning(original_df, new_DF_for_merge, length_new_df):
    if length_new_df >= 100: 
        new_DF_for_merge = review_sentiment_make_total()
        original_df = original_df._append(new_DF_for_merge, ignore_index=True)
        append_og_data_Learning(original_df, new_DF_for_merge)
        new_DF_for_merge.drop(index=new_DF_for_merge.index, inplace=True) # save after reset
        new_DF_for_merge.to_csv('new_DF_for_merge.csv', index=False)
    
    # Sentiment ananlysis(1-5) by Electra
    def review_sentiment_make_total():  
        # For backup save
        last_add_df = new_DF_for_merge.copy()      
        last_add_df.to_csv('last_extra_df.csv', index=False)
        # reviews prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained('nlp_tokenizer')
        model = AutoModelForSequenceClassification.from_pretrained('nlp_model').to(device)

        for i in range(length_new_df):
            texts = new_DF_for_merge['review/text']
            scores = new_DF_for_merge['review/score']
            predicted_classes = [] 

            for text, score in zip(texts, scores):
                if pd.isna(text):
                    predicted_class = int(score)  
                else:
                    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).to(device)
                            
                    with torch.no_grad():
                        outputs = model(**inputs)
                                
                    logits = outputs.logits
                    predicted_class = torch.argmax(logits, dim=1).item() +1

                predicted_classes.append(predicted_class)

            new_DF_for_merge['predicted_class'] = predicted_classes
            new_DF_for_merge['total_score'] = new_DF_for_merge.apply(lambda row: row['review/score']*0.8+(row['predicted_class'])*0.2, axis=1)
        return new_DF_for_merge
    
    def append_og_data_Learning(original_df, new_DF_for_merge):
        data = original_df[['User_id','Id','total_score']]
        extra = new_DF_for_merge[['User_id','Id','total_score']]
        data = data._append(extra, ignore_index=True)
        data = data.drop_duplicates(['User_id', 'Id'], keep='last')
        reader = Reader(rating_scale=(1,5)) # 1-5점 척도 지정
        data = Dataset.load_from_df(data,reader)

        trainset, testset = train_test_split(data, test_size=0.05, random_state=42)

        sim_options = {'name': 'cosine','user_based': False}
        new_algo = KNNBaseline(k=7, sim_options=sim_options)
        new_algo.fit(trainset)
        
        predictions = new_algo.test(testset)
        accuracy.rmse(predictions)
        model_filename = 'new_algo.joblib'
        dump(new_algo, model_filename)

##########################################################################################################
if __name__ == '__main__':
    ratings, books, algo = data_read()
    User_id = input('Enter your ID:')
    # seen_books, unseen_books = unseen_book(ratings, books, User_id)
    # recsys_random_book(algo, User_id, unseen_books, top_n=5)

    # 데이터 추가 및 재학습
    new_DF_for_merge, length_new_DF_for_merge = adding_new_book(User_id) 
    merge_to_original_ReLearning(ratings, new_DF_for_merge, length_new_DF_for_merge)

