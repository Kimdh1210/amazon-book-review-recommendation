from django.shortcuts import render
from scenario_test import data_read, unseen_book, recsys_random_book, adding_new_book, merge_to_original_ReLearning, top_n, merge_to_original_ReLearning, add_book_to_dataframe

# 데이터 로드 및 추천 도서 가져오기
ratings, books, algo = data_read()

def index(request):
    return render(request, 'book/index.html')

def Home(request):
    return render(request, 'book/Home.html')

def ver1(request):
    return render(request, 'book/ver1.html')

def ver1_result(request):
    if request.method == 'POST':
        # POST 요청이 들어오면 사용자가 입력한 아이디를 가져옴
        User_id = request.POST.get('name')

        # 사용자 아이디가 데이터셋에 있는 경우에만 추천을 수행
        if User_id in ratings['User_id'].unique():
            seen_books, unseen_books = unseen_book(ratings, books, User_id)
            rec_books = recsys_random_book(algo, User_id, unseen_books, books, top_n=5)

            # 추천 도서 및 사용자 아이디를 context에 저장
            context = {
                'User_id': User_id,
                'rec_books': rec_books,
            }
            # ver1_result.html로 전환
            return render(request, 'book/ver1_result.html', context)
        else:
            # 사용자 아이디가 데이터셋에 없는 경우에는 다른 처리를 하거나 에러 메시지를 보여줄 수 있습니다.
            return render(request, 'book/ver1_error.html')

# def ver2(request):
#     return render(request, 'book/ver2_result.html')

def ver2_result_old(request):
    if request.method == 'POST':
        # Assuming your form fields are 'name', 'book1', 'rating1', 'book2', 'rating2', 'book3', 'rating3'
        User_id = request.POST.get('name')
        print(f"User_id:{User_id}")
        global ratings, books, algo
        # Assuming you have a list of books available for selection
        book_list = ["Book1", "Book2", "Book3"]  # Replace with your actual book list

        # Get user preferences from the form
        book1 = request.POST.get('book1')
        rating1 = int(request.POST.get('rating1'))
        book2 = request.POST.get('book2')
        rating2 = int(request.POST.get('rating2'))
        book3 = request.POST.get('book3')
        rating3 = int(request.POST.get('rating3'))

        # Add user preferences to the dataset
        new_DF_for_merge, length_new_DF_for_merge = adding_new_book(User_id, book1, rating1, book2, rating2, book3, rating3)
        merge_to_original_ReLearning(ratings, new_DF_for_merge, length_new_DF_for_merge)

        # Get unseen books for the new user
        seen_books, unseen_books = unseen_book(ratings, books, User_id)

        # Get book recommendations for the new user
        rec_books = recsys_random_book(algo, User_id, unseen_books, books, top_n=5)

        return render(request, 'book/ver2_result.html', {'User_id': User_id, 'rec_books': rec_books})

    else:
        # Render your ver2.html page with the form for user input
        # Assuming you have a list of books available for selection
        book_list = ["Book1", "Book2", "Book3"]  # Replace with your actual book list
        return render(request, 'book/ver2.html', {'book_list': book_list})
    
def ver2(request):

    # top_n 함수를 호출하여 각 카테고리별로 상위 3개의 책을 가져옵니다.
    top_books_result = top_n(books, up_num=100, top_num=3)
    
    # 각 카테고리별로 데이터프레임을 그룹화하고 딕셔너리로 변환
    books_by_category = dict(tuple(top_books_result.groupby('categories')))
    
    # 결과를 context에 저장
    context = {'books_by_category': books_by_category}
    # ver2.html로 전환
    return render(request, 'book/ver2.html', context)