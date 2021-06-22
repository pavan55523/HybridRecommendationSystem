import pickle
import json
import pandas as pd
import re as re
from flask import Flask, request, render_template, jsonify
import numpy as np

app = Flask(__name__)

# load data and extract all the vectors both content and collaborative
with open('content.pkl', 'rb') as f:
    books_data = pickle.load(f)
# all data about bx-datasets for collaborative
with open('collab.pkl', "rb") as f:
    loaded_list = pickle.load(f)
# combined titles data
# loading all titles pkl file
with open('titles.pkl', "rb") as t:
    titles_list = pickle.load(t)


# loading contents of pickle file
# content based data
book_data = pd.DataFrame(books_data)
list_books = [book['title'] for book in books_data]

# collaborative based data
model = loaded_list[0]
titles = loaded_list[1]
df = loaded_list[2]
books = loaded_list[3]
knn_model = pickle.loads(model)


def rcmd_content(m):
    c=0
    m=m.lower()
    for x in list_books:
        y = str(x).lower()
        if m==y:
            # print("content c:", c)
            break
        c=c+1
    if c<10101:
        book_details = books_data[c]
        return [book_details['title'],c]
    else:
        return['Sorry! The book your searched is not in our database.',-1]


def rcmd_collaborative(m):
    c = 0
    m = m.lower()
    for i in titles:
        y = str(i).lower()
        if m in y:
            # print("collab c:", c)
            break
        else:
            c = c + 1
    if c<2444:
        book_details = titles[c]
        return [book_details, c]
    else:
        return['sorry! the book you searches is not in our database.', -1]


def get_suggestions():
    return list(titles_list['title'].str.capitalize())


@app.route("/", methods=['GET', 'POST'])
def home():
    book = request.form.get('book')
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)


@app.route("/recommend")
def recommend():
    book = request.args.get('book').strip()  # get book name from the URL
    print(book)
    # output of content rcmd
    output_content = rcmd_content(book)
    r_content = output_content[0]
    ind_content = output_content[1]
    # output of collaborative rcmd
    output_collaborative = rcmd_collaborative(book)
    r_collaborative = output_collaborative[0]
    ind_collaborative = output_collaborative[1]

    # if no dataset has given book
    if ind_content == -1 and ind_collaborative == -1:
        suggestions = get_suggestions()
        return render_template('recommend.html', title=book, t='s', suggestions=suggestions)

    # if only content dataset has book details
    elif ind_collaborative == -1 and ind_content != -1:
        content_reco = 10
        selected_title_index = ind_content
        selected_book_isbn = book_data.loc[selected_title_index, 'ISBN']
        selected_book_title = book_data.loc[selected_title_index, 'title']
        selected_book_author = book_data.loc[selected_title_index, 'author']
        selected_book_image = book_data.loc[selected_title_index, 'image']
        selected_book_summary = book_data.loc[selected_title_index, 'summary']
        selected_book_similar = book_data.loc[selected_title_index, 'cosine']
        # print("content based similar books:", selected_book_similar)
        selected_book_similar = selected_book_similar[0:content_reco]
        poster_list = []
        book_titles_list = []
        for i in selected_book_similar:
            poster_list.append(book_data.loc[i, 'image'])
            book_titles_list.append(book_data.loc[i, 'title'])
        book_cards = {poster_list[i]: book_titles_list[i] for i in range(len(selected_book_similar))}
        # get book names for auto completion
        suggestions = get_suggestions()
        return render_template('recommend.html', t='l', cards=book_cards, title=selected_book_title,
                               isbn=selected_book_isbn,
                               image=selected_book_image, author=selected_book_author, summary=selected_book_summary,
                               suggestions=suggestions)

    # if only collaborative dataset has book details
    elif ind_content == -1 and ind_collaborative != -1:
        content_reco = 0
        collaborative_reco = 10
        selected_title_index = ind_collaborative
        selected_book_similar_titles = []
        selected_book_isbn = books.loc[selected_title_index, 'ISBN']
        selected_book_title = books.loc[selected_title_index, 'title']
        selected_book_author = books.loc[selected_title_index, 'author']
        selected_book_image = books.loc[selected_title_index, 'image_url']
        selected_book_summary = "Sorry!! summary not available"
        distances, indices = knn_model.kneighbors(df.iloc[selected_title_index, :].values.reshape(1, -1),
                                                  n_neighbors=collaborative_reco)
        for i in range(0, len(distances.flatten())):
            selected_book_similar_titles.append(df.index[indices.flatten()[i]])
        selected_book_similar = selected_book_similar_titles[:collaborative_reco]
        # print("collaborative based similar books titles:", selected_book_similar)
        poster_list_collaborative = []
        book_titles_list_collaborative = []
        book_cards = {}
        for i in selected_book_similar:
            ind = books[books['title'] == i].index[0]
            poster_list_collaborative.append(books.loc[ind, 'image_url'])
            book_titles_list_collaborative.append(books.loc[ind, 'title'])
        for i in range(10):
            book_cards[poster_list_collaborative[i]] = book_titles_list_collaborative[i]
            # get book names for auto completion
        suggestions = get_suggestions()
        return render_template('recommend.html', t='l', cards=book_cards, title=selected_book_title,
                               isbn=selected_book_isbn, image=selected_book_image, author=selected_book_author,
                               summary=selected_book_summary, suggestions=suggestions)
    # if book name is in both datasets
    else:
        content_reco = 5
        collaborative_reco = 5
        selected_title_index = ind_content
        selected_book_similar_collaborative = []

        # retrieving user input book details
        selected_book_isbn = book_data.loc[selected_title_index, 'ISBN']
        selected_book_title = book_data.loc[selected_title_index, 'title']
        selected_book_author = book_data.loc[selected_title_index, 'author']
        selected_book_image = book_data.loc[selected_title_index, 'image']
        selected_book_summary = book_data.loc[selected_title_index, 'summary']

        # retrieving content based 5 recos
        selected_book_similar_content = book_data.loc[ind_content, 'cosine']
        selected_book_similar_content = selected_book_similar_content[0:content_reco]
        # print("content cosine values indexes:", selected_book_similar_content)
        # print("Details of input book:")
        # print("Book title: ", selected_book_title)
        # print("ISBN number: ", selected_book_isbn)
        # print(" Author: ", selected_book_author)
        # print("Summary: ", selected_book_summary)

        # retrieving collaborative based 5 recos
        distances, indices = knn_model.kneighbors(df.iloc[selected_title_index, :].values.reshape(1, -1), n_neighbors=5)
        for i in range(0, collaborative_reco):
            selected_book_similar_collaborative.append(df.index[indices.flatten()[i]])

        # arranging poster and titles
        # for content based
        poster_list_content = []
        book_titles_list_content = []
        for i in selected_book_similar_content:
            poster_list_content.append(book_data.loc[i, 'image'])
            book_titles_list_content.append(book_data.loc[i, 'title'])
        # print("similar_books_content", book_titles_list_content)

        # for collaborative based
        poster_list_collaborative = []
        book_titles_list_collaborative = []
        for i in selected_book_similar_collaborative:
            ind = books[books['title'] == i].index[0]
            poster_list_collaborative.append(books.loc[ind, 'image_url'])
            book_titles_list_collaborative.append(books.loc[ind, 'title'])
        # print("similar_books_collab:", book_titles_list_collaborative)

        # forming a dictionary of poster and titles
        book_cards = {}
        for i in range(5):
            book_cards[poster_list_content[i]] = book_titles_list_content[i]
        for i in range(5):
            book_cards[poster_list_collaborative[i]] = book_titles_list_collaborative[i]
        # get book names for auto completion
        suggestions = get_suggestions()
        return render_template('recommend.html', t='l', cards=book_cards, title=selected_book_title,
                               isbn=selected_book_isbn,image=selected_book_image, author=selected_book_author,
                               summary=selected_book_summary, suggestions=suggestions)


if __name__ == '__main__':
    app.run(debug=True)
