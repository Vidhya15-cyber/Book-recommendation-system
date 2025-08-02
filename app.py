import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Book Recommender - Content Based")

books = pd.read_csv("books.csv")
books['features'] = books['Title'] + " " + books['Author'] + " " + books['Genre']

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books['features'])
content_similarity = cosine_similarity(tfidf_matrix)

book_titles = books['Title'].tolist()
selected = st.selectbox("Choose a Book", book_titles)
index = books[books['Title'] == selected].index[0]

if st.button("Recommend"):
    similar_books = content_similarity[index].argsort()[::-1][1:4]
    st.write("### Recommended Books:")
    for i in similar_books:
        st.write("- " + books.iloc[i]['Title'])


import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
st.title("Book Recommender - User-Based Collaborative Filtering")

# Load data
ratings = pd.read_csv("ratings.csv")
books = pd.read_csv("books.csv")

# Merge to get book titles with ratings
merged = pd.merge(ratings, books, on="Book_ID")

# Create user-item matrix
user_item_matrix = merged.pivot_table(index='User_ID', columns='Title', values='Rating').fillna(0)

# Convert to sparse matrix
sparse_matrix = csr_matrix(user_item_matrix.values)

# Compute similarity between users
user_similarity = cosine_similarity(sparse_matrix)

# UI: Select a user
user_ids = user_item_matrix.index.tolist()
selected_user_id = st.selectbox("Choose a User ID", user_ids)

if st.button("Recommend", key="recommend_btn_1"):
    # Get index of selected user
    user_index = user_item_matrix.index.get_loc(selected_user_id)

    # Find similar users
    sim_scores = list(enumerate(user_similarity[user_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]  # exclude self

    # Take top N similar users
    top_users = [user_item_matrix.index[i] for i, _ in sim_scores[:3]]

    # Get books rated by similar users
    similar_users_ratings = user_item_matrix.loc[top_users]
    avg_ratings = similar_users_ratings.mean().sort_values(ascending=False)

    # Exclude books already rated by selected user
    already_rated = user_item_matrix.loc[selected_user_id]
    books_to_recommend = avg_ratings[already_rated == 0].head(3)

    st.write("### Recommended Books:")
    for title in books_to_recommend.index:
        st.write(f"- {title}")

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

st.title("Hybrid Book Recommender (Content + Collaborative)")

ratings = pd.read_csv("ratings.csv")
books = pd.read_csv("books.csv")

merged = pd.merge(ratings, books, on="Book_ID")

books['features'] = books['Title'] + " " + books['Author'] + " " + books['Genre']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['features'])
content_sim = cosine_similarity(tfidf_matrix)

book_id_to_index = {bid: idx for idx, bid in enumerate(books['Book_ID'])}
index_to_book_id = {idx: bid for bid, idx in book_id_to_index.items()}

user_item_matrix = merged.pivot_table(index='User_ID', columns='Book_ID', values='Rating').fillna(0)
sparse_matrix = csr_matrix(user_item_matrix.values)
collab_sim = cosine_similarity(sparse_matrix.T)

alpha = 0.5
hybrid_sim = alpha * content_sim + (1 - alpha) * collab_sim

book_options = books['Title'].tolist()
selected_title = st.selectbox("Choose a Book", book_options, key="book_selector")

if st.button("Recommend", key="recommend_button"):

    book_id = books[books['Title'] == selected_title]['Book_ID'].values[0]
    index = book_id_to_index[book_id]

    sim_scores = list(enumerate(hybrid_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]  # top 3

    st.write("Recommended Books:")
    for i, score in sim_scores:
        recommended_title = books.iloc[i]['Title']
        st.write(f"- {recommended_title} (Score: {score:.2f})")
