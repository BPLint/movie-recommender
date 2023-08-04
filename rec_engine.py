"""
    A movie recommendation engine built using the TMDb movie metadata dataset
    sourced from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata.
"""
import numpy as np
import pandas as pd

credits_data = 'data\\tmdb_5000_credits.csv'
movie_data = 'data\\tmdb_5000_movies.csv'

# read the movie data and the credits data into pandas dataframes 'movie_df' and 'credits_df'.
movie_df = pd.read_csv(movie_data)
credits_df = pd.read_csv(credits_data)

# correct the columns of the credits dataframe to match the movie dataframe in preparation for merge.
credits_df.columns = ['id','title','cast','crew']
# merge movie_df inplace with credits_df on the 'id' column
movie_df = movie_df.merge(credits_df, on='id')

# preview the merged dataframe
print(movie_df.head(5))
print(movie_df.columns)

"""
calculate the weighted ratings for the movies to account for sample size.
formula (WR) = ((v/v+m)*R) + ((m/v+m)*C) where:
WR = weighted rating
v = number of votes for each film
m = minimum number of votes a film requires to be listed in the chart
R = mean rating of the film
C = mean vote across the whole film report
"""

# the dataset already contains v and R, we must calculate C and m
# calculate C based on the 'vote_average' column
C = movie_df['vote_average'].mean()
print(C)

# define cutoff m as the 90th percentile of listed films
m = movie_df['vote_count'].quantile(0.9)
print(m)

# filter for movies that qualify (in the 90th percentile m)
q_films = movie_df.copy().loc[movie_df['vote_count'] >= m]
print(q_films.head(5))
print(q_films.shape)

# define a function to calculate the weighted rating for each film
def calc_wr(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculate using the aforementioned weighted rating formula
    return (v/(v+m)*R) + (m/(m+v) * C)
    
# define new feature 'score' and calculate using 'calc_wr()' function 
q_films['score'] = q_films.apply(calc_wr, axis=1)

# descending sort of qualified films by calculated weighted rating score
q_films = q_films.sort_values('score', ascending=False)
#q_fims = q_films.reset_index(drop=True)

# preview the top 10 films
print(q_films[['title_x', 'vote_count', 'vote_average', 'score']].head(10))
"""
===================================================================================================
=
implementation of content-based filtering to deliver more personalized recommendations to the user.
=
===================================================================================================
"""
# Create word vectors for each movie overview feature using sklearn term frequency-inverse document frequency vectorization functions
# import TfIdVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# define tf-idf vectorizer object. remove stop words.
tfidf = TfidfVectorizer(stop_words='english')

# replace nan values with empty string
q_films['overview'] = q_films['overview'].fillna('')
# reset index of qualified films dataframe
q_films = q_films.reset_index(drop=True)
# construct tf-idf matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(q_films['overview'])

# output the shape of the tf-idf matrix
print(tfidf_matrix.shape)

# calculate cosine similarity matrix of film overview word frequency vectors
# import linear kernel
from sklearn.metrics.pairwise import linear_kernel

# calculate cosine similarity matrix using 'linear_kernel()'
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# create reverse map of movie indices and titles
indices = pd.Series(q_films.index, index=q_films['title_x']).drop_duplicates()

# function that takes a film title as input and outputs the most similar films (based on cosine similarity matrix of overview word frequency vectors)
def get_recommendations(title, cosine_sim=cosine_sim):
    # get the index of the film matching the title arg
    idx = indices[title]
    # get pairwise similarity scores of all films with the given film
    sim_scores = list(enumerate(cosine_sim[idx]))
    # sort films based on similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    # get the movie indices for the 10 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    # return the 10 most similar movies
    return q_films['title_x'].iloc[movie_indices]

while True:
    user_movie_input = input('Please give the name of a movie you enjoy: ')
    try:
        print(get_recommendations(user_movie_input))
    except:
        print('Movie not found, please try again.')