# movie-recommender
A movie recommendation engine built using the TMDb movie metadata dataset sourced from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata.

# Purpose:
This project was created to recommend movies to the user based on movies that they enjoy; many services such as Netflix and Hulu use similar recommendation algorithms,
and I was interested in trying to build my own to see how they worked.

# How it works:
This recommendation model was created using self-reported movie ratings from users of TMDb. 
Weighted movie ratings were calculated using IMDb's movie weighted rating formula as follows:
formula (WR) = ((v/v+m)*R) + ((m/v+m)*C) where:
WR = weighted rating
v = number of votes for each film
m = minimum number of votes a film requires to be listed in the chart
R = mean rating of the film
C = mean vote across the whole film report

This calculation accounts for the extreme bias of low sample size on inflating or deflating movie ratings, as well as the bias of the userbase as a whole.
Movies in the 90th percentile of rating counts were used to calibrate the model for simplicity.

Movies were categorized based on their plot summaries. To accomplish this, word vectors were calculated for all film synopses using Scikit Learn's built-in 
term frequency-inverse document frequency vectorization functions. With an idea of word frequency, movie similarity can be approximated by calculating a
cosine similarity matrix based on the term frequency vector matrices for each film.

When presented with a movie, the recommendation engine will then return the 10 most similar movies based on the aforementioned calculated cosine similarity matrix.

# Limitations of the system:
- In its current rough iteration, the recommendation engine is only able to accept one "liked movie" as an input. Because film similarity is defined by term frequency similarity,
the similarity scores could be biased based on the writing style and quality of the plot synopsis provided.
- Additionally, the system does not take into account the preferences of previous users yet.
These aforementioned limitations result in strange recommendations sometimes; for example, someone who likes Christopher Nolan's "Interstellar" might not necessarily like Star Wars, for example.
While both contain sci-fi themes, plot structure, pacing, and tone are vastly different.

# Planned future improvements:
- Take multiple films as input
- Account for IMDb user ratings to tighten up similarity assessment
- Implement negative filtering (if I don't like a film, I want to see fewer films like it.)
