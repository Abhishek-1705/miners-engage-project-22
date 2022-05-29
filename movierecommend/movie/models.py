import warnings
from surprise.model_selection import cross_validate ,KFold
from surprise import Reader, Dataset, KNNWithMeans,SVD
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from ast import literal_eval
from scipy import stats
import seaborn as sns
import numpy as np
import pandas as pd
from django.forms import CharField
from django.shortcuts import render
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.storage import staticfiles_storage
from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField


# database connector extrenal
# import psycopg2
# from sqlalchemy import create_engine
# conn_string = 'postgresql://postgres:password@localhost:5432/Miners-movie'
# db = create_engine(conn_string)
# conn = db.connect()


# ------------------------------------------------------------------------------
# Importing the extrenal libary to process data
# ------------------------------------------------------------------------------


warnings.simplefilter('ignore')


class user(models.Model):
    id= models.OneToOneField("User", primary_key=True, on_delete=models.CASCADE)
    genres_selected = ArrayField(base_field=models.CharField(max_length=20), size=100, blank=True, null=True)
    movie_watched = ArrayField(base_field=models.CharField(max_length=20), size=100, blank=True, null=True)
    user_genders = models.CharField(max_length=20)

    class Meta:
        managed = False
        db_table = 'UserData'


class GenralRecommend(models.Model):
    id=models.BigIntegerField(primary_key=True)
    imdb_id = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'genral_recommend'

    def __str__(self):
        return self.imdb_id  





# This function convert he  pandas dataframe to sql and store to our postgressql DataBase
# def csv_to_sql(df,table_name):
#     conn_string = 'postgresql://postgres:password@localhost:5432/Miners-movie'
#     db = create_engine(conn_string)
#     conn = db.connect()

#     df.to_sql(table_name, con=conn, if_exists='replace',
#               index=True)
#     conn = psycopg2.connect(conn_string)
#     conn.autocommit = True
#     cursor = conn.cursor()
#     conn.close()





# __________________________________________________________________
    # Simple recommender
# __________________________________________________________________



md = pd. read_csv(staticfiles_storage.path("datasets/movies_metadata.csv"))


md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')

C = vote_averages.mean()
m = vote_counts.quantile(0.95)

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][[
    'title', 'vote_count', 'vote_average', 'popularity', 'imdb_id', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')

qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)
md2 = pd.DataFrame(qualified['imdb_id'].head(10))

# csv_to_sql(md2, 'genral_recommend')


# adding the genres column to dataframe

s = md.apply(lambda x: pd.Series(x['genres']),
             axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)


def build_chart(self,genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()
                       ]['vote_average'].astype('int')

    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (
        df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    qualified['wr'] = qualified.apply(lambda x: (
        x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)

    return qualified.head(10)





# ----------------------------------------------------------------------------
    # Content Based Recommender
# ---------------------------------------------------------------------------
credits = pd.read_csv(staticfiles_storage.path("datasets/credits.csv"))
keywords = pd.read_csv(staticfiles_storage.path("datasets/keywords.csv"))
links_small = pd.read_csv(staticfiles_storage.path("datasets/links_small.csv"))


links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')


md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')

smd = md[md['id'].isin(links_small)]

smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),min_df=0, stop_words='english')

tfidf_matrix = tf.fit_transform(smd['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])



keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')

md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')

smd = md[md['id'].isin(links_small)]

smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


smd['director'] = smd['crew'].apply(get_director)
smd['cast'] = smd['cast'].apply(lambda x: [i['name']for i in x] if isinstance(x, list) else [])

smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
smd['keywords'] = smd['keywords'].apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

smd['cast'] = smd['cast'].apply(
    lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['director'] = smd['director'].astype('str').apply(
    lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x, x, x])

s = smd.apply(lambda x: pd.Series(x['keywords']), axis=1).stack(
).reset_index(level=1, drop=True)
s.name = 'keyword'


s = s.value_counts()
s = s[s > 1]

stemmer = SnowballStemmer('english')


def filter_keywords(x):
    words = []
    for i in x:

        if i in s:
            words.append(i)
    return words


smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(
    lambda x: [str.lower(i.replace(" ", "")) for i in x])

smd['tags'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['tags'] = smd['tags'].apply(lambda x: ' '.join(x))

count = CountVectorizer(analyzer='word', ngram_range=(
    1, 2), min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['tags'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
smd = smd.reset_index()
titles = smd['title']

indices = pd.Series(smd.index, index=smd['title'])


def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][[
        'title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()
                         ]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull(
    )]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (
        movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values(
        'wr', ascending=False).head(10)
    return qualified.head(10)



# --------------------------------------------------------------------------------------
# Collaborative Filtering
# -------------------------------------------------------------------------------------



reader = Reader()

ratings = pd.read_csv(staticfiles_storage.path("datasets/ratings_small.csv"))

df = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
kf = KFold(n_splits=5)
kf.split(df) # Split the data into folds

 #Use Single Value Decomposition (SVD) for cross-validation and fitting
svd = SVD()
result=cross_validate(svd, df, measures=['RMSE', 'MAE'])
trainset = df.build_full_trainset()
svd.fit(trainset)

# -----------------------------------------
               # hybrid
# --------------------------------------------

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


id_map = pd.read_csv(staticfiles_storage.path( "datasets/links_small.csv"))[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
#id_map = id_map.set_index('tmdbId')
indices_map = id_map.set_index('id')



def hybrid(self,userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    #print(idx)
    movie_id = id_map.loc[title]['movieId']
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)
