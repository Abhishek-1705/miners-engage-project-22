

# Minres 




### A Movie recommender system

Miners is a movie recommender Web application, that recommend you the Movie based on the our previous watched, and other user collective rating on a movie.

This recommendation bases work on  principle:-
1. #### Content based recommendation
(https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering)

   A Content-Based Recommender works by the data that we take from the user, either explicitly (rating) or implicitly (clicking on a link). By the data we create a user profile, which is then used to suggest to the user, as the user provides more input or take more actions on the recommendation, the engine becomes more accurate. 

2. #### Collaborative Filtering (https://en.wikipedia.org/wiki/Collaborative_filtering)
In Collaborative Filtering, we tend to find similar users and recommend what similar users like. In this type of recommendation system, we don’t use the features of the item to recommend it, rather we classify the users into the clusters of similar types, and recommend each user according to the preference of its cluster. 

3. #### Hybrid System
(https://en.wikipedia.org/wiki/Recommender_system#Hybrid_recommender_systems)
A hybrid recommendation system is a special type of recommendation system which can be considered as the combination of the content and collaborative filtering method. Combining collaborative and content-based filtering


## Recommendation used

1. #### Simpler Recommendation
  The Simple Recommender offers generalized recommnendations to every user based on movie popularity and (sometimes) genre. 
   The implementation of this model is extremely trivial. All we have to do is sort our movies based on ratings and popularity and display the top movies of our list
 
  I have use weighted rating formula to construct my chart. Mathematically, it is represented as follows: 

  Weighted Rating (WR) =  (v/v+m.R)+(m/v+m.C) 
where,

- v is the number of votes for the movie
- m is the minimum votes required to be listed in the chart
- R is the average rating of the movie
- C is the mean vote across the whole report

2. #### Content Based Recommender
To personalise our recommendations more, I have build an engine that computes similarity between movies based on certain metrics and suggests movies that are most similar to a particular movie that a user liked.  i have use movie metadata (or content) to build this engine, this also known as Content Based Filtering.
- Cosine Similarity
I have  use the Cosine Similarity to calculate a numeric quantity that denotes the similarity between two movies. Mathematically, it is defined as follows:

cosine(x,y)=x.y⊺||x||.||y||

I have used  the TF-IDF Vectorizer, calculating the Dot Product will directly give us the Cosine Similarity Score. 

3. #### Collaborative Filtering 

To make recommendations to Movie Watchers. Collaborative Filtering is based on the idea that users similar to a me can be used to predict how much I will like a particular product or service those users have used/experienced but I have not.

 I have use the Surprise library that used extremely powerful algorithms like Singular Value Decomposition (SVD) to minimise RMSE (Root Mean Square Error) and give great recommendations.

 

 - #### SVD(Signular Value Decomposition) Algorithm 
 Singular value decomposition (SVD) is a matrix factorization method that generalizes the eigendecomposition of a square matrix (n x n) to any matrix (n x m) (source).
 It uses Probabilistic Matrix Factorization (PMF) 

4. #### Hybrid Recommender
I have build a simple hybrid recommender that brings together techniques we have implemented in the content based and collaborative filter based engines. 
It use :
- Input: User ID and the Title of a Movie
- Output: Similar movies sorted on the basis of expected ratings by that particular user

from the content based i have pickedup the cosine similarity matrix ,and from i have use the SVD predict.





## Installation

To run this project you should have python3.7.9  install in your system,
if not install checkout this https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe

To check your python version
```bash
Python --version 
```


Also Need to have  pip python package manager install ,
if not install, Follow this step:

```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
pip --version
```
Install the Database, we using the postgresql
- For Windows
Download the postgresql from here :
https://www.enterprisedb.com/postgresql-tutorial-resources-training?uuid=db55e32d-e9f0-4d7c-9aef-b17d01210704&campaignId=7012J000001NhszQAC

Connecting to the  Database
```bash
psql -U userName
when promoted enter password
```
- For linux 
```bash
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
```
conecting to the Database in linux
```bash
sudo su - postgres
psql

```
create the Database
```bash
CREATE DATABASE Miners-movie WITH ENCODING 'UTF8' LC_COLLATE='English_United Kingdom' LC_CTYPE='English_United Kingdom';
```
clone this repository
```bash
git clone 
```
```bash
  cd 
```

Now, python virtual environment 
```bash
python3 -m pip install --user virtualenv 
py -m venv myenv
.\myenv\Scripts\activate
```
Install the necessary packages/libary to run this project
```bash
pip install requirements.txt
```


```bash
cd .\movierecommend\
```
collecting the Static files 
```bash
py manage.py collectstatic
```
migrating the models to the database
```bash
py manage.py makemigrations

py manage.py migrate
```
##### Now start the sever 
```bash
py manage.py runserver
```
The aplication is hosted on your localhost
view it by visiting link
```bash 
 http://127.0.0.1:8000/
```

