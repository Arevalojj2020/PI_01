#Import Modules

from fastapi import FastAPI
import numpy as np
import pandas as pd
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random
import time

#Read datasets

clean_dataset = pd.read_csv("clean_movies_dataset.csv")
df = pd.read_csv("movies_dataset.csv")

# API structure

app = FastAPI()
    
# Welcome

@app.get("/")
def welcome():
    return ("Welcome to the Movies API")

# Endpoints functions

@app.get("/peliculas_mes/{mes}")
def peliculas_mes(mes:str):
    
    month_movies = clean_dataset.loc[clean_dataset["release_month"] == mes].count()
    month_movies = str(month_movies["release_month"])
    
    return {"mes" : mes, "cantidad" : month_movies}

@app.get("/peliculas_dia/{dia}")
def peliculas_dia(dia:str):
    
    weekday_movies = clean_dataset.loc[clean_dataset["release_weekday"] == dia].count()
    weekday_movies = str(weekday_movies["release_weekday"])
    
    return {"dia" : dia, "cantidad" : weekday_movies}

@app.get("/franquicia/{franquicia}")
def franquicia(franquicia:str):
    
    franchise = clean_dataset.loc[clean_dataset["belongs_to_collection"] == franquicia]
    count = str(franchise["belongs_to_collection"].count())
    total_earning = str(round(franchise["revenue"].sum(), 2))
    earning_mean = str(round(franchise["revenue"].mean(), 2))
    
    return {"franquicia" : franquicia, "cantidad" : count, "ganancia_total" : total_earning, "ganancia_promedio" : earning_mean}

@app.get("/peliculas_pais/{pais}")
def peliculas_pais(pais:str):
    
    clean_dataset["production_countries"] = clean_dataset["production_countries"].fillna("")
    country = clean_dataset.loc[clean_dataset["production_countries"].str.contains(pais)]
    count = str(country["production_countries"].count())
    
    return {"pais" : pais, "cantidad" : count}

@app.get("/productora/{productora}")
def productoras(productora:str):
    
    clean_dataset["production_companies"] = clean_dataset["production_companies"].fillna("")
    company = clean_dataset.loc[clean_dataset["production_companies"].str.contains(productora)]
    count = str(company["production_companies"].count())
    total_earning = str(round(company["revenue"].sum(), 2))
    
    return {"productora" : productora, "ganancia_total" : total_earning, "cantidad" : count}

@app.get("/retorno/{pelicula}")
def retorno(pelicula:str):
    
    movie = clean_dataset.loc[clean_dataset["title"] == pelicula]
    movie = movie.head(1)
    investment = str(round(movie["budget"].iloc[0], 2))
    earning = str(round(movie["revenue"].iloc[0], 2))
    returns = str(round(movie["return"].iloc[0], 2))
    realese_year = str(movie["release_year"].iloc[0])
      
    return {"pelicula" : pelicula, "inversion" : investment, "ganacia" : earning, "retorno" : returns, "anio" : realese_year}

# Recomendation structure

tfidf = TfidfVectorizer(stop_words = "english")
df["overview"] = df["overview"].fillna('')
tfidf_matrix_1 = tfidf.fit_transform(df["overview"][:11366])
tfidf_matrix_2 = tfidf.fit_transform(df["overview"][11366:22732])
tfidf_matrix_3 = tfidf.fit_transform(df["overview"][22732:34098])
tfidf_matrix_4 = tfidf.fit_transform(df["overview"][34098:])
tfidf.get_feature_names_out()
cosine_sim_1 = linear_kernel(tfidf_matrix_1, tfidf_matrix_1)
time.sleep(5)
cosine_sim_2 = linear_kernel(tfidf_matrix_2, tfidf_matrix_2)
time.sleep(5)
cosine_sim_3 = linear_kernel(tfidf_matrix_3, tfidf_matrix_3)
time.sleep(5)
cosine_sim_4 = linear_kernel(tfidf_matrix_4, tfidf_matrix_4)
time.sleep(5)
index_1 = df["title"][:11366]
index_1.reset_index(drop = True, inplace = True)
index_1.drop_duplicates(inplace=True)
serie_1 = pd.Series(index_1.index)
df_index_1 = pd.DataFrame({"title":index_1.values, "indices":serie_1.values})
df_index_1.set_index("title")
serie_index_1 = pd.Series(df_index_1.index, index = df_index_1["title"]).drop_duplicates()
index_2 = df["title"][11366:22732]
index_2.reset_index(drop = True, inplace = True)
index_2.drop_duplicates(inplace=True)
serie_2 = pd.Series(index_2.index)
df_index_2 = pd.DataFrame({"title":index_2.values, "indices":serie_2.values})
df_index_2.set_index("title")
serie_index_2 = pd.Series(df_index_2.index, index = df_index_2["title"]).drop_duplicates()
index_3 = df["title"][22732:34098]
index_3.reset_index(drop = True, inplace = True)
index_3.drop_duplicates(inplace=True)
serie_3 = pd.Series(index_3.index)
df_index_3 = pd.DataFrame({"title":index_3.values, "indices":serie_3.values})
df_index_3.set_index("title")
serie_index_3 = pd.Series(df_index_3.index, index = df_index_3["title"]).drop_duplicates()
index_4 = df["title"][34098:]
index_4.reset_index(drop = True, inplace = True)
index_4.drop_duplicates(inplace=True)
serie_4 = pd.Series(index_4.index)
df_index_4 = pd.DataFrame({"title":index_4.values, "indices":serie_4.values})
df_index_4.set_index("title")
serie_index_4 = pd.Series(df_index_4.index, index = df_index_4["title"]).drop_duplicates()

@app.get("/recomendacion/{titulo}")
def recomendacion(titulo:str):
    
    functions = [get_recommendations_1, get_recommendations_2, get_recommendations_3, get_recommendations_4]
    
    for function in functions:
        try:
            result = function(titulo)
            return {"lista recomendada":list(result)}
        except Exception:
            pass

    return "Movie not found. Please try another one!"

def get_recommendations_1(titulo, cosine_sim = cosine_sim_1):
    idx = serie_index_1[titulo]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df["title"].iloc[movie_indices]


def get_recommendations_2(titulo, cosine_sim = cosine_sim_2):
    idx = serie_index_2[titulo]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df["title"].iloc[movie_indices]


def get_recommendations_3(titulo, cosine_sim = cosine_sim_3):
    idx = serie_index_3[titulo]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df["title"].iloc[movie_indices]

def get_recommendations_4(titulo, cosine_sim = cosine_sim_4):
    idx = serie_index_4[titulo]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df["title"].iloc[movie_indices]