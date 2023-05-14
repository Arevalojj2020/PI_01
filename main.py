#Import Modules

from fastapi import FastAPI
import numpy as np
import pandas as pd
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#Read datasets

clean_dataset = pd.read_csv("clean_movies_dataset.csv")
ML_dataset = pd.read_csv("movies_dataset.csv")

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

ML_dataset = ML_dataset.drop_duplicates(subset = "title")
c = ML_dataset["vote_average"].mean()
m = ML_dataset["vote_count"].quantile(0.90)
ML_dataset = ML_dataset.loc[ML_dataset["vote_count"] >= m]
def weighted_rating(x, m = m, c = c):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v / (v + m) * R) + (m / (m + v) * c)
ML_dataset["score"] = ML_dataset.apply(weighted_rating, axis=1)
ML_dataset = ML_dataset.sort_values("score", ascending = False)
tfidf = TfidfVectorizer(stop_words = "english")
ML_dataset["overview"] = ML_dataset["overview"].fillna('')
tfidf_matrix = tfidf.fit_transform(ML_dataset["overview"])
tfidf.get_feature_names_out()
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
ML_dataset.reset_index(drop = True, inplace = True)
index = pd.Series(ML_dataset.index, index = ML_dataset["title"]).drop_duplicates()

@app.get("/recomendacion/{titulo}")
def recomendacion(titulo:str):
    local_cosine_sim = cosine_sim
    if titulo not in index:
        return "La película no se encuentra en el top 25 de mejores películas. Intenta con una mejor!"
    idx = index[titulo]
    sim_scores = list(enumerate(local_cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    result = ML_dataset["title"].iloc[movie_indices]
    return {"lista recomendada" : list(result)}