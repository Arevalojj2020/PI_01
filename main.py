# Import Modules

from fastapi import FastAPI
import numpy as np
import pandas as pd
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load datasets

clean_dataset = pd.read_csv("datasets/clean_movies_dataset.csv")
ML_dataset = pd.read_csv("datasets/movies_dataset.csv")

# API structure

app = FastAPI()
    
# Welcome

@app.get("/")
def Bienvenida():
    return ("Bienvenido al sistema de recomendación de películas!")

# Endpoints functions

@app.get("/peliculas_mes/{mes}")
def peliculas_mes(mes:str):
    
    '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes históricamente.'''
    
    months = list(clean_dataset["release_month"].unique())
    mes = mes.lower()
    if mes in months:
        month_movies = clean_dataset.loc[clean_dataset["release_month"] == mes].count()
        return {"mes" : mes, "cantidad" : str(month_movies["release_month"])}
    else: return "Ingrese un mes válido"

@app.get("/peliculas_dia/{dia}")
def peliculas_dia(dia:str):
    
    '''Se ingresa el dia y la función retorna la cantidad de películas que se estrenaron ese dia históricamente.'''
    
    days = list(clean_dataset["release_weekday"].unique())
    dia = dia.lower()
    if dia in days:
        weekday_movies = clean_dataset.loc[clean_dataset["release_weekday"] == dia].count()
        return {"dia" : dia, "cantidad" : str(weekday_movies["release_weekday"])}
    else: return "Ingrese un dia válido"

@app.get("/franquicia/{franquicia}")
def franquicia(franquicia:str):
    
    '''Se ingresa la franquicia, retornando la cantidad de películas, ganancia total y promedio.'''
    
    franchise = clean_dataset["belongs_to_collection"].unique()
    franchise = list(franchise)
    if franquicia in franchise:
        franchise = clean_dataset.loc[clean_dataset["belongs_to_collection"] == franquicia]
        count = franchise["belongs_to_collection"].count()
        total_earning = franchise["revenue"].sum()
        earning_mean = franchise["revenue"].mean()
        return {"franquicia" : franquicia, "cantidad" : str(count), "ganancia_total" : str(round(total_earning, 2)), "ganancia_promedio" : str(round(earning_mean, 2))}
    
    else: return "Ingrese una franquicia válida"

@app.get("/peliculas_pais/{pais}")
def peliculas_pais(pais:str):
    
    '''Ingresas el país, retornando la cantidad de películas producidas en el mismo.'''
    
    clean_dataset["production_countries"] = clean_dataset["production_countries"].fillna("")
    country = clean_dataset.loc[clean_dataset["production_countries"].str.contains(pais)]
    count = country["production_countries"].count()
    if count == 0:
        return "Ingrese un país válido"
    else: return {"pais" : pais, "cantidad" : str(count)}

@app.get("/productora/{productora}")
def productoras(productora:str):
    
    '''Ingresas la productora, retornando la ganancia total y la cantidad de películas que produjeron.'''
    
    clean_dataset["production_companies"] = clean_dataset["production_companies"].fillna("")
    company = clean_dataset.loc[clean_dataset["production_companies"].str.contains(productora)]
    count = company["production_companies"].count()
    total_earning = company["revenue"].sum()
    if count == 0:
        return "Ingrese una productora válida"
    else: return {"productora" : productora, "ganancia_total" : str(round(total_earning, 2)), "cantidad" : str(count)}

@app.get("/retorno/{pelicula}")
def retorno(pelicula:str):
    
    '''Ingresas la película, retornando la inversión, la ganancia, el retorno y el año en el que se lanzó.'''
    
    titles = list(clean_dataset["title"].unique())
    if pelicula in titles:
        movie = clean_dataset.loc[clean_dataset["title"] == pelicula]
        movie = movie.head(1)
        investment = movie["budget"].iloc[0]
        earning = movie["revenue"].iloc[0]
        returns = movie["return"].iloc[0]
        realese_year = movie["release_year"].iloc[0]
        return {"pelicula" : pelicula, "inversion" : str(round(investment, 2)), "ganacia" : str(round(earning, 2)), "retorno" : str(round(returns, 2)), "anio" : str(realese_year)}
    else: return "Ingrese un película válida"

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
    
    '''Ingresas un nombre de pelicula y te recomienda 5 similares en una lista.'''
    
    local_cosine_sim = cosine_sim
    if titulo not in index:
        return "La película no se encuentra entre el 10% de las mejores películas. Intenta con una mejor!"
    idx = index[titulo]
    sim_scores = list(enumerate(local_cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    result = ML_dataset["title"].iloc[movie_indices]
    return {"lista recomendada" : list(result)}