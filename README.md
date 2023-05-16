# Machine Learning Operations Project

In this project, we will analyze the lifecycle of a movie dataset from its data transformation to training a machine learning model that can take a movie title as input and return a list recommending 5 movies similar to it.

#### Datasets that will be used: https://github.com/Arevalojj2020/PI_01/tree/main/datasets

The workflow consists of: 
- ETL perform (https://github.com/Arevalojj2020/PI_01/blob/main/Processing/ETL.ipynb)
    + Clean and prepare the data, finally save the clean dataset for the API building
- API functions (https://github.com/Arevalojj2020/PI_01/blob/main/Processing/API_Functions.ipynb)
    + Build the structure of the functions that will be implemented in the API, using the clean dataset
        + The functions *peliculas_mes(mes)* and *peliculas_dia(dia)* are case insensitive
        + In the functions *franquicia(franquicia)*, *peliculas_pais(pais)*, *productoras(productora)* and *retorno(pelicula)*, the                   parameter should be entered exactly as it appears in the dataset to obtain the expected result
- Realize an exploratory data analysis (EDA)
- Implementing a recommendation system
- Deploying API
