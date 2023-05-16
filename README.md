# Machine Learning Operations Project

In this project, we will analyze the lifecycle of a movie dataset from its data transformation to training a machine learning model that can take a movie title as input and return a list recommending 5 movies similar to it.

#### Datasets that will be used: https://github.com/Arevalojj2020/PI_01/tree/main/datasets

The workflow consists of: 
- ETL perform (https://github.com/Arevalojj2020/PI_01/blob/main/Processing/ETL.ipynb)
    + Clean and prepare the data, finally save the clean dataset for the API functions building
- API functions (https://github.com/Arevalojj2020/PI_01/blob/main/Processing/API_Functions.ipynb)
    + Build the structure of the functions that will be implemented in the API, using the clean dataset
        + The functions *peliculas_mes(mes)* and *peliculas_dia(dia)* are case insensitive
        + In the functions *franquicia(franquicia)*, *peliculas_pais(pais)*, *productoras(productora)* and *retorno(pelicula)*, the                   parameter should be entered exactly as it appears in the dataset to obtain the expected result
- Exploratory data analysis (EDA) (https://github.com/Arevalojj2020/PI_01/blob/main/Processing/EDA.ipynb)
    + Gather information and draw conclusions from the available attributes. The dataset used is the complete one, not the cleaned one,         to have much more information to analyze
- Recommendation system (https://github.com/Arevalojj2020/PI_01/blob/main/Processing/Recommendation_System.ipynb)
    + After performing the EDA, we discovered that we have more data (+45,000 movies) than the computer used on this project can process.       Therefore, to calculate a similarity score between movies and group them together, it was decided to consider the movie overview,         turning it into a natural language processing problem. 
      The solution we will implement is as follows:
        + Knowing that there are repeated movie titles, we will only use one of them
        + Vectorize the overviews and remove stop words to determine the frequency of each word's usage
        + Build a matrix with the results
        + Create a word map of the attribute
        + Calculate the relationship between overviews using cosine similarity scores
        + Create a movie title map                                                  
        + Finally, build a function that takes a movie title as input and returns a list of five similar movies
    + Due to limited resources, we will only be able to use approximately 10% of the dataset. However, we will not randomly select             movies. Instead, we will order the movies based on a rating ranking derived from reviews. To do this, we will use the average             vote of the movie (vote_average). However, there is a possibility that a movie with a high average vote has very few reviews,             which would not be balanced with another movie with the same average vote but a higher number of reviews. Therefore, we will use         the values of the vote_count attribute to calculate a "weighted_rating" in order to maintain a balanced rating ranking. 
      To calculate this new attribute, we will use an equation that includes the following variables:
         + v = number of votes per movie (vote_count)
         + R = average vote per movie (vote_average)
         + m = minimum number of votes to be classified in the ranking (this variable will determine the percentage of movies we will                keep)
         + c = average of all vote averages in the dataset
      The equation is as follows:
      $$(v/(v+m) * R) + (m/(m+v) * c)$$
- Deploying API
