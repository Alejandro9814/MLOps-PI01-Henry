from fastapi import FastAPI
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



app = FastAPI()



@app.get("/")
def read_root():
    return {"message": "FastAPI funcionando en Render 🚀"}



@app.get("/cantidad_estrenos_mes")
def cantidad_estrenos_mes(mes: str):
    """
    Calcula la cantidad de películas estrenadas en un mes específico.

    Args:
        mes (str): Nombre del mes en español (ejemplo: 'enero', 'febrero').

    Returns:
        Mensaje con la cantidad de películas estrenadas en el mes consultado.
    """
    df_movies = pd.read_parquet('DataSet/Movies.parquet')
    df_movies['release_date'] = pd.to_datetime(df_movies['release_date'], unit='ms')

    meses_dict = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, 
            "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}
    
    mes_numero = meses_dict.get(mes.lower())
    if mes_numero is None:
        return {"error": f"'{mes}' no es un mes válido. Usa un mes en español, como 'enero'."}

    cantidad_xmes = df_movies.loc[df_movies['release_date'].dt.month == mes_numero, "release_date"].count()
    
    return {f"{cantidad_xmes} películas fueron estrenadas en {mes}."}


@app.get("/cantidad_filmaciones_dia")
def cantidad_filmaciones_dia(dia: str):
    """
    Calcula la cantidad de películas estrenadas en un día específico de la semana.

    Args:
        dia (str): Día de la semana en español (ejemplo: 'lunes', 'martes').

    Returns:
        Mensaje con la cantidad de películas estrenadas en el día consultado.
    """
    df_movies = pd.read_parquet('DataSet/Movies.parquet')
    df_movies['release_date'] = pd.to_datetime(df_movies['release_date'], unit='ms')

    dias_semana_dict = {"lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3,"viernes": 4,"sábado": 5, "domingo": 6}
    
    dia_numero = dias_semana_dict.get(dia.lower())
    if dia_numero is None:
        return {"error": f"'{dia}' no es un día válido. Usa un día en español, como 'lunes'."}

    cantidad_xdia = df_movies.loc[df_movies['release_date'].dt.dayofweek == dia_numero, "release_date"].count()

    return {f"{cantidad_xdia} películas fueron estrenadas un {dia}."}



@app.get("/score_titulo")
def score_titulo(titulo_de_la_filmación: str):
    """
    Obtiene el año de estreno y la popularidad de una película según su título.

    Args:
        titulo_de_la_filmación (str): Título de la película a buscar.

    Returns:
        dict: Un diccionario con un mensaje sobre el año de estreno, la popularidad y titulo 
        de la película, o un mensaje de error si no se encuentra.
    """
    df_movies = pd.read_parquet('DataSet/Movies.parquet')

    #Se asigna a una variable un pequeño df de una sola fila según la coincidencia con el título de la película
    pelicula = df_movies[df_movies['title'].str.lower() == titulo_de_la_filmación.lower()] 
    if pelicula.empty:
        return {"error": "No se encontró ninguna película con ese título."}
    
    #Se obtienen los datos contenidos en la fila de la variable película y los asigna a una variable por separado.
    titulo = pelicula['title'].iloc[0]
    año = pelicula['release_date'].dt.year.iloc[0]
    score = pelicula['popularity'].iloc[0]

    return {f"La película '{titulo}' fue estrenada en el año {año} con una popularidad de {score}"}



@app.get("/votos_titulo")
def votos_titulo(titulo_de_la_filmación: str):
    """
    Obtiene años de estreno, la cantidad de votos y el promedio de valoraciones de una película, sólo si tiene más de 2000 valoraciones.

    Args:
        titulo_film (str): Título de la película a consultar.

    Returns:
        dict: 
            - Si la película tiene al menos 2000 valoraciones, devuelve el título, año de estreno, 
              cantidad de votos y promedio de valoraciones.
            - Si la película tiene menos de 2000 votos, indica que no cumple con el requisito.
            - Si el título no se encuentra en la base de datos, devuelve un mensaje de error.
    """
    df_movies = pd.read_parquet('DataSet/Movies.parquet')

    #Se asigna a una variable un pequeño df de una sola fila según la coincidencia con el título de la película
    pelicula = df_movies[df_movies['title'].str.lower() == titulo_de_la_filmación.lower()]
    
    if pelicula.empty:
        return {"error": "No se encontró ninguna película con ese título."}
    
    votos = pelicula['vote_count'].iloc[0]
    if votos < 2000:
        return {f"La película '{titulo_de_la_filmación}' no cumple con la condición de tener al menos 2000 valoraciones. Solo tiene {votos} valoraciones."}
    
    # Se obtienen los datos contenidos en la fila de la variable película y los asigna a una variable por separado.
    promedio_votos = pelicula['vote_average'].iloc[0]
    titulo = pelicula['title'].iloc[0]
    año = pelicula['release_date'].dt.year.iloc[0]
    
    return {f"La película '{titulo}' fue estrenada en el año {año}. La misma cuenta con un total de {votos} valoraciones, con un promedio de {promedio_votos}."}



@app.get("/get_actor")
def get_actor(nombre_actor: str):
    """
    Obtiene las películas la cantidad de películas en las que ha participado un actor, el retorno total de las películas y el promedio.

    Args:
        nombre_actor (str): Nombre actor.

    Returns:
        Mensaje con las peliculas en las que participó el actor, el retorno que ha conseguido por película y un promedio de retorno,
        o error si no encuentra al actor consultado.
    """
    df_movies = pd.read_parquet("DataSet/Movies.parquet")
    df_cast_movies = pd.read_parquet("DataSet/Relation_Cast_Movies.parquet")
    df_cast = pd.read_parquet("DataSet/Cast_Movies.parquet")

    df_actor_movies = pd.merge(df_cast_movies, df_cast, left_on="idCast", right_on="idCast", how="inner")
    df_actor_movies = pd.merge(df_actor_movies, df_movies, left_on="idMovies", right_on="idMovies", how="inner")

    #Se asigna a una variable un pequeño df con las filas donde aparece el actor buscado.
    actor = df_actor_movies[df_actor_movies['nameCast'].str.lower() == nombre_actor.lower()]
    
    if actor.empty:
        return {"error": f"No se encontró ningún actor con el nombre {nombre_actor}"}
    
    cantidad_peliculas = len(actor)
    
    # Sumar la tasa de retorno (return) y calcular el promedio
    total_retorno = actor['return'].sum()  
    promedio_retorno = actor['return'].mean() if cantidad_peliculas > 0 else 0
    
    return {f"El actor {nombre_actor} ha participado de {cantidad_peliculas} filmaciones, el mismo ha conseguido un retorno de {total_retorno} con un promedio de {promedio_retorno} por filmación."}



@app.get("/get_director")
def get_director(nombre_director: str):
    """
    Obtiene las películas, el retorno total de las películas y el promedio.

    Args:
        nombre_actor (str): Nombre actor.

    Returns:
        Mensaje con las peliculas en las que participó el actor, el retorno que ha conseguido por película y un promedio de retorno.
    """
    df_movies_director = pd.read_parquet("DataSet/Relation_Director_Movies.parquet")
    df_directors = pd.read_parquet("DataSet/Director_Movies.parquet")
    df_movies = pd.read_parquet("DataSet/Movies.parquet")

    #Se asigna a una variable un pequeño df con las filas donde aparece el director buscado.
    director = df_directors[df_directors['nameCrew'].str.lower() == nombre_director.lower()]
    
    if director.empty:
        return {"error": f"No se encontró ningún director con el nombre {nombre_director}"}
    
    id_director = director['idCrew'].iloc[0]
    
    # Obtiene las películas del director usando la tabla intermedia 'movie_director'
    director_peliculas = df_movies_director[df_movies_director['idCrew'] == id_director]
    
    peliculas_con_detalles = pd.merge(director_peliculas, df_movies, left_on='idMovies', right_on='idMovies', how='left')
    
    """Recorre el DataFrame para extraer información relevante de cada película, 
    incluyendo título, fecha de lanzamiento, presupuesto, ingresos y retorno de inversión. 
    Luego, almacena estos datos en una lista de diccionarios."""
    peliculas_detalle = []
    total_retorno = 0
    for _, row in peliculas_con_detalles.iterrows():
        nombre_pelicula = row['title']
        fecha_lanzamiento = row['release_date'].strftime('%Y-%m-%d')
        costo = row['budget']
        ganancia = row['revenue']
        retorno_individual = row['return']
        
        peliculas_detalle.append({
            "titulo": nombre_pelicula,
            "fecha_lanzamiento": fecha_lanzamiento,
            "retorno_individual": retorno_individual,
            "costo": costo,
            "ganancia": ganancia
        })
        
        # Calcular el retorno total
        total_retorno += retorno_individual
    
    return {
        "mensaje": f"El director {nombre_director} ha dirigido {len(director_peliculas)} filmaciones, el mismo ha conseguido un retorno total de {total_retorno}.",
        "detalles_peliculas": peliculas_detalle
    }



@app.get("/get_recomendacion")
def recomendar_peliculas(titulo):
    """
    Recomienda películas similares basadas en el título dado.

    Args:
        titulo (str): Título de la película de referencia.

    Returns:
        list: Lista de títulos de películas recomendadas o error si no encuentra el título en la tabla.
    """
    n_recomendaciones= 5
    titulo = titulo.lower()

    df_to_ML = pd.read_parquet("DataSet/Data-to-ML.parquet")

    with open('DataSet/cosine_similarity_matrix.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)

    if titulo not in df_to_ML["title"].str.lower().values:
        return "Título no encontrado en la base de datos."

    # Obtiene el índice de la película
    idx = df_to_ML.loc[df_to_ML["title"].str.lower() == titulo].index[0]

    # Calcula la similitud del coseno con todas las películas
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordena por similitud en orden descendente
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtiene los índices de las películas más similares
    sim_indices = [i[0] for i in sim_scores[1:n_recomendaciones+1]]

    lista_recomendacion = df_to_ML.iloc[sim_indices]["title"].tolist()
    return lista_recomendacion
