#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:53:52 2026

@author: sahy

Creo que estamuy bien ee23 enero!
"""

# =============================================================================
# LIBRERIAS
# =============================================================================

import pandas as pd
# from pathlib import Path
# import pyreadr
# import gudhi as gd
# import numpy as np
from joblib import dump#, load #PAra guardar archivos RDS muy grandes

#Script mios
import cfg as cfg 

# =============================================================================
# CREACIÓN DE DATAFRAMES
# =============================================================================
def todos_los_experimentos(data_dir,result_dir):
    """
    Lee los archivos de todos los experimentos contenidos en data_dir,
    construye un DataFrame por experimento y almacena la información
    procesada en una estructura unificada.

    Parameters
    ----------
    data_dir : pathlib.Path
        Ruta donde se encuentran los datos de los experimentos.
    result_dir : pathlib.Path
        Ruta donde se guardará el DataFrame final en formato .csv.

    Returns
    -------
    list of dict
        Lista de diccionarios con las claves:
        - label: etiqueta del experimento
        - experimento: identificador del experimento
        - df: DataFrame procesado
    """
    datos = []
    
    for carpeta in sorted(data_dir.iterdir()):
        if not carpeta.is_dir() or carpeta.name.startswith("."):
            continue
    
        etiqueta = carpeta.name
        
        # Recorremos los experimentos dentro de cada etiqueta
        for carpeta2 in sorted(carpeta.iterdir()):
            if not carpeta2.is_dir() or carpeta2.name.startswith("."):
                continue
    
            experimento_id = carpeta2.name
            print(f"Experimento: {experimento_id}")
    
            dfs_experimento = [] # DataFrames de cada individuo del experimento
            
            
            for i, archivo in enumerate(sorted(carpeta2.glob("*.dat")), start=1):
    
                try:
                    df = pd.read_csv(
                        archivo,
                        sep=r"\s+",
                        header=None,
                        names=cfg.COLUMN_NAMES
                    )
    
                    # columnas derivadas
                    df[cfg.ID_PART_COL] = f"{i:03d}"
                    df[cfg.TIEMPO_COL] = df[cfg.FRAME_COL] / cfg.FPS
    
                    dfs_experimento.append(df)
                except Exception as e:
                    print(f"Error leyendo {archivo}: {e}")
                
            #Concatenamos  TODOS los individuos del experimento
            if dfs_experimento:
                df_experimento = pd.concat(dfs_experimento, ignore_index=True)
            
                # Filtrado de tiempos inicial y final
                df_exp_filtrado = quitar_tiempo_inicial(df_experimento)
                df_exp_filtrado = quitar_tiempo_final(df_exp_filtrado)
    
                
                datos.append({
                    cfg.LABEL_COL: etiqueta,
                    cfg.ID_COL : experimento_id,
                    "df": df_exp_filtrado
                })
    
    # Unimos todos los experimentos en un único DataFrame  
    datos_df_total = pd.concat(
        [
            d["df"].assign(
                label=d[cfg.LABEL_COL],
                experimento=d[cfg.ID_COL]
            )
            for d in datos
        ],
        ignore_index=True
    )
    
    #Guardando el df de todos los experimentos
    nombre_archivo_df_total = cfg.NAME_ALL_EXPERIMENTOS 
    path_resultados = result_dir/nombre_archivo_df_total
    dump(datos_df_total,path_resultados)
    print(f"Ruta del df de todos los experimentos guardados en : {path_resultados}")
    
    return datos_df_total

# =============================================================================
# FUNCIONES PARA LA LIMPIEZA TEMPORAL
# =============================================================================

def quitar_tiempo_inicial(df):
    """
    Elimina el tramo inicial del experimento en el que todavía no están
    presentes todos los individuos considerados inicialmente.

    La función identifica a los individuos que aparecen en el primer instante
    dentro de la zona definida como 'puerta' y elimina los tiempos previos
    al momento en que alguno de esos individuos desaparece, marcando así
    el inicio efectivo del experimento.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con los datos completos del experimento.

    Returns
    -------
    pandas.DataFrame
        DataFrame filtrado a partir del tiempo inicial válido.
    """
    # Filtramos los individuos que se encuentran en la zona de la puerta
    # y ordenamos por tiempo
    df_puerta = individuos_cuadrado_df(df, longX=cfg.XLONG_PUERTA, longY=cfg.YLONG_PUERTA).sort_values(cfg.TIEMPO_COL)
   
    # Primer instante temporal registrado en la puerta
    primer_valor = df_puerta[cfg.TIEMPO_COL].iloc[0]
    
    # Identificamos los individuos presentes en ese primer instante
    individuos_iniciales = df_puerta[df_puerta[cfg.TIEMPO_COL] == primer_valor][cfg.ID_PART_COL].tolist()
    
    # Si no hay individuos detectados, devolvemos el DataFrame original
    if not individuos_iniciales:
        return df.copy()
    
    # Inicializamos el tiempo de inicio
    t_inicio = primer_valor
    
    # Recorremos los tiempos hasta encontrar el instante
    # en el que falta al menos uno de los individuos iniciales
    for t in sorted(df_puerta[cfg.TIEMPO_COL].unique()):
        presentes = df_puerta[(df_puerta[cfg.TIEMPO_COL] == t) & (df_puerta[cfg.ID_PART_COL].isin(individuos_iniciales))]
        if len(presentes) < len(individuos_iniciales):
            t_inicio = t
            break
        
    # Devolvemos los datos a partir del tiempo de inicio válido    
    return df[df[cfg.TIEMPO_COL] >= t_inicio].copy()



def quitar_tiempo_final(df, umbral: float = 0.2):
    """
   Elimina el tramo final del experimento cuando el número de individuos
   presentes cae por debajo de un umbral relativo al máximo observado.

   Parameters
   ----------
   df : pandas.DataFrame
       DataFrame con los datos del experimento.
   umbral : float, optional
       Fracción mínima del número máximo de individuos para considerar
       un tiempo como válido (por defecto 0.2).

   Returns
   -------
   pandas.DataFrame
       DataFrame filtrado, excluyendo los tiempos finales no representativos.
   """
    # Contamos el número de individuos presentes en cada instante de tiempo
    conteo = df.groupby(cfg.TIEMPO_COL).size().reset_index(name=cfg.ID_PART_COL)
    
    # Número máximo de individuos observados en el experimento
    max_p = conteo[cfg.ID_PART_COL].max()
    
    # Seleccionamos los tiempos donde el número de individuos supera el umbral definido
    tiempos_validos = conteo[conteo[cfg.ID_PART_COL] > umbral * max_p][cfg.TIEMPO_COL]
    
    # Filtramos el DataFrame original usando los tiempos válidos
    return df[df[cfg.TIEMPO_COL].isin(tiempos_validos)].copy()


# =============================================================================
# FILTROS ESPACIALES
# =============================================================================

def individuos_cuadrado_df(df, longX = None, longY = None):
    """
   Filtra los individuos que se encuentran dentro de una región cuadrada
   centrada en el origen del sistema de coordenadas.

   La región está definida por los límites:
   - |x| <= longX
   - y <= longY

   Si no se especifican los valores de longX o longY, se utilizan los
   valores definidos en la configuración global (cfg).

   Parameters
   ----------
   df : pandas.DataFrame
       DataFrame con las posiciones de los individuos.
   longX : float, optional
       Semi-longitud del cuadrado en el eje X.
   longY : float, optional
       Límite del cuadrado en el eje Y.

   Returns
   -------
   pandas.DataFrame
       DataFrame filtrado con los individuos dentro de la región cuadrada.
   """

    # Valores por defecto desde la configuración si no se especifican    
    if longX is None:
        longX = cfg.XLONG
    if longY is None:
        longY = cfg.YLONG
    
    # Filtrado espacial de los individuos dentro del cuadrado
    return df[(df[cfg.X_COL].abs() <= longX) & (df[cfg.Y_COL] <= longY)].copy()


def datos_consecutivos(data, n_t):
    """
    Genera subconjuntos de un DataFrame usando ventanas de frames consecutivos.

    Parámetros
    ----------
    data : pandas.DataFrame
        DataFrame de entrada que debe contener una columna identificadora
        de frame (cfg.FRAME_COL).

    n_t : int
        Número de frames consecutivos que se quieren agrupar en cada
        subconjunto.

    Retorna
    -------
    lista_df : list of pandas.DataFrame
        Lista de DataFrames.  
        Cada elemento de la lista contiene todas las filas de `data`
        cuyos valores en la columna cfg.FRAME_COL pertenecen a un grupo
        de `n_t` frames consecutivos.
    """

    # Lista donde se almacenarán los DataFrames generados
    lista_df = []

    # Se ordena el DataFrame por la columna de frames
    data = data.sort_values(cfg.FRAME_COL)

    # Se obtienen los valores únicos de frame, ya ordenados
    valores_unicos = data[cfg.FRAME_COL].unique()

    # Se crean ventanas consecutivas de tamaño n_t sobre los frames
    for i in range(len(valores_unicos) - n_t + 1):

        # Frames seleccionados para esta ventana
        fotogramas_sel = valores_unicos[i:i + n_t]

        # Se filtran las filas del DataFrame que pertenecen a esos frames
        data_red = data[data[cfg.FRAME_COL].isin(fotogramas_sel)]

        # Se añade el subconjunto a la lista de salida
        lista_df.append(data_red)

    # Se devuelve la lista con todos los subconjuntos
    return lista_df
    


# #----------------------------------------------------------------------
# #Para saber en que intervalo se encuentran los datos con los que estamos trabajando
# import matplotlib.pyplot as plt

# # Graficar todas las coordenadas X, Y
# plt.figure(figsize=(8, 6))
# plt.scatter(
#     datos_df_total["x"], 
#     datos_df_total["y"], 
#     s=1,            # tamaño de los puntos pequeño por la cantidad de datos
#     alpha=0.5       # transparencia para ver densidad
# )
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Posiciones de los individuos en el plano")
# plt.axis("equal")   # para que la escala X e Y sea igual
# plt.show()








