#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:09:52 2026

@author: sahy

Creamos una funcion para annadir un pequenno ruido a nuestros datos 
"""

#------------------------------------------------------------------------------
#Librerias

from pathlib import Path
import os
import pandas as pd
import pyreadr
import gudhi as gd
import numpy as np
from joblib import dump, load #PAra guardar archivos RDS muy grandes

#Script mios
import cfg as cfg
from organizacion_data import *
from crocker_plot_betti import *
from procesado import *

# =============================================================================
# FUNCIONES
# =============================================================================

def add_fixed_distance_noise(pos,error = None, seed=None):
    """
    Añade desplazamiento aleatorio de magnitud fija a cada posición y dibuja el resultado.
    
    pos: array (N,2) con posiciones [x,y]
    error: porcentaje relativo (ej. 0.02 = 2%)
    seed: semilla opcional para reproducibilidad
    
    Retorna: pos_ruido, array de posiciones desplazadas
    """
    if seed is not None:
        np.random.seed(seed)
    if error is None:
        error = cfg.ERROR_ADD_RUIDO
 
    N = pos.shape[0]
    
    r_original = np.sqrt(pos['x']**2 + y**2)

    # Dirección aleatoria uniforme
    theta = np.random.uniform(0.0, 2*np.pi, size=N)
    
    
    # Generar r con distribución uniforme
    r_max = 0.2 * (1 + error)
    r = np.random.uniform(0, r_max, size=N)
    
    


    

    # Desplazamiento máximo por punto
    delta_r_max = np.minimum(error_relativo * r_original, max_desplazamiento)

    # Generar desplazamientos aleatorios uniformes en el círculo
    theta = np.random.uniform(0, 2*np.pi, N)
    u = np.random.uniform(0, 1, N)
    r_error = delta_r_max * np.sqrt(u)
    











    
    # Desplazamiento en coordenadas cartesianas
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)

    pos_ruido = pos + np.column_stack((dx, dy))
    
    
    # # --- Dibujo ---
    # plt.figure(figsize=(6, 6))
    # plt.scatter(pos[:, 0], pos[:, 1], s=30, label="original")
    # plt.scatter(pos_ruido[:, 0], pos_ruido[:, 1], s=30, label="con ruido")

    # # Flechas mostrando el desplazamiento
    # plt.quiver(
    #     pos[:, 0], pos[:, 1], 
    #     dx, dy, 
    #     angles='xy', scale_units='xy', scale=1, width=0.005, color='gray', alpha=0.6
    # )

    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.axis("equal")
    # plt.legend()
    # plt.title(f"Desplazamiento fijo de {distance} unidades")
    # plt.show()
    
    

    return pos_ruido




# =============================================================================
# RUTAS Y PARÁMETROS
# =============================================================================

#Detectamos si estamos en un .py o en un notebook. Para poder quedarnos con la ruta.
try:
    root_dir = Path(__file__).resolve().parent
except NameError:
    root_dir = Path().resolve() 

data_dir = root_dir.parent / cfg.FOLDER_DATA_NAME #ruta donde se encuentan los datos 
result_dir = root_dir.parent / cfg.FOLDER_RESULTADOS_NAME_RUIDO # ruta donde guarderemos todos los resultados obtenidos 

#------------------------------------------------------------------------------
#Leemos los datos guardados
ruta_salida = result_dir/ cfg.NAME_ALL_EXPERIMENTOS
df_todos_experimentos = load(ruta_salida)
#------------------------------------------------------------------------------
#Nos quedamos con las columnas de las posiciones de los peatones 
#Y a continuacion, calculamos el ruido y lo reemplazamos por las columnas originales


pos = df_todos_experimentos[[cfg.X_COL, cfg.Y_COL]].to_numpy()

pos_ruido = add_fixed_distance_noise(pos)

df_todos_experimentosruido = df_todos_experimentos

df_todos_experimentosruido[[cfg.X_COL, cfg.Y_COL]] = pos_ruido

#-----------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Recortamos los datos a un cuadrado 2x2
data_archivos_total_delimitado = individuos_cuadrado_df(df_todos_experimentosruido)
#------------------------------------------------------------------------------

# Recorremos cada etiqueta (ej. "empujando", "sin_empujar")
for etiqueta, data_archivos in data_archivos_total_delimitado.groupby(cfg.LABEL_COL):

    print(f"\nProcesando etiqueta: {etiqueta}")

    # Carpeta específica para guardar resultados esta etiqueta 
    rutas_base = result_dir / etiqueta
    os.makedirs(rutas_base, exist_ok=True)

    # IDs de evacuación para esta etiqueta  
    ids_evacuaciones = data_archivos[cfg.ID_COL].unique()
    
    # Llamada a la función genial
    procesar_evacuaciones_genial(
        data_archivos=data_archivos,
        ids=ids_evacuaciones,
        etiqueta=etiqueta,
        ruta_salida=rutas_base,
        n_t=cfg.NUM_FRAMES,        # tamaño de ventana, puedes ajustar
        salto=cfg.SALTO,           # salto entre frames
        curve_type=cfg.CURVA_DE_PERSISTENCIA,  # Cambiar por "betti", "entropy" o "euler" si quieres
        dim=cfg.DIMENSION,                   # Solo si usas "betti" o "entropy"
        mymaxscale=cfg.MYMAXSCALE,
        numepsilon=cfg.NUMEPSILON
    )



