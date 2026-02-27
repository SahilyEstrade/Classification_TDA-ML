#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 10:02:31 2026

@author: sahy

Organizamos todos los calculos 
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
#------------------------------------------------------------------------------


# =============================================================================
# RUTAS Y PARÁMETROS
# =============================================================================

#Detectamos si estamos en un .py o en un notebook. Para poder quedarnos con la ruta.
try:
    root_dir = Path(__file__).resolve().parent
except NameError:
    root_dir = Path().resolve() 

data_dir = root_dir.parent / cfg.FOLDER_DATA_NAME #ruta donde se encuentan los datos 
result_dir = root_dir.parent / cfg.FOLDER_RESULTADOS_NAME # ruta donde guarderemos todos los resultados obtenidos 

#------------------------------------------------------------------------------
#Esta funcion une todos los exprimentos pero ademas los guarda!
todos_df = todos_los_experimentos(data_dir,result_dir)

#Leemos los datos guardados
ruta_salida = result_dir/ cfg.NAME_ALL_EXPERIMENTOS
df_todos_experimentos = load(ruta_salida)

#------------------------------------------------------------------------------
#Recortamos los datos a un cuadrado 2x2
data_archivos_total_delimitado = individuos_cuadrado_df(df_todos_experimentos)

#------------------------------------------------------------------------------
#
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



#------------------------------------------------------------------------------




#-------------------Calculando Homologia --------------------------------------

# homology = calcular_homologia_bloque(data_experimento) #Hasta aca esta todo okok

# #-------------------Calculando CURVAS DE BETTI--------------------------------------
# griddata = turnIntervalsIntoGrid(homology)

# betti_0, betti_1, betti_euler, betti = crear_matrices_betti(griddata)

# #GEnermos graficas de las curvas de betti
# matrizBetti(griddata,0,maxYlim = cfg.MYMAXSCALE)
# matrizBetti(griddata,1,maxYlim = cfg.MYMAXSCALE)

# #GEneramos las graficas de los crocker plot 
# crockerplot_grafic(griddata,0,maxYlim = cfg.MYMAXSCALE)  
# crockerplot_grafic(griddata,1,maxYlim = cfg.MYMAXSCALE)

#-----------------------------------------------------------------------------
