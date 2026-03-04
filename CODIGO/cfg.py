#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 10:56:10 2026

@author: sahy
"""
#------------------------------------------------------------------------------
#--------------NOMBRE DE CARPETAS----------------------------------------------

#Nombre de la carpeta donde estan los datos 
FOLDER_DATA_NAME = "DATA" 
#Nombre de la carpeta donde vamos a guardar los resultados
FOLDER_RESULTADOS_NAME = "RESULTADOS" 

FOLDER_RESULTADOS_NAME_RUIDO = "RESULTADOS_RUIDO"
#------------------------------------------------------------------------------
#------------NOMBRE DE LAS COLUMNAS PARA CREAR EL DATA FRAME INICIAL-----------
FRAME_COL = "frames"
X_COL = "x"
Y_COL = "y"
LABEL_COL = "label" #aqui estaran guardadas las etiquetas 
ID_COL = "experimento" #aque tenemos guardado que 
TIEMPO_COL = "tiempo"
ID_PART_COL = "id_part" #representa el numero del peaton en estudio

COLUMN_NAMES = [FRAME_COL,X_COL,Y_COL]

#------------------------------------------------------------------------------
#----------NOMBRE DE ARCHIVOS -------------------------------------------------
#Nombre del archivo en el que guardamos el df de todos los experimentos
NAME_ALL_EXPERIMENTOS = "todos_df_experimentos.rds" 

#------------------------------------------------------------------------------
#--------------VARIABLES-------------------------------------------------------
NUMEPSILON = 50  #Valor max de num epsilon para calcular los num de betti 

MYMAXSCALE = 0.35  #Escala máxima para el parámetro de la filtración
MYMAXDIMENSION = 2 #Escala maxima de la homologia 

# Definimos los limites de las puertas 
XLONG_PUERTA = 1
YLONG_PUERTA = 0.5

#Definimos el marco de estudio 
XLONG = 2
YLONG = 2


CANT_EXPERIMENTOS = 3 #seleccionamos la cantidad de experimentos con los que trabajaremos 
SALTO =  5

#Seleccionamos la cantidad de frames que represnetan los segundos que queremos trabajar!
SEGUNDOS = 4
FPS = 25 #Valor por el que sacamos el tiempo de los dataframe


NUM_FRAMES = SEGUNDOS * FPS #(100 frames=4 segundos) PARA TRABAJAR CON 1 SOLO FRMAE TENGO QUE PONER 0,04 SEGUNDO

L = 2.5 #Longitud de los intervalos temporales

#curve_type : betti |entropy | euler
CURVA_DE_PERSISTENCIA ='betti'

DIMENSION = 1
#------------------------------------------------------------------------------
#---------------------------------------------------------------------

ERROR_ADD_RUIDO =0.02


