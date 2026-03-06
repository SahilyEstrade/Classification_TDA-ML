#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline ML para clasificación de evacuaciones usando matrices Betti
Optimizado para:
- menos código repetido
- más métricas
- guardar resultados automáticamente



Version del script crockerpltos_4seg.py pero mia
"""

import numpy as np
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import mode

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from pathlib import Path

import cfg as cfg

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


curva_persistencia = 'betti_1'
ruta_datos_empujando =  Path('/home/sahily/Documentos/Classification_TDA-ML/RESULTADOS/empujando/')
ruta_datos_sinempujar =  Path('/home/sahily/Documentos/Classification_TDA-ML/RESULTADOS/sinempujar/')

#empujando = 0
#sin empujar = 1
ruta_0 = ruta_datos_empujando / curva_persistencia
ruta_1 =ruta_datos_sinempujar / curva_persistencia


# -----------------------------------------------------------------------------
# FUNCIONES
# -----------------------------------------------------------------------------

def descargar_matrices_por_id(ruta_carpeta):

    id_dict = {}

    for archivo in os.listdir(ruta_carpeta):

        if archivo.endswith('.csv'):

            partes = archivo.split('_')
            id_str = partes[3]

            if id_str not in id_dict:
                id_dict[id_str] = []

            ruta_archivo = os.path.join(ruta_carpeta, archivo)

            df = pd.read_csv(ruta_archivo, skiprows=1, header=None)
            id_dict[id_str].append(df.values)

    ids_ordenados = sorted(id_dict.keys())

    lista_de_listas = [id_dict[k] for k in ids_ordenados]

    return lista_de_listas


def convertir_lista_array(lista, idx):

    train = [lista[i] for i in idx]

    lista_train = list(itertools.chain.from_iterable(train))

    array_train = np.array(lista_train)

    return array_train


def calcular_metricas(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }


# -----------------------------------------------------------------------------
# CARGA DE DATOS
# -----------------------------------------------------------------------------

lista_0 = descargar_matrices_por_id(ruta_0)
lista_1 = descargar_matrices_por_id(ruta_1)

# separación manual
idx_train_0 = [2,0]
idx_test_0  = [1]

idx_train_1 = [2,0]
idx_test_1  = [1]

array_train_0 = convertir_lista_array(lista_0,idx_train_0)
array_train_1 = convertir_lista_array(lista_1,idx_train_1)

array_test_0  = convertir_lista_array(lista_0,idx_test_0)
array_test_1  = convertir_lista_array(lista_1,idx_test_1)


# -----------------------------------------------------------------------------
# BALANCEO
# -----------------------------------------------------------------------------

n_train = len(array_train_1)
n_test  = len(array_test_1)

idx_train_0 = np.random.choice(array_train_0.shape[0], size=n_train, replace=False)
idx_test_0  = np.random.choice(array_test_0.shape[0], size=n_test, replace=False)

datos_train_0 = array_train_0[idx_train_0]
datos_train_1 = array_train_1

datos_test_0 = array_test_0[idx_test_0]
datos_test_1 = array_test_1


# -----------------------------------------------------------------------------
# VECTORIZE
# -----------------------------------------------------------------------------

matriz_train_0 = datos_train_0.reshape((n_train,-1))
matriz_train_1 = datos_train_1.reshape((n_train,-1))

matriz_test_0  = datos_test_0.reshape((n_test,-1))
matriz_test_1  = datos_test_1.reshape((n_test,-1))


labels_train_0 = np.zeros(n_train)
labels_train_1 = np.ones(n_train)

labels_test_0  = np.zeros(n_test)
labels_test_1  = np.ones(n_test)


X_train = np.concatenate((matriz_train_0, matriz_train_1))
y_train = np.concatenate((labels_train_0, labels_train_1))

X_test  = np.concatenate((matriz_test_0, matriz_test_1))
y_test  = np.concatenate((labels_test_0, labels_test_1))


# mezclar
X_train, y_train = shuffle(X_train, y_train, random_state=RANDOM_STATE)
X_test, y_test   = shuffle(X_test, y_test, random_state=RANDOM_STATE)


# -----------------------------------------------------------------------------
# NORMALIZACIÓN
# -----------------------------------------------------------------------------

scaler = StandardScaler()

X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)


# -----------------------------------------------------------------------------
# PCA
# -----------------------------------------------------------------------------

pca_model = PCA(n_components=0.95, random_state=RANDOM_STATE)

X_train_pca = pca_model.fit_transform(X_train_norm)
X_test_pca  = pca_model.transform(X_test_norm)


# -----------------------------------------------------------------------------
# MODELOS
# -----------------------------------------------------------------------------

modelos = {

    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),

    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),

    "KNN": KNeighborsClassifier(n_neighbors=3),

    "LDA": LinearDiscriminantAnalysis()
}

datasets = {

    "Normalizado": (X_train_norm, X_test_norm),

    "PCA": (X_train_pca, X_test_pca)
}


resultados = []


for nombre_modelo, modelo in modelos.items():

    for tipo_datos, (Xtr, Xte) in datasets.items():

        modelo.fit(Xtr, y_train)

        y_pred = modelo.predict(Xte)

        metricas = calcular_metricas(y_test, y_pred)

        resultados.append({
            "modelo": nombre_modelo,
            "datos": tipo_datos,
            **metricas
        })


# -----------------------------------------------------------------------------
# K-MEANS (NO SUPERVISADO)
# -----------------------------------------------------------------------------

kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=20)

kmeans.fit(X_train_pca)

y_train_cluster = kmeans.predict(X_train_pca)
y_test_cluster  = kmeans.predict(X_test_pca)


def obtener_mapeo(clusters, etiquetas):

    mapeo = {}

    for c in np.unique(clusters):

        mask = clusters == c

        mapeo[c] = mode(etiquetas[mask], keepdims=True).mode[0]

    return mapeo


mapeo = obtener_mapeo(y_train_cluster, y_train)

y_pred_ajustado = np.array([mapeo[c] for c in y_test_cluster])

metricas_kmeans = calcular_metricas(y_test, y_pred_ajustado)

resultados.append({
    "modelo": "KMeans",
    "datos": "PCA",
    **metricas_kmeans
})


# -----------------------------------------------------------------------------
# RESULTADOS
# -----------------------------------------------------------------------------

df_resultados = pd.DataFrame(resultados)

print("\nRESULTADOS:")
print(df_resultados)


# -----------------------------------------------------------------------------
# GUARDAR CSV
# -----------------------------------------------------------------------------

ruta_resultados = Path('/home/sahily/Documentos/Classification_TDA-ML/RESULTADOS/')
# Crear el nombre de archivo dinámicamente
archivo_salida = ruta_resultados / f'resultados_modelos_{curva_persistencia}_ruido_{cfg.ERROR_ADD_RUIDO}.csv'

df_resultados.to_csv(archivo_salida, index=False)

print("\nResultados guardados en:", archivo_salida)