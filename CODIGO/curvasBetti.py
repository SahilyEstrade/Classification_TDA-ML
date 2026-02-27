#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 11:10:16 2026

@author: Codigo Cristina 
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import itertools
#import umap


#leer archivos CSV de una carpeta y devolverlos en una lista como dataframes

def descargar_matrices(ruta_carpeta):

    lista = []
    for archivo in os.listdir(ruta_carpeta):
        if archivo.endswith('.csv'):  
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            df = pd.read_csv(ruta_archivo, skiprows=1, header=None)
            lista.append(df)

    return lista

# Descargar Matrices 
#------------------------------------------------------------------------------
#empujando = 0
#sin empujar = 1
ruta_0 = './be_empujando_frame'
ruta_1 = './be_sin_empujar_frame'

lista_0 = descargar_matrices(ruta_0)
lista_1 = descargar_matrices(ruta_1)


# Separar en train y test
#------------------------------------------------------------------------------

#separamos manualmente 
idx_train_0 = [2,0]
idx_test_0 = [1]

idx_train_1 = [0,1]
idx_test_1 = [2]

#FUNCIÓN:  convierte una lista de matrices en una única matriz donde las filas son los frames 
def convertir_lista_array(lista,idx):
    
    np.random.seed(42) 
    train =  [lista[i] for i in idx]             #nos quedamos con los indices que queremos de la lista
    matriz_train = np.hstack(train)             #unificamos columnas
    matriz_train = matriz_train.T               #hacemos las traspuesta para que el tiempo este en filas
    np.random.shuffle(matriz_train)             #mezclamos filas                 

    return matriz_train

array_train_0 = convertir_lista_array(lista_0,idx_train_0)
array_train_1 = convertir_lista_array(lista_1,idx_train_1)
array_test_0 = convertir_lista_array(lista_0,idx_test_0)
array_test_1 = convertir_lista_array(lista_1,idx_test_1)

print('train:',array_train_0.shape,array_train_1.shape)
print('test:',array_test_0.shape,array_test_1.shape)

#balanceamos:

n_train = len(array_train_1)  #numero de muestras
n_test = len(array_test_1) 

# Elegir el mismo num de matrices con índices aleatorios de la clase mayoritaria
idx_train_0 = np.random.choice(array_train_0.shape[0], size=n_train, replace=False)
idx_test_0 = np.random.choice(array_test_0.shape[0], size=n_test, replace=False)

# Subconjuntos balanceados
datos_train_0 = array_train_0[idx_train_0]
datos_train_1 = array_train_1

datos_test_0 = array_test_0[idx_test_0]
datos_test_1 = array_test_1

print('train:',datos_train_0.shape,datos_train_1.shape)
print('test:',datos_test_0.shape,datos_test_1.shape)

#creamos las etiquetas
labels_train_0 = np.array([[0]] * n_train)   # shape (n_train, 1)
labels_train_1 = np.array([[1]] * n_train)   # shape (n_train, 1)

labels_test_0 = np.array([[0]] * n_test)     # shape (n_test, 1)
labels_test_1 = np.array([[1]] * n_test)     # shape (n_test, 1)


#añadimos los labels a los frames
completo_train_0 = np.hstack([datos_train_0, labels_train_0])
completo_train_1 = np.hstack([datos_train_1, labels_train_1])
completo_test_0  = np.hstack([datos_test_0,  labels_test_0])
completo_test_1  = np.hstack([datos_test_1,  labels_test_1])

print('train:',completo_train_0.shape,completo_train_1.shape)
print('test:',completo_test_0.shape,completo_test_1.shape)


#juntamos datos
completo_train = np.vstack([completo_train_0, completo_train_1])
completo_test  = np.vstack([completo_test_0,  completo_test_1])

# mezclamos
np.random.shuffle(completo_train)
np.random.shuffle(completo_test)

#separamos
X_train = completo_train[:, :-1]   # todas las columnas menos la última
y_train = completo_train[:, -1]    # solo la última columna (etiquetas)

X_test = completo_test[:, :-1]
y_test = completo_test[:, -1]

print('train:',X_train.shape, y_train.shape)
print('test:',X_test.shape, y_test.shape)


#normalizamos
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)   # Ajusta y transforma solo sobre train
X_test_norm = scaler.transform(X_test)         # Transforma test con el mismo scaler


#------------------------------------------------------------------------------
# MODELOS NO SUPERVISADOS
#------------------------------------------------------------------------------

# PCA
#------------------------------------------------------------------------------

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X_train_norm)

X_train_pca = pca.transform(X_train_norm)
X_test_pca = pca.transform(X_test_norm)


plt.figure(figsize=(8,6))

# Train clase 0
plt.scatter(
    X_train_pca[y_train == 0, 0], 
    X_train_pca[y_train == 0, 1], 
    label='Train Clase 0', alpha=0.7, edgecolors='b', s=10
)

# Train clase 1
plt.scatter(
    X_train_pca[y_train == 1, 0], 
    X_train_pca[y_train == 1, 1], 
    label='Train Clase 1', alpha=0.7, edgecolors='r', s=10
)

plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('Proyección PCA de los datos de entrenamiento y test')
plt.legend()
plt.grid(True)
plt.show()



# K-MEANS
#------------------------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import numpy as np

# Elegimos 2 clusters porque tenemos 2 clases
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_norm)

# Predicciones (etiquetas de cluster)
y_pred_cluster = kmeans.predict(X_test_norm)

# Como KMeans asigna etiquetas arbitrarias (0 o 1 sin relación con tus clases reales),
# necesitamos hacer una asignación "inteligente":
def ajustar_etiquetas(pred, true):
    labels = np.zeros_like(pred)
    for i in np.unique(pred):
        mask = pred == i
        labels[mask] = mode(true[mask])[0]
    return labels

y_pred_ajustado = ajustar_etiquetas(y_pred_cluster, y_test)

# Medimos la precisión
accuracy_kmeans = accuracy_score(y_test, y_pred_ajustado)
print(f"Accuracy K-Means: {accuracy_kmeans * 100:.2f}%")


#------------------------------------------------------------------------------
# MODELOS  SUPERVISADOS
#------------------------------------------------------------------------------

# Regresion Logistica 
#------------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model1 = LogisticRegression(random_state=42) 

model1.fit(X_train_norm, y_train)

y_pred = model1.predict(X_test_norm)

accuracy_LR = accuracy_score(y_test, y_pred)

print(f"Accuracy_LR: {accuracy_LR * 100:.2f}%")

#con PCA

model1 = LogisticRegression(random_state=42)

# Entrenar con datos transformados por PCA
model1.fit(X_train_pca, y_train)

# Predecir con datos transformados por PCA
y_pred = model1.predict(X_test_pca)

accuracy_LR_pca = accuracy_score(y_test, y_pred)

print(f"Accuracy_LR con PCA: {accuracy_LR_pca * 100:.2f}%")


# Random Forest 
#------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators = 30,random_state=42)

model2.fit(X_train_norm, y_train)
y_pred2 = model2.predict(X_test_norm)
accuracy_RF = accuracy_score(y_test, y_pred2)

print(f"Accuracy_RF: {accuracy_RF * 100:.2f}%")
    
#con PCA

model2 = RandomForestClassifier(n_estimators=30, random_state=42)

# Entrenar con datos PCA
model2.fit(X_train_pca, y_train)

# Predecir con datos PCA
y_pred2 = model2.predict(X_test_pca)

accuracy_RF_pca = accuracy_score(y_test, y_pred2)

print(f"Accuracy_RF con PCA: {accuracy_RF_pca * 100:.2f}%")


# KNeighbors
#------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
model3 = KNeighborsClassifier(n_neighbors=3)

model3.fit(X_train_norm, y_train)
y_pred3 = model3.predict(X_test_norm)

accuracy_KN = accuracy_score(y_test, y_pred3)
print(f"Accuracy_KN: {accuracy_KN * 100:.2f}%")

#con PCA

model3 = KNeighborsClassifier(n_neighbors=3)

# Entrenar con datos PCA
model3.fit(X_train_pca, y_train)

# Predecir con datos PCA
y_pred3 = model3.predict(X_test_pca)

accuracy_KN_pca = accuracy_score(y_test, y_pred3)
print(f"Accuracy_KN con PCA: {accuracy_KN_pca * 100:.2f}%")


# Linear Discriminant Analysis
#------------------------------------------------------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Instanciar el modelo
lda = LinearDiscriminantAnalysis()

# Entrenar con datos normalizados
lda.fit(X_train_norm, y_train)

# Predecir
y_pred_lda = lda.predict(X_test_norm)

# Evaluar
accuracy_lda = accuracy_score(y_test, y_pred_lda)
print(f"Accuracy LDA: {accuracy_lda * 100:.2f}%")

#con PCA

lda2 = LinearDiscriminantAnalysis()

# Entrenar con datos normalizados
lda2.fit(X_train_pca, y_train)

# Predecir
y_pred_lda2 = lda2.predict(X_test_pca)

# Evaluar
accuracy_lda2 = accuracy_score(y_test, y_pred_lda2)
print(f"Accuracy LDA con PCA: {accuracy_lda2 * 100:.2f}%")


#------------------------------------------------------------------------------
# RESULTADOS 
#------------------------------------------------------------------------------
print(f"Accuracy_LR : {accuracy_LR * 100:.2f}%")
print(f"Accuracy_RF : {accuracy_RF * 100:.2f}%")
print(f"Accuracy_KN : {accuracy_KN * 100:.2f}%")
print(f"Accuracy LDA: {accuracy_lda * 100:.2f}%")
print(f"Accuracy K-Means: {accuracy_kmeans * 100:.2f}%")

#con PCA

print(f"Accuracy_LR con PCA: {accuracy_LR_pca * 100:.2f}%")
print(f"Accuracy_RF con PCA: {accuracy_RF_pca * 100:.2f}%")
print(f"Accuracy_KN con PCA: {accuracy_KN_pca * 100:.2f}%")
print(f"Accuracy LDA con PCA: {accuracy_lda2 * 100:.2f}%")





















































