#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:03:20 2026

@author: sahily
"""

import numpy as np
import matplotlib.pyplot as plt

# Generamos 1525 puntos aleatorios para el ejemplo
N = 5
np.random.seed(42)

x = np.random.uniform(-10, 10, N)
y = np.random.uniform(-10, 10, N)

# Parámetros de error
error_relativo = 1
max_desplazamiento = 0.2

r_original = np.sqrt(x**2 + y**2)
delta_r_max = np.minimum(error_relativo * r_original, max_desplazamiento)

# Generar desplazamientos aleatorios en círculo
theta = np.random.uniform(0, 2*np.pi, N)
u = np.random.uniform(0, 1, N)
r_error = delta_r_max * np.sqrt(u)

dx = r_error * np.cos(theta)
dy = r_error * np.sin(theta)

x_new = x + dx
y_new = y + dy

# Graficamos
plt.figure(figsize=(8, 8))
plt.scatter(x, y, color='blue', alpha=0.4, label='Original')
plt.scatter(x_new, y_new, color='red', alpha=0.4, label='Con desplazamiento')
plt.title('Desplazamiento aleatorio representando 2% de error (máx 0.2)')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.legend()
plt.show()