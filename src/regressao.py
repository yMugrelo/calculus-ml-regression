import pandas as pd 
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score
caminho = r"C:\Users\User\Desktop\trabalho_calculo\data\calculus_data.csv"
def carregar_dados(caminho):
    return pd.read_csv(caminho)

def regressao_polinomi
-al(X,y, grau = 2):
    poly = PolynomialFeatures(degree = grau)
    X_poly = poly.fit_transform(X)
    modelo = LinearRegression()
    modelo.fit(X_poly, y)
    return modelo, poly

def avaliar_modelo(y_real, y_pred):
    mse = mean_squared_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    return mse, r2

def regressao_linear(X, y):
    modelo = LinearRegression()
    modelo.fit(X,y)
    return modelo

def derivada_analitica(funcao_str, var):
    x = sp.Symbol(var)
    func = sp.sympify(funcao_str)
    derivada = sp.diff(func, x)
    return derivada

def plotar_resultados(X, y, modelo, titulo, poly=None):
    """Plota dados reais e previsão"""
    plt.figure(figsize=(8,5))
    plt.scatter(X, y, color='blue', label='Dados reais')

    X_grid = np.linspace(X.min(), X.max(), 200).reshape(-1,1)
    if poly:
        X_poly = poly.transform(X_grid)
        plt.plot(X_grid, modelo.predict(X_poly), color='red', label='Regressão Polinomial')
    else:
        plt.plot(X_grid, modelo.predict(X_grid), color='red', label='Regressão Linear')

    plt.title(titulo)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()