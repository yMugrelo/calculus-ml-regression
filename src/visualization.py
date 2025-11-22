import matplotlib.pyplot as plt
import numpy as np

def plot_regression(X, y, y_pred, save_path=None):
    plt.figure(figsize=(10,6))
    plt.scatter(X, y, label="Dados reais", alpha=0.6)
    plt.plot(X, y_pred, "r", label="Regressão", linewidth=2)
    plt.legend()
    plt.title("Regressão — Modelo vs Dados")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_loss(losses, save_path=None):
    plt.figure(figsize=(10,5))
    plt.plot(losses)
    plt.title("Curva de perda")
    if save_path:
        plt.savefig(save_path)
    plt.show()
