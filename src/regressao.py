import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class RegressionModels:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_linear_regression(self, X_train, X_test, y_train, y_test):
        """Treina modelo de regressão linear"""
        print("🧠 Treinando Regressão Linear...")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        

        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)
        
        self.models['linear'] = model
        self.results['linear'] = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': {'train': y_pred_train, 'test': y_pred_test}
        }
        
        print("✅ Regressão Linear treinada!")
        self._print_metrics("Treino", train_metrics)
        self._print_metrics("Teste", test_metrics)
        return model
    
    def train_polynomial_regression(self, X_train, X_test, y_train, y_test, degree=2):
        """Treina regressão polinomial"""
        print(f"🎯 Treinando Regressão Polinomial (grau {degree})...")
        
        poly = PolynomialFeatures(degree=degree)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_poly_train, y_train)
        
        y_pred_train = model.predict(X_poly_train)
        y_pred_test = model.predict(X_poly_test)
        
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)
        
        self.models[f'polynomial_degree_{degree}'] = model
        self.results[f'polynomial_degree_{degree}'] = {
            'model': model,
            'poly_transformer': poly,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': {'train': y_pred_train, 'test': y_pred_test}
        }
        
        print(f"✅ Regressão Polinomial (grau {degree}) treinada!")
        self._print_metrics("Treino", train_metrics)
        self._print_metrics("Teste", test_metrics)
        return model, poly
    
    def train_ridge_regression(self, X_train, X_test, y_train, y_test, alpha=1.0):
        """Treina regressão Ridge"""
        print(f"🏔️ Treinando Ridge Regression (alpha={alpha})...")
        
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)
        
        self.models['ridge'] = model
        self.results['ridge'] = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': {'train': y_pred_train, 'test': y_pred_test}
        }
        
        print("✅ Ridge Regression treinada!")
        self._print_metrics("Treino", train_metrics)
        self._print_metrics("Teste", test_metrics)
        return model
    
    def train_lasso_regression(self, X_train, X_test, y_train, y_test, alpha=1.0):
        """Treina regressão Lasso"""
        print(f"🎯 Treinando Lasso Regression (alpha={alpha})...")
        
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)
        
        self.models['lasso'] = model
        self.results['lasso'] = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': {'train': y_pred_train, 'test': y_pred_test}
        }
        
        print("✅ Lasso Regression treinada!")
        self._print_metrics("Treino", train_metrics)
        self._print_metrics("Teste", test_metrics)
        return model
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calcula métricas de avaliação"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}
    
    def _print_metrics(self, dataset_name, metrics):
        """Imprime métricas formatadas"""
        print(f"📊 {dataset_name} - MSE: {metrics['MSE']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, R²: {metrics['R²']:.4f}")
    
    def compare_models(self):
        """Compara o desempenho de todos os modelos"""
        print("\n" + "="*50)
        print("📈 COMPARAÇÃO DE MODELOS")
        print("="*50)
        
        comparison = []
        for model_name, result in self.results.items():
            comparison.append({
                'Modelo': model_name,
                'MSE_Treino': result['train_metrics']['MSE'],
                'RMSE_Treino': result['train_metrics']['RMSE'],
                'R²_Treino': result['train_metrics']['R²'],
                'MSE_Teste': result['test_metrics']['MSE'],
                'RMSE_Teste': result['test_metrics']['RMSE'],
                'R²_Teste': result['test_metrics']['R²']
            })
        
        return pd.DataFrame(comparison)

def carregar_dados(caminho):
    return pd.read_csv(caminho)

def regressao_polinomial(X, y, grau=2):
    poly = PolynomialFeatures(degree=grau)
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
    modelo.fit(X, y)
    return modelo

def derivada_analitica(funcao_str, var):
    x = sp.Symbol(var)
    func = sp.sympify(funcao_str)
    derivada = sp.diff(func, x)
    return derivada

def plotar_resultados(X, y, modelo, titulo, poly=None):
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

if __name__ == "__main__":
    print("✅ regressao.py carregado com sucesso!")