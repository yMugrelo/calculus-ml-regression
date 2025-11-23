# Modelagem e Previsão de Funções com Regressão e Cálculo Diferencial

## Integrantes
- Murilo Rosa
- João Victor Hugo  
- Almir Rafael
- Gustavo Pepece

**Professor:** Christian Bussmann  
**Disciplina:** Cálculo I

## Descrição do Projeto

Este projeto integra conceitos de Cálculo Diferencial e Aprendizado de Máquina para demonstrar como técnicas de regressão podem ser utilizadas para aproximar funções reais e calcular suas derivadas numericamente. Utilizamos dados experimentais de aceleração para:

- Modelar o comportamento dos dados usando regressão linear e polinomial
- Calcular derivadas numericamente a partir dos modelos treinados
- Analisar pontos críticos, taxas de variação e comportamento das funções
- Comparar diferentes modelos de machine learning

## Objetivos Específicos

- Aplicar regressão linear e polinomial em dados experimentais
- Calcular derivadas numéricas de funções aproximadas
- Identificar pontos críticos (máximos e mínimos)
- Calcular integrais numéricas e áreas sob a curva
- Comparar performance de diferentes algoritmos de regressão
- Visualizar resultados de forma clara e educativa

## Estrutura do Projeto
calculus-ml-regression/
├── data/
│ └── calculus_data.csv
├── src/
│ ├── preprocessing.py
│ ├── regressao.py
│ ├── visualization.py
│ └── calculus.py
│ └── train.py
│ └── visualization.py
└── notebook/
│ └── trabalho_calculo.ipynb


## Como Usar o Código

### 1. Configuração Inicial

Execute o notebook `calculus_ml_analysis.ipynb` em ordem sequencial. Comece com a célula de configuração:


import sys
import os
from pathlib import Path

def setup_environment():
    current_dir = Path.cwd()
    if (current_dir / 'src').exists():
        project_root = current_dir
    elif (current_dir.parent / 'src').exists():
        project_root = current_dir.parent
    else:
        project_root = current_dir

    src_path = project_root / 'src'
    data_path = project_root / 'data'
    results_path = project_root / 'results'

    data_path.mkdir(exist_ok=True)
    results_path.mkdir(exist_ok=True)

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    return project_root, src_path, data_path, results_path

project_root, src_path, data_path, results_path = setup_environment()

### 2. Carregamento dos Dados

# Os dados devem estar no formato CSV na pasta data/. O arquivo padrão é calculus_data.csv com colunas: time, ax, ay, az, aT.

import pandas as pd

df = pd.read_csv(data_path / "calculus_data.csv")
print(f"Dados carregados: {df.shape[0]} amostras, {df.shape[1]} variáveis")
display(df.head())

### 3. Configuração da Análise
# Selecione a variável para análise:

DATA_FILENAME = "calculus_data.csv"
TARGET_COLUMN = "aT"
FEATURE_ESCOLHIDA = "time"

4. Pré-processamento
python
from sklearn.model_selection import train_test_split

class SimplePreprocessor:
    def prepare_data(self, df, target_column, feature_column=None):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        if feature_column and feature_column in X.columns:
            X = X[[feature_column]]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

preprocessor = SimplePreprocessor()
X_train, X_test, y_train, y_test = preprocessor.prepare_data(
    df, TARGET_COLUMN, FEATURE_ESCOLHIDA
)
5. Treinamento dos Modelos
python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

class RegressionManager:
    def train_linear(self, X_train, X_test, y_train, y_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    
    def train_polynomial(self, X_train, X_test, y_train, y_test, degree=2):
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        model.fit(X_train, y_train)
        return model

regressor = RegressionManager()
linear_model = regressor.train_linear(X_train, X_test, y_train, y_test)
poly_model_2 = regressor.train_polynomial(X_train, X_test, y_train, y_test, degree=2)
poly_model_3 = regressor.train_polynomial(X_train, X_test, y_train, y_test, degree=3)
6. Análise de Cálculo
python
import numpy as np

class CalculusAnalysis:
    def calculate_gradient(self, x, y):
        return np.gradient(y, x)
    
    def calculate_second_derivative(self, x, y):
        first_derivative = self.calculate_gradient(x, y)
        return np.gradient(first_derivative, x)
    
    def numerical_integral(self, x, y):
        integral_total = np.trapz(y, x)
        return integral_total
    
    def find_critical_points(self, x, y):
        primeira_derivada = self.calculate_gradient(x, y)
        pontos_criticos = []
        # Lógica para encontrar pontos críticos
        return pontos_criticos

calculus = CalculusAnalysis()
analysis_df = df.sort_values(by=FEATURE_ESCOLHIDA)
x_data = analysis_df[FEATURE_ESCOLHIDA].values
y_data = analysis_df[TARGET_COLUMN].values

primeira_derivada = calculus.calculate_gradient(x_data, y_data)
segunda_derivada = calculus.calculate_second_derivative(x_data, y_data)
integral_total = calculus.numerical_integral(x_data, y_data)
pontos_criticos = calculus.find_critical_points(x_data, y_data)

Interpretação dos Resultados
Métricas de Avaliação
R² (Coeficiente de Determinação)

0.9-1.0: Excelente
0.7-0.9: Bom
0.5-0.7: Moderado
<0.5: Fraco

MSE (Mean Squared Error): Erro quadrático médio

Análise de Derivadas
Primeira Derivada: Taxa de variação instantânea

Segunda Derivada: Concavidade da função

Pontos Críticos: Onde a derivada é zero

### Exemplos de Aplicação

## Análise Temporal

FEATURE_ESCOLHIDA = "time"
TARGET_COLUMN = "aT"

## Relação entre Componentes

FEATURE_ESCOLHIDA = "ax"
TARGET_COLUMN = "ay"


### Dependências

bash
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
seaborn>=0.11.0
jupyter>=1.0.0
Troubleshooting

### Problemas Comuns

Arquivo não encontrado: Verifique se o CSV está na pasta data/

Coluna não existe: Confirme os nomes das colunas no DataFrame

Erro de importação: Execute a célula de configuração primeiro

### Conclusões
Este projeto demonstra na prática como machine learning pode aproximar funções complexas e como conceitos de cálculo diferencial podem ser aplicados em problemas reais de engenharia, integrando matemática teórica com aplicações computacionais modernas.

## Projeto desenvolvido para a disciplina de Cálculo I - Integrando conceitos matemáticos com aprendizado de máquina.

