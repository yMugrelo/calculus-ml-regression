# Modelagem e Previsão de Funções com Regressão e Cálculo Diferencial

## Integrantes
- **Murilo Rosa**
- **João Victor Hugo**  
- **Almir Rafael**
- **Gustavo Pepece**

**Professor:** Christian Bussmann  
**Disciplina:** Cálculo I

##  Descrição do Projeto

Este projeto integra conceitos de **Cálculo Diferencial** e **Aprendizado de Máquina** para demonstrar como técnicas de regressão podem ser utilizadas para aproximar funções reais e calcular suas derivadas numericamente. Utilizamos dados experimentais de aceleração para:

-  **Modelar** o comportamento dos dados usando regressão linear e polinomial
-  **Calcular** derivadas numericamente a partir dos modelos treinados
-  **Analisar** pontos críticos, taxas de variação e comportamento das funções
-  **Comparar** diferentes modelos de machine learning

##  Objetivos Específicos

- Aplicar regressão linear e polinomial em dados experimentais
- Calcular derivadas numéricas de funções aproximadas
- Identificar pontos críticos (máximos e mínimos)
- Calcular integrais numéricas e áreas sob a curva
- Comparar performance de diferentes algoritmos de regressão
- Visualizar resultados de forma clara e educativa

##  Estrutura do Projeto
calculus-ml-regression/
├── data/
│ └── calculus_data.csv # Dados de aceleração vs tempo
├── src/
│ ├── preprocessing.py # Pré-processamento de dados
│ ├── regressao.py # Modelos de regressão
│ ├── visualization.py # Funções de plotagem
│ └── calculus.py # Operações de cálculo
├── results/ # Resultados e análises
└── notebooks/
└── calculus_ml_analysis.ipynb # Notebook principal


##  Como Usar o Código

### 1. Configuração Inicial

Execute o notebook `calculus_ml_analysis.ipynb` em ordem sequencial. Comece com a célula de configuração:

# Configuração automática do ambiente
import sys
import os
from pathlib import Path

def setup_environment():
    current_dir = Path.cwd()
    # Configura paths automaticamente...
    return project_root, src_path, data_path, results_path

project_root, src_path, data_path, results_path = setup_environment()

2. Carregamento dos Dados
Os dados devem estar no formato CSV na pasta data/. O arquivo padrão é calculus_data.csv com colunas:

time: variável independente

ax, ay, az: componentes de aceleração

aT: aceleração total (variável target)

python
# Carrega e explora os dados
df = pd.read_csv(data_path / "calculus_data.csv")
print(f"Dados carregados: {df.shape[0]} amostras, {df.shape[1]} variáveis")
display(df.head())
3. Configuração da Análise
Escolha da Feature: Você pode selecionar qual variável usar como feature independente:

python
# CONFIGURAÇÃO DO USUÁRIO
DATA_FILENAME = "calculus_data.csv"
TARGET_COLUMN = "aT"                    # Variável a ser prevista
FEATURE_ESCOLHIDA = "time"              # Escolha: "time", "ax", "ay", "az"

# Features disponíveis:
# - "time": análise temporal
# - "ax": aceleração no eixo x
# - "ay": aceleração no eixo y  
# - "az": aceleração no eixo z
4. Pré-processamento
O código automaticamente:

Divide os dados em treino (80%) e teste (20%)

Prepara as features para treinamento

Lida com múltiplas variáveis

python
# Pré-processamento automático
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test, scaler = preprocessor.prepare_data(
    df, TARGET_COLUMN, FEATURE_ESCOLHIDA
)
5. Treinamento dos Modelos
São treinados 3 tipos de modelos:

python
# 1. Regressão Linear
linear_model = regressor.train_linear_regression(X_train, X_test, y_train, y_test)

# 2. Regressão Polinomial (Grau 2)
poly_model_2 = regressor.train_polynomial_regression(X_train, X_test, y_train, y_test, degree=2)

# 3. Regressão Polinomial (Grau 3)  
poly_model_3 = regressor.train_polynomial_regression(X_train, X_test, y_train, y_test, degree=3)
6. Análise de Cálculo
Derivadas:

python
# Calcula primeira e segunda derivadas
primeira_derivada = calculus.calculate_gradient(x_data, y_data)
segunda_derivada = calculus.calculate_second_derivative(x_data, y_data)
Integrais:

python
# Calcula integral numérica
integral_total, integral_acumulativa = calculus.numerical_integral(x_data, y_data)
area_total = calculus.calculate_area_under_curve(x_data, y_data)
Pontos Críticos:

python
# Identifica máximos e mínimos
pontos_criticos = calculus.find_critical_points(x_data, y_data)
7. Visualização dos Resultados
O código gera automaticamente:

Gráficos de regressão para cada modelo

Comparação de performance (R², MSE)

Análise de derivadas e integrais

Identificação de pontos críticos

 Interpretação dos Resultados
Métricas de Avaliação
R² (Coeficiente de Determinação): Mede quanto da variância é explicada pelo modelo

0.9-1.0: Excelente

0.7-0.9: Bom

0.5-0.7: Moderado

<0.5: Fraco

MSE (Mean Squared Error): Erro quadrático médio das previsões

Análise de Derivadas
Primeira Derivada: Representa a taxa de variação instantânea

Segunda Derivada: Indica concavidade (máximos/minimos)

Pontos Críticos: Onde a derivada é zero (mudança de comportamento)

 Exemplos de Aplicação
Caso 1: Análise Temporal
python
FEATURE_ESCOLHIDA = "time"
TARGET_COLUMN = "aT"
Uso: Analisar como a aceleração total varia ao longo do tempo.

Caso 2: Relação entre Componentes
python
FEATURE_ESCOLHIDA = "ax" 
TARGET_COLUMN = "ay"
Uso: Estudar a correlação entre diferentes componentes de aceleração.

Personalização
Adicionando Novos Modelos
python
# Exemplo: Adicionar Ridge Regression
def train_ridge_regression(self, X_train, X_test, y_train, y_test, alpha=1.0):
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    # ... código de avaliação
    return model
Modificando Visualizações
As funções em visualization.py podem ser customizadas para alterar cores, estilos e layouts dos gráficos.

 Dependências
bash
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
seaborn>=0.11.0
jupyter>=1.0.0
 Troubleshooting
Problemas Comuns
Arquivo não encontrado: Verifique se o CSV está na pasta data/

Coluna não existe: Confirme os nomes das colunas no DataFrame

Erro de importação: Execute a célula de configuração primeiro

Para Desenvolvedores
O código é modular e extensível

Novos modelos podem ser adicionados em regressao.py

Funções de cálculo adicional em calculus.py

Visualizações customizáveis em visualization.py

 Conclusões Educativas
Este projeto demonstra na prática:

Como ML pode aproximar funções complexas

A relação entre derivadas numéricas e analíticas

A importância da escolha do modelo (underfitting/overfitting)

Aplicações do cálculo em problemas reais de engenharia

Projeto desenvolvido para a disciplina de Cálculo I - Integrando conceitos matemáticos com aprendizado de máquina.
