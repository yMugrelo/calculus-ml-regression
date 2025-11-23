import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SRC_PATH = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_PATH = PROJECT_ROOT / "results"
NOTEBOOKS_PATH = PROJECT_ROOT / "notebooks"

DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42
POLYNOMIAL_DEGREE = 2

PLOT_STYLE = 'seaborn-v0_8'
COLOR_PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
FIG_SIZE = (12, 8)

# Criar diretórios se não existirem
for path in [DATA_PATH, RESULTS_PATH, NOTEBOOKS_PATH]:
    path.mkdir(exist_ok=True)

def get_data_path(filename="calculus_data.csv"):
    """Retorna path completo para arquivo de dados"""
    return DATA_PATH / filename

def get_results_path(filename):
    """Retorna path completo para arquivo de resultados"""
    return RESULTS_PATH / filename

def show_project_info():
    """Mostra informações do projeto"""
    print("📊 Calculus ML Regression Project")
    print("📍 Machine Learning com Cálculo Integral")
    print("🚀 Desenvolvido para análise de dados matemáticos")
    print(f"📁 Project root: {PROJECT_ROOT}")