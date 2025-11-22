#!/usr/bin/env python3
"""
Setup script para o projeto Calculus ML Regression
Configura o ambiente e paths automaticamente
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Verifica a versão do Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 ou superior é necessário")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")

def setup_paths():
    """Configura os paths do projeto automaticamente"""
    # Obtém o diretório root do projeto
    project_root = Path(__file__).parent.absolute()
    src_path = project_root / "src"
    
    # Adiciona ao sys.path
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Configura variáveis de ambiente
    os.environ["PROJECT_ROOT"] = str(project_root)
    os.environ["DATA_PATH"] = str(project_root / "data")
    os.environ["RESULTS_PATH"] = str(project_root / "results")
    
    print(f"📁 Project Root: {project_root}")
    print(f"📁 Source Path: {src_path}")
    
    return project_root, src_path

def create_directories():
    """Cria diretórios necessários"""
    directories = ["data", "results", "notebooks", "models"]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"📂 Diretório criado: {dir_path}")

def install_requirements():
    """Instala dependências do requirements.txt"""
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print("⚠️  requirements.txt não encontrado, criando...")
        create_requirements_file()
        return
    
    print("📦 Instalando dependências...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("✅ Dependências instaladas com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")

def create_requirements_file():
    """Cria arquivo requirements.txt se não existir"""
    requirements = """numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
jupyter>=1.0.0
plotly>=5.0.0
notebook>=6.4.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✅ requirements.txt criado!")

def check_system_info():
    """Exibe informações do sistema"""
    system = platform.system()
    machine = platform.machine()
    print(f"💻 Sistema: {system} {machine}")

def main():
    """Função principal de setup"""
    print("🚀 Configurando Calculus ML Regression...")
    print("="*50)
    
    # Executa todas as etapas
    check_system_info()
    check_python_version()
    project_root, src_path = setup_paths()
    create_directories()
    install_requirements()
    
    print("="*50)
    print("✅ Setup concluído com sucesso!")
    print(f"📁 Seu projeto está em: {project_root}")
    print("\n🎯 Próximos passos:")
    print("   1. Coloque seus dados em /data/")
    print("   2. Execute: python src/train.py")
    print("   3. Ou: jupyter notebook trabalho_calculo.ipynb")

if __name__ == "__main__":
    main()