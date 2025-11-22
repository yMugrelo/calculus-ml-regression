import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample(path: str, n = 200, noise = 5.0, seed = 42):
    np.random.seed(seed)
    x = np.linspace(-10, 10, n)

    y = (x**3)/30 - 0.5 * x + 10 *np.sin(x/2) + np.random.randn(n) * noise

    df = pd.DataFrame({"x": x, "y" : y})
    Path(path).parent.mkdir(exist_ok=True )
    df.to_csv(path, index= False)
    print(f"Dataset Salvo em {path}")

def load_csv(path : str):
    return pd.read_csv(path)