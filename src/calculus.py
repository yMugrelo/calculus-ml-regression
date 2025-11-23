import numpy as np 
from scipy.integrate import quad 
def compute_area_under_curve(fn, a, b):
    area, _ = quad(fn, a, b)
    return area

def compute_integral_error(model_fn, true_fn, a, b):
    integrand = lambda x: abs(model_fn(x) - true_fn(x))
    area, _ = quad(integrand, a, b)
    return area
