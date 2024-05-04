
import numpy as np

def month_sin(t_dat):
    month = t_dat.month - 1
    C = 2*np.pi/12
    return np.sin(month*C).item()

def month_cos(t_dat):
    month = t_dat.month - 1
    C = 2*np.pi/12
    return np.cos(month*C).item()
