import ROOT
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# --- Common input parameters for testing ---

# J/Psi Yield
N_L_up = 300
N_R_up = 290
N_L_down = 263
N_R_down = 276
N_up = N_L_up + N_R_up
N_down = N_L_down + N_R_down

# Polarization
p_up = 0.86
p_down = 0.83
p_mean = ( p_up + p_down ) / 2.

# Dilution and packing fraction
df = 0.18
pf = 0.60

# Relative luminosity (L_up / L_down)
R = 0.5

# left-right asymmetry
def LR_asymmetry(N_L, N_R):
    return (N_L - N_R) / (N_L + N_R)

def raw_cross_ratio_asym():
    return ( sqrt(N_L_up * N_R_down) - sqrt(N_L_down * N_R_up) ) / ( sqrt(N_L_up * N_R_down) + sqrt(N_L_down * N_R_up) )

# Final asymmetry Method 1: Average polarization asymmetry
def average_pol_asym(An_up, An_down):
    return 1. / (df * pf) / p_mean * ( ( An_up - An_down ) / 2. )

# Final asymmetry Method 2: Inverse polarization weighting asymmetry
def inv_pol_weight_asym(An_up, An_down):
    return 1. / (df * pf) / ( ( N_up / p_up ) + ( N_down / p_down ) )  * ( ( An_up - An_down ) / 2. )

# Final asymmetry Method 3: Cross-ratio average polarization asymmentry
def cross_ratio_averageP_asym(An_raw_CR):
    return 1. / (df * pf) / p_mean * An_raw_CR

# Final asymmetry Method 4: Cross-ratio inverse polarization weighting asymmentry
def cross_ratio_inv_pol_weight_asym(An_raw_CR):
    return 1. / (df * pf) / ( ( N_up / p_up ) + ( N_down / p_down ) ) * An_raw_CR

# Final asymmetry Method 5: Inverse polarization weighting asymmetry with relative luminosity
def inv_pol_weight_R_asym(An_up, An_down):
    return 1. / (df * pf) / ( ( N_up / R / p_up ) + ( N_down / p_down ) )  * ( ( An_up - An_down ) / 2. )

# --- Calculations ---

An_up = LR_asymmetry(N_L_up, N_R_up)
An_down = LR_asymmetry(N_L_down, N_R_down)
An_M1 = average_pol_asym(An_up, An_down)
An_M2 = inv_pol_weight_asym(An_up, An_down)
An_raw_CR = raw_cross_ratio_asym()
An_M3 = cross_ratio_averageP_asym(An_raw_CR)
An_M4 = cross_ratio_inv_pol_weight_asym(An_raw_CR)
An_M5 = inv_pol_weight_R_asym(An_up, An_down)

# --- Output ---
print(f"A_N^up     = {An_up:.7f}")
print(f"A_N^down   = {An_down:.7f}")
print(f"A_N^Method1 = {An_M1:.7f}")
print(f"A_N^Method2 = {An_M2:.7f}")
print(f"A_N^Method3 = {An_M3:.7f}")
print(f"A_N^Method4 = {An_M4:.7f}")
print(f"A_N^Method5 = {An_M5:.7f}")


