import numpy as np
import matplotlib.pyplot as plt

# --- Constants & Inputs ---
# Yields
N_L_up, N_R_up = 300, 290
N_L_down, N_R_down = 263, 276
N_up = N_L_up + N_R_up
N_down = N_L_down + N_R_down

# Polarization
p_up, p_down = 0.86, 0.83
p_mean = (p_up + p_down) / 2.

# Dilution & packing fraction
df, pf = 0.18, 0.60

# Relative luminosity
R = 0.5

def print_result(label, value, error):
    print(f"{label:<15} = {value:.7f} ± {error:.7f}")

# --- Left-Right Asymmetry ---
def LR_asymmetry(N_L, N_R):
    return (N_L - N_R) / (N_L + N_R)

def LR_asymmetry_error(N_L, N_R):
    denom_sq = (N_L + N_R) ** 2
    dA_dNL = 2 * N_R / denom_sq
    dA_dNR = -2 * N_L / denom_sq
    return np.sqrt(dA_dNL**2 * N_L + dA_dNR**2 * N_R)

# --- Cross-ratio Raw Asymmetry ---
def raw_cross_ratio_asym():
    num = np.sqrt(N_L_up * N_R_down) - np.sqrt(N_L_down * N_R_up)
    denom = np.sqrt(N_L_up * N_R_down) + np.sqrt(N_L_down * N_R_up)
    return num / denom

def raw_cross_ratio_asym_error():
    A = np.sqrt(N_L_up * N_R_down)
    B = np.sqrt(N_L_down * N_R_up)
    denom = (A + B)**2
    dA_dN_L_up    = N_R_down * B / (A * denom)
    dA_dN_R_down  = N_L_up   * B / (A * denom)
    dA_dN_L_down  = -N_R_up  * A / (B * denom)
    dA_dN_R_up    = -N_L_down * A / (B * denom)

    return np.sqrt(
        dA_dN_L_up**2 * N_L_up +
        dA_dN_R_down**2 * N_R_down +
        dA_dN_L_down**2 * N_L_down +
        dA_dN_R_up**2 * N_R_up
    )

# --- Method 1: Average polarization asymmetry ---
def average_pol_asym(An_up, An_down):
    return (1. / (df * pf * p_mean)) * ((An_up - An_down) / 2.)

def average_pol_asym_error(err_up, err_down):
    coeff = 1. / (2 * df * pf * p_mean)
    return np.sqrt(coeff**2 * (err_up**2 + err_down**2))

# --- Method 2: Inverse polarization weighting ---
def inv_pol_weight_asym(An_up, An_down):
    denom = (N_up / p_up) + (N_down / p_down)
    return (1. / (df * pf)) / denom * ((An_up - An_down) / 2.)

def inv_pol_weight_asym_error(An_up, An_down, err_up, err_down):
    denom = (N_up / p_up) + (N_down / p_down)
    dN_up = -(An_up - An_down) / (2 * df * pf * p_up * denom**2)
    dN_down = -(An_up - An_down) / (2 * df * pf * p_down * denom**2)
    dAn = 1. / (2 * df * pf * denom)
    return np.sqrt(dN_up**2 * N_up + dN_down**2 * N_down + dAn**2 * (err_up**2 + err_down**2))

# --- Method 3: Cross-ratio with average polarization ---
def cross_ratio_averageP_asym(A_raw):
    return (1. / (df * pf * p_mean)) * A_raw

def cross_ratio_averageP_asym_error(A_raw_err):
    return np.abs(1. / (df * pf * p_mean)) * A_raw_err

# --- Method 4: Cross-ratio with inverse pol weight ---
def cross_ratio_inv_pol_weight_asym(A_raw):
    denom = (N_up / p_up) + (N_down / p_down)
    return (1. / (df * pf)) / denom * A_raw

def cross_ratio_inv_pol_weight_asym_error(A_raw, A_raw_err):
    denom = (N_up / p_up) + (N_down / p_down)
    dN_up = -A_raw / (df * pf * p_up * denom**2)
    dN_down = -A_raw / (df * pf * p_down * denom**2)
    dA = 1. / (df * pf * denom)
    return np.sqrt(dN_up**2 * N_up + dN_down**2 * N_down + dA**2 * A_raw_err**2)

# --- Method 5: Inv pol weight with relative luminosity ---
def inv_pol_weight_R_asym(An_up, An_down):
    denom = (N_up / (R * p_up)) + (N_down / p_down)
    return (1. / (df * pf)) / denom * ((An_up - An_down) / 2.)

def inv_pol_weight_R_asym_error(An_up, An_down, err_up, err_down):
    denom = (N_up / (R * p_up)) + (N_down / p_down)
    dN_up = -(An_up - An_down) / (2 * df * pf * R * p_up * denom**2)
    dN_down = -(An_up - An_down) / (2 * df * pf * p_down * denom**2)
    dAn = 1. / (2 * df * pf * denom)
    return np.sqrt(dN_up**2 * N_up + dN_down**2 * N_down + dAn**2 * (err_up**2 + err_down**2))

# --- Final calculations ---
An_up = LR_asymmetry(N_L_up, N_R_up)
An_down = LR_asymmetry(N_L_down, N_R_down)
delta_An_up = LR_asymmetry_error(N_L_up, N_R_up)
delta_An_down = LR_asymmetry_error(N_L_down, N_R_down)

An_raw_CR = raw_cross_ratio_asym()
delta_An_raw_CR = raw_cross_ratio_asym_error()

An_M1 = average_pol_asym(An_up, An_down)
delta_An_M1 = average_pol_asym_error(delta_An_up, delta_An_down)

An_M2 = inv_pol_weight_asym(An_up, An_down)
delta_An_M2 = inv_pol_weight_asym_error(An_up, An_down, delta_An_up, delta_An_down)

An_M3 = cross_ratio_averageP_asym(An_raw_CR)
delta_An_M3 = cross_ratio_averageP_asym_error(delta_An_raw_CR)

An_M4 = cross_ratio_inv_pol_weight_asym(An_raw_CR)
delta_An_M4 = cross_ratio_inv_pol_weight_asym_error(An_raw_CR, delta_An_raw_CR)

An_M5 = inv_pol_weight_R_asym(An_up, An_down)
delta_An_M5 = inv_pol_weight_R_asym_error(An_up, An_down, delta_An_up, delta_An_down)

# --- Output ---
print_result("A_N^up", An_up, delta_An_up)
print_result("A_N^down", An_down, delta_An_down)
print_result("A_N^Method1", An_M1, delta_An_M1)
print_result("A_N^Method2", An_M2, delta_An_M2)
print_result("A_N^Method3", An_M3, delta_An_M3)
print_result("A_N^Method4", An_M4, delta_An_M4)
print_result("A_N^Method5", An_M5, delta_An_M5)
