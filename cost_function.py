from variables import variables
from parameters import parameters
import numpy as np

def cost_function(x):
    # Extract variables
    V_a = variables['V_a']
    G_s = variables['G_s']
    A_s = variables['A_s']
    h3 = variables['h3']
    h4 = variables['h4']
    TSSef = variables['TSSef']
    CODef = variables['CODef']
    BODef = variables['BODef']
    TKNef = variables['TKNef']
    S_NO = variables['S_NO']
    Qef = variables['Qef']

    # Constants
    beta_TSS = parameters['beta_TSS']
    beta_COD = parameters['beta_COD']
    beta_BOD = parameters['beta_BOD']
    beta_TKN = parameters['beta_TKN']
    beta_NO = parameters['beta_NO']

    # Objective functions
    f1 = 174.2214 * V_a ** 1.0699 + 12486.713 * G_s ** 0.6216 + 114.8094 * G_s + 955.5 * A_s ** 0.9633 + 41.2706 * (A_s * (1 + h3 + h4)) ** 1.0699
    f2 = 1 / 1000 * (beta_TSS * TSSef + beta_COD * CODef + beta_BOD * BODef + beta_TKN * TKNef + beta_NO * S_NO) * Qef

    return np.array([f1, f2])
