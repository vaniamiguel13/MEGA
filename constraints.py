import numpy as np
from variables import variables
from parameters import parameters  # Assuming parameters are stored in a similar way as variables

def constraints(x):
    # Unpack variables
    for key in variables:
        globals()[key] = variables[key]

    # Unpack parameters
    for key in parameters:
        globals()[key] = parameters[key]

    # Initialize constraints
    C = np.zeros(1)
    Ceq = np.zeros(99)

    # Define constraints
    C[0] = Q_P - 2 * A_s

    # Hydraulic retention time
    Ceq[0] = HRT * Q - V_a

    # Aeration tank balances (ASM1 + steady state + CSTR)
    Ceq[1] = -u_H/Y_H * S_S / (K_S + S_S) * (S_O / (K_OH + S_O) + eta_g * K_OH / (K_OH + S_O) * S_NO / (K_NO + S_NO)) * X_BH + k_h * X_BH / (K_X * X_BH + X_S) * (S_O / (K_OH + S_O) + eta_h * K_OH / (K_OH + S_O) * S_NO / (K_NO + S_NO)) * X_S + Q / V_a * (S_Sent - S_S)
    Ceq[2] = (1 - f_p) * b_H * X_BH + (1 - f_p) * b_A * X_BA - k_h * X_BH / (K_X * X_BH + X_S) * (S_O / (K_OH + S_O) + eta_h * K_OH / (K_OH + S_O) * S_NO / (K_NO + S_NO)) * X_S + Q / V_a * (X_Sent - X_S)
    Ceq[3] = u_H * S_S / (K_S + S_S) * (S_O / (K_OH + S_O) + eta_g * K_OH / (K_OH + S_O) * S_NO / (K_NO + S_NO)) * X_BH - b_H * X_BH + Q / V_a * (X_BHent - X_BH)
    Ceq[4] = u_A * S_NH / (K_NH + S_NH) * S_O / (K_OA + S_O) * X_BA - b_A * X_BA + Q / V_a * (X_BAent - X_BA)
    Ceq[5] = f_p * b_H * X_BH + f_p * b_A * X_BA + Q / V_a * (X_Pent - X_P)
    Ceq[6] = -(1 - Y_H) / (2.86 * Y_H) * u_H * S_S / (K_S + S_S) * K_OH / (K_OH + S_O) * S_NO / (K_NO + S_NO) * eta_g * X_BH + u_A / Y_A * S_NH / (K_NH + S_NH) * S_O / (K_OA + S_O) * X_BA + Q / V_a * (S_NOent - S_NO)
    Ceq[7] = -u_H * S_S / (K_S + S_S) * (S_O / (K_OH + S_O) + eta_g * K_OH / (K_OH + S_O) * S_NO / (K_NO + S_NO)) * i_XB * X_BH - u_A * (i_XB + 1 / Y_A) * S_NH / (K_NH + S_NH) * S_O / (K_OA + S_O) * X_BA + k_a * S_ND * X_BH + Q / V_a * (S_NHent - S_NH)
    Ceq[8] = -k_a * X_BH * S_ND + k_h * X_BH / (K_X * X_BH + X_S) * (S_O / (K_OH + S_O) + eta_h * K_OH / (K_OH + S_O) * S_NO / (K_NO + S_NO)) * X_ND + Q / V_a * (S_NDent - S_ND)
    Ceq[9] = b_H * (i_XB - f_p * i_XP) * X_BH + b_A * (i_XB - f_p * i_XP) * X_BA - k_h * X_BH / (K_X * X_BH + X_S) * (S_O / (K_OH + S_O) + eta_h * K_OH / (K_OH + S_O) * S_NO / (K_NO + S_NO)) * X_ND + Q / V_a * (X_NDent - X_ND)
    Ceq[10] = -i_XB / 14 * u_H * (S_S / (K_S + S_S)) * (S_O / (K_OH + S_O)) * X_BH - ((1 - Y_H) / (14 * 2.86 * Y_H) + i_XB / 14) * u_H * (S_S / (K_S + S_S)) * (K_OH / (K_OH + S_O)) * (S_NO / (K_NO + S_NO)) * eta_g * X_BH - (i_XB / 14 + 1 / (7 * Y_A)) * u_A * S_NH / (K_NH + S_NH) * S_O / (K_OA + S_O) * X_BA + 1 / 14 * k_a * S_ND * X_BH + Q / V_a * (S_alkent - S_alk)

    # Oxygen balance
    Ceq[11] = KLa - alfa * G_s * fracO2 * eta * 1333.3 / (V_a * SOST) * 1.024 ** (T - 20)
    Ceq[12] = Q * S_Oent - Q * S_O + KLa * (SOST - S_O) * V_a + ((-(1 - Y_H) / Y_H) * u_H * (S_S / (K_S + S_S)) * (S_O / (K_OH + S_O)) * X_BH - (4.57 - Y_A) / Y_A * u_A * (S_NH / (K_NH + S_NH)) * (S_O / (K_OA + S_O)) * X_BA) * V_a

    # Composite variables
    Ceq[13] = Sent - (S_I + S_Sent)
    Ceq[14] = S - (S_I + S_S)
    Ceq[15] = Xent - (X_I + X_Sent + X_BHent + X_BAent + X_Pent)
    Ceq[16] = X - (X_I + X_S + X_BH + X_BA + X_P)
    Ceq[17] = Xr - (X_Ir + X_Sr + X_BHr + X_BAr + X_Pr)
    Ceq[18] = Xef - (X_Ief + X_Sef + X_BHef + X_BAef + X_Pef)
    Ceq[19] = CODent - (Xent + Sent)
    Ceq[20] = COD - (X + S)
    Ceq[21] = CODr - (Xr + S)
    Ceq[22] = CODef - (Xef + S)
    Ceq[23] = VSSent - Xent * 1 / icv
    Ceq[24] = VSS - X * 1 / icv
    Ceq[25] = VSSr - Xr * 1 / icv
    Ceq[26] = VSSef - Xef * 1 / icv
    Ceq[27] = TSSent - (VSSent + SSI)
    Ceq[28] = TSS - (VSS + SSI)
    Ceq[29] = TSSr - (VSSr + SSIr)
    Ceq[30] = TSSef - (VSSef + SSIef)
    Ceq[31] = BODent - f_BOD * (S_Sent + X_Sent + X_BHent + X_BAent)
    Ceq[32] = BOD - f_BOD * (S_S + X_S + X_BH + X_BA)
    Ceq[33] = BODr - f_BOD * (S_S + X_Sr + X_BHr + X_BAr)
    Ceq[34] = BODef - f_BOD * (S_S + X_Sef + X_BHef + X_BAef)
    Ceq[35] = TKNent - (S_NHent + S_NDent + X_NDent + i_XB * (X_BHent + X_BAent) + i_XP * (X_Pent + X_I))
    Ceq[36] = TKN - (S_NH + S_ND + X_ND + i_XB * (X_BH + X_BA) + i_XP * (X_P + X_I))
    Ceq[37] = TKNr - (S_NH + S_ND + X_NDr + i_XB * (X_BHr + X_BAr) + i_XP * (X_Pr + X_Ir))
    Ceq[38] = TKNef - (S_NH + S_ND + X_NDef + i_XB * (X_BHef + X_BAef) + i_XP * (X_Pef + X_Ief))
    Ceq[39] = Nent - (TKNent + S_NOent)
    Ceq[40] = N - (TKN + S_NO)
    Ceq[41] = Nr - (TKNr + S_NO)
    Ceq[42] = Nef - (TKNef + S_NO)
    Ceq[43] = (Qw * Xr) * SRT - V_a * X
    Ceq[44] = (1 + r) * Qinf * Xent - (Qinf * Xinf + (1 + r) * Qinf * X - V_a * X / (SRT * Xr) * (Xr - Xef) - Qinf * Xef)
    Ceq[45] = (1 + r) * Qinf * SSI - (Qinf * TSSinf * 0.2 + (1 + r) * Qinf * SSI - V_a * SSI / (SRT * Xr) * (SSIr - SSIef) - Qinf * SSIef)
    Ceq[46] = (1 + r) * Qinf * X_NDent - (Qinf * X_NDinf + (1 + r) * Qinf * X_ND - V_a * X / (SRT * Xr) * (X_NDr - X_NDef) - Qinf * X_NDef)
    Ceq[47] = (1 + r) * Qinf * S_Sent - (Qinf * S_Sinf + r * Qinf * S_S)
    Ceq[48] = (1 + r) * Qinf * S_Oent - (Qinf * S_Oinf + r * Qinf * S_O)
    Ceq[49] = (1 + r) * Qinf * S_NOent - (Qinf * S_NOinf + r * Qinf * S_NO)
    Ceq[50] = (1 + r) * Qinf * S_NHent - (Qinf * S_NHinf + r * Qinf * S_NH)
    Ceq[51] = (1 + r) * Qinf * S_NDent - (Qinf * S_NDinf + r * Qinf * S_ND)
    Ceq[52] = (1 + r) * Qinf * S_alkent - (Qinf * S_alkinf + r * Qinf * S_alk)
    Ceq[53] = r * Qinf - Qr
    Ceq[54] = Q - (Qinf + Qr)
    Ceq[55] = Q - (Qef + Qr + Qw)
    Ceq[56] = Q_P - 2400 * A_s * (0.7 * TSS / 1000 * IVLD) ** -1.34
    Ceq[57] = h3 - 0.3 * TSS / 1000 * V_a * IVLD / (480 * A_s)
    Ceq[58] = h4 - 0.7 * TSS / 1000 * IVLD / 1000
    Ceq[59] = r * (TSSr_max - TSS) - TSS
    Ceq[60] = r_p * (TSSr_max_p - 0.7 * TSS) - 0.7 * TSS
    Ceq[61] = Qr_p - r_p * Q_P
    Ceq[62] = VSS - 0.7 * TSS
    Ceq[63] = VSSef - 0.7 * TSSef
    Ceq[64] = v_up * A_s - Qef
    Ceq[65] = v_dn * A_s - (Qr + Qw)
    Ceq[66] = v_s1 - max(0, min(v1_0, v_0 * (np.exp(-r_h * (ST1 - f_ns * TSS)) - np.exp(-r_P * (ST1 - f_ns * TSS)))))
    Ceq[67] = v_s2 - max(0, min(v1_0, v_0 * (np.exp(-r_h * (ST2 - f_ns * TSS)) - np.exp(-r_P * (ST2 - f_ns * TSS)))))
    Ceq[68] = v_s3 - max(0, min(v1_0, v_0 * (np.exp(-r_h * (ST3 - f_ns * TSS)) - np.exp(-r_P * (ST3 - f_ns * TSS)))))
    Ceq[69] = v_s4 - max(0, min(v1_0, v_0 * (np.exp(-r_h * (ST4 - f_ns * TSS)) - np.exp(-r_P * (ST4 - f_ns * TSS)))))
    Ceq[70] = v_s5 - max(0, min(v1_0, v_0 * (np.exp(-r_h * (ST5 - f_ns * TSS)) - np.exp(-r_P * (ST5 - f_ns * TSS)))))
    Ceq[71] = v_s6 - max(0, min(v1_0, v_0 * (np.exp(-r_h * (ST6 - f_ns * TSS)) - np.exp(-r_P * (ST6 - f_ns * TSS)))))
    Ceq[72] = v_s7 - max(0, min(v1_0, v_0 * (np.exp(-r_h * (ST7 - f_ns * TSS)) - np.exp(-r_P * (ST7 - f_ns * TSS)))))
    Ceq[73] = v_s8 - max(0, min(v1_0, v_0 * (np.exp(-r_h * (ST8 - f_ns * TSS)) - np.exp(-r_P * (ST8 - f_ns * TSS)))))
    Ceq[74] = v_s9 - max(0, min(v1_0, v_0 * (np.exp(-r_h * (ST9 - f_ns * TSS)) - np.exp(-r_P * (ST9 - f_ns * TSS)))))
    Ceq[75] = v_s10 - max(0, min(v1_0, v_0 * (np.exp(-r_h * (ST10 - f_ns * TSS)) - np.exp(-r_P * (ST10 - f_ns * TSS)))))

    # Constraints related to ST and J variables
    if ST2 <= ST_t:
        Ceq[76] = J1 - v_s1 * ST1
    else:
        Ceq[76] = J1 - min(v_s1 * ST1, v_s2 * ST2)

    if ST3 <= ST_t:
        Ceq[77] = J2 - v_s2 * ST2
    else:
        Ceq[77] = J2 - min(v_s2 * ST2, v_s3 * ST3)

    if ST4 <= ST_t:
        Ceq[78] = J3 - v_s3 * ST3
    else:
        Ceq[78] = J3 - min(v_s3 * ST3, v_s4 * ST4)

    if ST5 <= ST_t:
        Ceq[79] = J4 - v_s4 * ST4
    else:
        Ceq[79] = J4 - min(v_s4 * ST4, v_s5 * ST5)

    if ST6 <= ST_t:
        Ceq[80] = J5 - v_s5 * ST5
    else:
        Ceq[80] = J5 - min(v_s5 * ST5, v_s6 * ST6)

    if ST7 <= ST_t:
        Ceq[81] = J6 - v_s6 * ST6
    else:
        Ceq[81] = J6 - min(v_s6 * ST6, v_s7 * ST7)

    if ST8 <= ST_t:
        Ceq[82] = J7 - v_s7 * ST7
    else:
        Ceq[82] = J7 - min(v_s7 * ST7, v_s8 * ST8)

    # Additional constraints for ST and J
    Ceq[83] = J8 - v_s8 * ST8
    Ceq[84] = J9 - v_s9 * ST9
    Ceq[85] = J10 - v_s10 * ST10

    # Feed layer (m=7)
    Ceq[86] = ((Q * TSS) / A_s + J7 - (v_up + v_dn) * ST7 - min(J8, J9)) / (h / 10)

    # Intermediate layers below the feed layer (m=8 and m=9)
    Ceq[87] = (v_dn * (ST7 - ST8) + min(J8, J7) - min(J8, J9)) / (h / 10)
    Ceq[88] = (v_dn * (ST8 - ST9) + min(J9, J8) - min(J9, J10)) / (h / 10)

    # Lower layer (m=10)
    Ceq[89] = (v_dn * (ST9 - ST10) + min(J9, J10)) / (h / 10)

    # Intermediate layers above the feed layer (m=2 to 6)
    Ceq[90] = (v_up * (ST3 - ST2) + J1 - J2) / (h / 10)
    Ceq[91] = (v_up * (ST4 - ST3) + J2 - J3) / (h / 10)
    Ceq[92] = (v_up * (ST5 - ST4) + J3 - J4) / (h / 10)
    Ceq[93] = (v_up * (ST6 - ST5) + J4 - J5) / (h / 10)
    Ceq[94] = (v_up * (ST7 - ST6) + J5 - J6) / (h / 10)

    # Upper layer (m=1)
    Ceq[95] = (v_up * (ST2 - ST1) - J1) / (h / 10)

    # Additional constraints
    Ceq[96] = ST1 - TSSef
    Ceq[97] = ST10 - TSSr
    Ceq[98] = h - (h3 + h4 + 1)


    return np.array(C), np.array(Ceq)
