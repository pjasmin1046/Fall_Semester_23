import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# This loop will calculate Pressure for different Ts from 273(K) to 285(K)
T_range = np.arange(273, 286, 1)
phi_matrix = np.zeros_like(T_range, dtype=float)
pressure_matrix = np.zeros_like(T_range, dtype=float)
fugacity_matrix = np.zeros_like(T_range, dtype=float)

for i, T in enumerate(T_range):
    # methane and water mole percents in gas phase equilibrium
    yM = 0.99
    yw = 0.01

    # Gas Constants
    R = 8.314

    # Langmuir Constant
    Cm = ((0.0037237 / T) * np.exp(2708.8 / T)) + ((0.018327 / T) * np.exp(2738 / T))

    # reference hydrate formation P-T empirical equation
    pr = 101325 * (np.exp(-1212.2 + (44344 / T) + (187.719 * np.log(T))))

    # Chemical potential
    dmiwL = 1263.6 / (R * 273.15)

    # deltaH Integration
    dhf = \
    spi.quad(lambda y: (4858.9 + (-86 * (y - 273.15) + (0.141 * (((y ** 2) - (273.15 ** 2)) / 2)))) / (R * y ** 2),
             273.15, T)[0]

    # dmio(T,Pr)
    dmiwL += spi.quad(lambda Tf: (1.8e-3) / (R * Tf) * (pr - dhf), 273.15, T)[0]

    # Methane Properties
    TcM = 190.55
    PcM = 4599200
    wM = 0.0104
    TrM = T / TcM
    aM = (0.4274 * ((R ** 2) * (TcM ** 2) / PcM)) * (
                1 + ((0.480 + (1.574 * wM) - (0.176 * (wM ** 2))) * (1 - (TrM ** 0.5)))) ** 2
    bM = 0.08664 * (R * TcM / PcM)

    # Water Properties
    Tcw = 647.14
    Pcw = 22064000
    ww = 0.3442
    Trw = T / Tcw
    aw = (0.4274 * ((R ** 2) * (Tcw ** 2) / Pcw)) * (
                1 + ((0.480 + (1.574 * ww) - (0.176 * (ww ** 2))) * (1 - (Trw ** 0.5)))) ** 2
    bw = 0.08664 * (R * Tcw / Pcw)

    # Mixture parameters of SRK EOS
    am = ((yM * (aM ** 0.5)) + (yw * (aw * 0.5))) ** 2
    bm = yM * bM + yw * bw
    A1m = am / ((R * T) ** 2)
    B2m = bm / (R * T)

    # Pressure variable
    p = 101325  # Set an initial value for pressure

    # Phase equilibrium rule, dmioL = dmiobeta
    phi = 0.9
    F = (dmiwL + ((1.8e-3) * (p - pr))) - (np.log(1 + Cm * phi * yM * (p / 101325)) * (3 / 23)) - 6.2

    # SRK mixing parameters Functions for phi&Z
    Am = (A1m * p)
    Bm = (B2m * p)
    dp = 1e-8  # Small change in pressure
    f = np.gradient(F, dp)

    # First Guess for P, Solving with Newton-Raphson method
    p0 = 10e6
    ite = 0

    while ite < 10 ** 3:
        Am = (A1m * p0)
        Bm = (B2m * p0)

        # Newton-Raphson loop
        for ite in range(1000):
            Am = A1m * p0
            Bm = B2m * p0

            Z = np.roots([1, -1, Am - Bm - Bm ** 2, -Am * Bm])
            Z = np.real(Z[np.isreal(Z)])  # Keep only real roots

            # Calculate fugacity coefficient using SRK EOS for mixtures
            phi = np.exp((bM / bm) * (Z - 1) - np.log(Z - Bm) - (Am / Bm) * (2 * (aM / am) ** 0.5 - bM / bm) * np.log(
                1 + Bm / Z))

            # Newton-Raphson update
            pf = p0 - F / f

            # Check convergence
            if abs(pf - p0) < 1:
                break

            # Update p0
            p0 = abs(pf)

    # Matrix of calculated phi
    phi_matrix[i] = phi

    # Matrix of calculated pressures
    pressure_matrix[i] = pf

    # Matrix of calculated Fugacities in ATM
    fugacity_matrix[i] = (phi * pf) / 101325

# Plotting Hydrate Formation Pressure based on given Temperature
fig1 = go.Figure(data=go.Scatter(x=T_range, y=pressure_matrix, mode='lines+markers'))
fig1.update_layout(title='Hydrate Formation Pressure vs. Temperature', xaxis_title='Temperature (K)',
                   yaxis_title='Pressure (Pa)')
fig1.show()

# Plotting Methane Fugacity in hydrate Formation temperature
fig2 = go.Figure(data=go.Scatter(x=T_range, y=fugacity_matrix, mode='lines+markers'))
fig2.update_layout(title='Methane Fugacity vs. Temperature', xaxis_title='Temperature (K)',
                   yaxis_title='Fugacity (atm)')
fig2.show()
