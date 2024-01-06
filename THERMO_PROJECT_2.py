import numpy as np
import scipy
import plotly.graph_objects as go
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import root_scalar, newton
from plotly.subplots import make_subplots

# Constants to find xW
Henrys_law_ref_methane = (2.56325 * (10**-5)) / 1.01325 # 1/bar
Vinf = 0.04134 # L/mol
Psat_W = 0.031285 * 1.01325 # bar for water at 298K

# Parameters for dP/Dt empirical relationship
A = -1212.2
B = 44344
C = 187.719

# From Table 5
del_HWAlpha = -11.506  # HectoJoules

# from Table 5
del_HWf = 60.095  # HectoJoules

# From Table 5
del_VWAlpha = 0.003  # L/ mol

# From densities of ice and water and Mw = 18.02
del_VWf = 0.001782  # L/mol

deltaMuWAlpha = 12.6357

nu = 6 / 46  # cavities/ water m'cule

R = 0.08314  # L Bar/ mol K
# R = 8.314 # m^3 Pa /mol K

def SRK_EOS_FUGACITY_MIXTURE(T, P, y, debug = False):
    # R = 8.314 # m^3 Pa /mol K
    if debug:
        print("***")
        print(f'FUGACITY FROM SRK EQUATION @ T = {T} K, P = {P} Pa, y = {y}\n')

    # # Critical Properties [Tc: K, Pc: Bar, W, Vc: m^3/kg]
    # Methane_crit = [190.56, 45.99, 0.0104, 0.00615]
    # H2O_crit = [647, 220.64, 0.3442, 0.003155]

    # Critical Properties [Tc: K, Pc: Pa, W, Vc: m^3/kg]
    Methane_crit = [190.56, 4599200, 0.0104, 0.00615]
    H2O_crit = [647, 22064000, 0.3442, 0.003155]

    Tr_Water = T / H2O_crit[0]
    Tr_Methane = T / Methane_crit[0]

    # a, b for each component
    aM = (0.4274 * ((R ** 2) * (Methane_crit[0] ** 2) / Methane_crit[1])) * (1 + ((0.480 + (1.574 * Methane_crit[2]) - (0.176 * Methane_crit[2] ** 2)) * (1 - (Tr_Methane ** 0.5)))) ** 2
    aW = (0.4274 * ((R ** 2) * (H2O_crit[0] ** 2) / H2O_crit[1])) * (1 + ((0.480 + (1.574 * H2O_crit[2]) - (0.176 * H2O_crit[2] ** 2)) * (1 - (Tr_Water ** 0.5)))) ** 2
    bM = 0.08664 * (R * Methane_crit[0] / Methane_crit[1])
    bW = 0.08664 * (R * H2O_crit[0] / H2O_crit[1])

    if debug:
        print(f'aM: {aM}')
        print(f'bM: {bM}')
        print(f'aW: {aW}')
        print(f'bW: {bW}')

    # Mixture Parameters
    am = ((y * (aM ** 0.5)) + ((1-y) * (aW * 0.5))) ** 2
    bm = (y * bM) + ((1-y) * bW)
    A = (am * P) / ((R * T) ** 2)
    B = (bm * P) / (R * T)

    if debug:
        print(f'\nam: {am}')
        print(f'bm: {bm}')
        print(f'A: {A}')
        print(f'B: {B}')

    Z = lambda Z: (Z ** 3) - (Z ** 2) + Z * (A - B - (B ** 2)) - (A * B)

    z = root_scalar(Z, bracket=[0.1, 100])

    if debug:
        print(f'\nz: {z}')

    # lnPhi = z.root - 1 - np.log(z.root - B) - ((A/B) * np.log((z.root -B) / z.root))
    lnPhi = ((bM/bm) * (z.root-1)) - np.log(z.root - B) - ((A/B)) * (2 * ((aM / am) ** 0.5) - (bM/bm)) * np.log(1 + (B / z.root))
    phi = np.exp(lnPhi)

    if debug:
        print(f'\nlnPhi: {lnPhi}')
        print(f'\nPhi: {phi}')

    return phi


def RK_EOS_FUGACITY_MIXTURE(T, P, y, debug = False):

    if debug:
        print("***")
        print(f'FUGACITY FROM RK EQUATION @ T = {T} K, P = {P} bar, y = {y}\n')

    # Critical Properties [Tc: K, Pc: Bar, W, Vc: m^3/kg]
    Methane_crit = [190.56, 45.99, 0.008, 0.00615]
    H2O_crit = [647, 220.64, 0.3433, 0.003155]

    # a and b for each component
    am = (0.42748 * (R ** 2) * (Methane_crit[0] ** 2.5)) / Methane_crit[1]
    aw = (0.42748 * (R ** 2) * (H2O_crit[0] ** 2.5)) / H2O_crit[1]

    bm = (0.08664 * R * Methane_crit[0]) / Methane_crit[1]
    bw = (0.08664 * R * H2O_crit[0]) / H2O_crit[1]

    if debug:
        print(f'COMPONENT a AND b VALUES:')
        print(f'a WATER: {aw}')
        print(f'b WATER: {bw}')
        print(f'a METHANE: {am}')
        print(f'b METHANE: {bm}\n')

    # RK FUGACITY CALCULATIONS

    # Mixture parameters
    a12 = (am * aw) ** 0.5
    aMix = (y ** 2) * am + 2 * y * (1-y) * a12 + ((1-y) ** 2) * aw
    bMix = y * bm + (1-y) * bw

    if debug:
        print('MIXING PARAMETERS:')
        print(f'a12: {a12}')
        print(f'a MIXTURE: {aMix}')
        print(f'b MIXTURE: {bMix}\n')

    RK = lambda V: R*T/(V-bMix)-aMix/((T**0.5)*V*(V+bMix))-P

    V_Newton = newton(RK, x0 = 0.5)
    Z = P * V_Newton / (R * T)

    lnPhi = (bm / bMix) * (Z - 1) - (V_Newton - bMix) * P / (R * T) + (1 / (bMix * R * T ** 1.5)) * (aMix * bm/bMix - 2 * (y * am + (1 - y) * a12)) * (1 + bMix / V_Newton)

    phiM = np.exp (lnPhi)

    # f = phiM * y * P

    if debug:
        print(f'V: {V_Newton}\n')
        print(f'Z: {Z}')
        print(f'Phi MIXTURE: {phiM}')
        # print(f'f: {f}')
        print('***\n')

    return V_Newton, phiM


def findxW(T, P, y):
    R = 0.08314 # L bar/ mol K
    phi = RK_EOS_FUGACITY_MIXTURE(T, P, y)
    f = phi * y * P
    xM = np.exp(f/(np.log(Henrys_law_ref_methane) +Vinf * (P - Psat_W) / (R*T)))
    xW = 1 - xM
    return xW

# Calulates Langmuir
def EQN_A1(T, ST, debug = False):
    '''
    Calulates Langmuir constant for Methane between 260-300 K using empirical relationship
    :param T: Temperature (K)
    :param ST: Structure Type (1 or 2)
    :param debug: Prints
    :return: CL (Langmuir Constant)
    '''


    if T < 260 or T > 300:
        print("INPUT TEMPERATURE TO A-1 NOT WITHIN ACCEPTABLE RANGE (260 - 300 K)")
        return

    # Small and Large cavity constants
    Al = 1.8372 * (10**-2)
    Bl = 2.7379 * (10**3)
    As = 3.7237 * (10**-3)
    Bs = 2.7088 * (10**3)


    if debug:
        print('***')
        print(f'EQN - A-1: Langmuir Constant Calculation @ T = {T} K')
        print(f'As: {As}')
        print(f'Bs: {Bs}')
        print(f'Al: {Al}')
        print(f'Bl: {Bl}')


    if ST == 1:
        res = ((Al / T) * np.exp(Bl / T)) + ((As / T) * np.exp(Bs / T))
    elif ST == 2:
        res = (Al / T) * np.exp(Bl / T)

    if debug:
        print(f'EQN - A-1 RESULT: {res}')
        print("***\n")

    return res


def findPR(T, debug = False):
    # Returns Pr from empirical relationship (EQN - 14) with parameters as described below
    A = -1212.2
    B = 44344
    C = 187.719

    res = np.exp(A + (B/T) + (C * np.log(T)))

    if debug:
        print("***")
        print(f'PR (bar) at {T} K: {res}')
        print("***\n")
    return res * 1.01325


def EQN_12(T, y, debug = False):
    # Constants and initial parameters
    R = 0.08314 # L bar / mol K
    T0 = 273
    P0 = findPR(T0)


    if T>=273:
        if debug:
            print('***')
            print(f'EQN - 12a @ T = {T} K:\n')
            print(f'T0: {T0} K')
            print(f'P0: {P0} Bar')

        # dPrdT = (np.exp(A)) * (((-B / (T ** 2)) * np.exp(B / T) * (T ** C)) + (C * np.exp(B / T) * (T ** (C - 1))))
        # dPrdT = ((C/T) - (B/(T**2))) * np.exp((C * np.log(T)) + (B/T) + A)

        # First integral term
        def integrand1(T):
            return (del_HWAlpha + del_HWf) / (R * (T**2))
        integral1, error1 = quad(integrand1, T0, T)

        if debug:
            print(f'With del_HWAlpha + del_HWf = {del_HWAlpha + del_HWf}, first integral EVAL to {integral1}')

        # Second integral term
        def integrand2(T):
            return (del_VWf / (R * T)) * (((C/T) - (B/(T**2))) * np.exp((C * np.log(T)) + (B/T) + A))

        integral2, error2 = quad(integrand2, T0, T)

        res = ((deltaMuWAlpha / (R * T0)) - integral1 + integral2)

        if debug:
            print(f'With del_VWf = {del_VWf}, second integral EVAL to {integral2}')
            print(f'EQN 12a RESULT = {res}')
            print('***\n')

        return res

    else:

        if debug:
            print('***')
            print(f'EQN - 12b @ T = {T} K:\n')
            print(f'T0: {T0} K')
            print(f'P0: {P0} Bar')


        # First integral term
        def integrand1(T):
            return (del_HWAlpha) / (R * (T**2))
        integral1, error1 = quad(integrand1, T0, T)

        if debug:
            print(f'With del_HWAlpha = {del_HWAlpha}, first integral EVAL to {integral1}')

        # Second integral term
        def integrand2(T):
            return ((del_VWAlpha) / (R * T)) * (((C/T) - (B/(T**2))) * np.exp((C * np.log(T)) + (B/T) + A))

        integral2, error2 = quad(integrand2, T0, T)

        res = ((deltaMuWAlpha / (R * T0)) - integral1 + integral2)

        if debug:
            print(f'With del_VWAlpha = {del_VWAlpha}, second integral EVAL to {integral2}')
            print(f'EQN - 12b RESULT = {res}')
            print('***')

        return res


def EQN_15(T, P, y, debug = False):

    if debug:
        print('***')
        print(f'EQN - 15 @ T = {T} K and P = {P} bar\n')
        phi = SRK_EOS_FUGACITY_MIXTURE(T, P * (10**6), y, debug =True)
        xW = 1
        Langmuir = EQN_A1(T, 1, debug =True)

    if not debug:
        phi = SRK_EOS_FUGACITY_MIXTURE(T, P * (10**6), y)
        xW = 1
        Langmuir = EQN_A1(T, 1)

    delta_muW =  nu * np.log(1 + (Langmuir * phi * y * P))

    if debug:
        print(f'phi: {phi}')
        print(f'xW: {xW}')
        print(f'Cm: {Langmuir}')
        print(f'delta_muW: {delta_muW}')

    if T>=273:
        res = delta_muW + (np.log(xW))
        if debug:
            print(f'EQN - 15 RESULT: {res}')
            print('***\n')
        return res
    else:
        if debug:
            print(f'EQN - 15 RESULT: {delta_muW}')
            print('***\n')
        return delta_muW


def EQN_13(T, P, y, debug = False):
    if not debug:
        res_15 = EQN_15(T, P, y)
        res_12 = EQN_12(T, P, y)
        Pr = findPR(T)

    if debug:
        print(f'***')
        print(f'EQN - 13 @ T = {T} K and P = {P} bar')
        res_15 = EQN_15(T, P, y, debug = True)
        res_12 = EQN_12(T, y, debug = True)
        Pr = findPR(T, debug = True)


    if T >= 273:
        res = ((res_15 - res_12) / del_VWf) + Pr
        if debug:
            print(f'EQN - 13b RESULT: {res}')
            print(f'***')
        return res
    else:
        res = ((res_15 - res_12) / (del_VWAlpha)) + Pr
        if debug:
            print(f'EQN - 13a RESULT: {res}')
            print(f'***')
        return res


def EQN_13_V2(T, P, y, debug = False):
    if not debug:
        res_15 = EQN_15(T, P, y)
        res_12 = EQN_12(T, P, y)
        Pr = findPR(T)

    if debug:
        print(f'***')
        print(f'EQN - 13_V2 @ T = {T} K and P = {P} bar')
        res_15 = EQN_15(T, P, y, debug = True)
        res_12 = EQN_12(T, y, debug = True)
        Pr = findPR(T, debug = True)


    F = lambda P: res_12  + del_VWAlpha * (P - Pr) - res_15
    res = root_scalar(F, x0 = 0)

    if debug:
        print(f'EQN_13_V2 RESULT: {(res.root)}')
        print('***')

    return np.abs(res.root)


def solveBlock(T, y, threshold = 1, maxIterations = 50, debug = False):
    print("***")
    print(f'SOLVE BLOCK:')

    PGuess = 1
    print(f'Initial PGuess = Pr = {PGuess} bar')

    i = 0
    while(True):
        if debug:
            Pres = EQN_13_V2(T, PGuess, y, debug = True)
        if not debug:
            Pres = EQN_13_V2(T, PGuess, y)

        if (Pres - PGuess) ** 2 < threshold:
            if debug:
                print(f'CONVERGENCE REACHED AFTER {i} ITERATIONS. RETURNING {Pres}')
            return Pres

        else:
            PGuess = Pres
            i += 1

        if i > maxIterations:
            print(f'REACHED MAXIMUM ALLOWABLE ITERATIONS IN SOLVE BLOCK ({maxIterations}) WITHOUT CONVERGENCE')
            return


def Phi_Sensitivity(T, P_values, y, debug=False):
    phi_values = []
    V_values = []

    for P in P_values:
        try:
            V, phi = RK_EOS_FUGACITY_MIXTURE(T, P, y, debug=debug)
            V_values.append(V)
            phi_values.append(phi)
        except RuntimeError as e:
            print(f"Newton solver failed to converge at P={P} bar. Error: {e}")

    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=['Pressure vs. Phi', 'Pressure vs. V'])

    # Add traces to subplots
    fig.add_trace(go.Scatter(x=P_values, y=phi_values, mode='markers+lines', name='Pressure vs. Phi'), row=1, col=1)
    fig.add_trace(go.Scatter(x=P_values, y=V_values, mode='markers+lines', name='Pressure vs. V'), row=2, col=1)

    # Update layout
    fig.update_layout(title=f'Phi and V Sensitivity at T={T} K and y={y}',
                      xaxis_title='Pressure (bar)',
                      showlegend=False)

    fig.show()

def Temperature_Sensitivity(start_temp=260, end_temp=300, step=1, y=0.99, debug=False):
    temperature_values = np.arange(start_temp, end_temp + step, step)
    pressure_values = []

    for T in temperature_values:
        try:
            P = solveBlock(T, y, debug=debug)
            pressure_values.append(P)
        except RuntimeError as e:
            print(f"Temperature {T} K: Newton solver failed to converge. Error: {e}")

    if debug:
        print(f'PRESSURE VALUES: {pressure_values}')

    # Deaton data
    TDeaton = np.array([273.7, 274.3, 275.4, 275.9, 275.9, 277.1, 279.3, 280.4, 280.9, 281.5, 282.6, 284.3, 285.9])
    PDeaton = np.array([2.77e6, 2.9e6, 3.24e6, 3.42e6, 3.43e6, 3.81e6, 4.77e6, 5.35e6, 5.71e6, 6.06e6, 6.77e6, 8.12e6, 9.78e6])

    # Convert Deaton pressure values to bar
    PDeaton_bar = PDeaton / 1e5

    # Create plot
    fig = go.Figure()

    # Add trace to the plot
    fig.add_trace(go.Scatter(x=temperature_values, y=pressure_values, mode='markers+lines', name='Model Prediction'))
    fig.add_trace(go.Scatter(x=TDeaton, y=PDeaton_bar, mode='markers', name='Experimental Data (Deaton)'))

    # Update layout
    fig.update_layout(title=f'Temperature (260 - 300 K) vs. Pressure Sensitivity for y={y}',
                      xaxis_title='Temperature (K)',
                      yaxis_title='Pressure (bar)',
                      legend=dict(x=0, y=1, traceorder='normal'))

    fig.show()


# EQN_A1(274, 1, debug = True)
# findPR(300, debug = True)
# RK_EOS_FUGACITY_MIXTURE(290, 90, 0.99, debug = True)
# EQN_12(300, 0.99, debug = True)
# EQN_15(300, 1, 0.99, debug = True)
EQN_13(274, 10, 0.99, debug = True)
# EQN_13_V2(274, 10, 0.99, debug = True)
# solveBlock(274, 0.99, debug = True)
# Phi_Sensitivity(300, np.linspace(1, 100), 0.99, debug = True)
# SRK_EOS_FUGACITY_MIXTURE(274, 10000000, 0.99, debug = True)

# Temperature_Sensitivity(debug = True)


def Execute():
    T = np.linspace(273, 300, 27)
    resList = []
    for e in T:
        resList.append(solveBlock(e, 0.99, debug = True))

    PTrace = go.Scatter(
            x=T,  # X-axis data
            y=resList,  # Y-axis data
            mode='lines',  # 'lines' mode for a line plot
            name='',  # Name of the trace
            line=dict(color='blue', width=2)  # Line color and width
        )

    # Figure Layout
    layout = go.Layout(
        title='Pressure Vs Temp: Methane in Water: Hydrate',
        xaxis=dict(title='T (K)'),
        yaxis=dict(title='Pressure (Bar)'),
    )

    # Instantiate and Show Plotly Figure
    fig = go.Figure(
        data=[PTrace],
        layout=layout)

    fig.show()

# Execute()



