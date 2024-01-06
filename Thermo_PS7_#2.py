# The following calculates fugacity coefficients (Phi) for CO2/H2O mixture gas at a specified input temperature over a pre-defined
# range of pressures using the Peng-Robinson Equation of State (PR-EOS)

import numpy as np
from scipy.optimize import fsolve
import plotly.graph_objects as go

# Constants
R = 8.314 * (10**-5) # m^3/Bar K

# Critical Properties [Tc: K, Pc: Bar, W (Accentric Factor: Unitless), Vc: m^3/kg]
CO2_crit = [304.1812, 73.773, 0.22394, 0.00214]
H2O_crit = [647, 220.64, 0.3433, 0.003155]

# Array of desired range of pressure values
PArray = list(np.linspace(0.1, 100, 100))

def PR_EOS_MIXTURE(T, debug = False):
    '''
    :param T: input temperature (K)
    :param debug: optional parameter to print inputs and both intermediate and final results to console
    :return: arrays of fugacity coefficients for water and carbon dioxide for each pressure in the PArray defined above
    '''

    global PArray

    # Tr, a, b, alpha parameters for each species
    Tr_H2O = T/H2O_crit[0]
    a_H2O = (0.45724 * (R**2) * H2O_crit[0])/ H2O_crit[1]
    b_H2O = (0.078 * R * H2O_crit[0])/ H2O_crit[1]
    alpha_H2O = (1 + ((0.37464 + (1.54226 * H2O_crit[2]) - (0.26992 * (H2O_crit[2] ** 2))) * (1 - (Tr_H2O ** 0.5))))**2

    Tr_CO2 = T / CO2_crit[0]
    a_CO2 = (0.45724 * (R**2) * CO2_crit[0])/ CO2_crit[1]
    b_CO2 = (0.078 * R * CO2_crit[0])/ CO2_crit[1]
    alpha_CO2 = (1 + ((0.37464 + (1.54226 * CO2_crit[2]) - (0.26992 * (CO2_crit[2] ** 2))) * (1 - (Tr_CO2 ** 0.5)))) ** 2

    if debug:
        print(f'Tr_H2O: {Tr_H2O}')
        print(f'a_H2O: {a_H2O}')
        print(f'b_H2O: {b_H2O}')
        print(f'alpha_H2O: {alpha_H2O}\n')
        print(f'Tr_CO2: {Tr_CO2}')
        print(f'a_CO2: {a_CO2}')
        print(f'b_CO2: {b_CO2}')
        print(f'alpha_CO2: {alpha_CO2}\n')

    # Compressibility factor (Zc) for each species
    Zc_H2O = (H2O_crit[1] * H2O_crit[3]) / (R * H2O_crit[0])
    Zc_CO2 = (CO2_crit[1] * CO2_crit[3]) / (R * CO2_crit[0])

    if debug:
        print(f'Zc_H2O: {Zc_H2O}')
        print(f'Zc_CO2: {Zc_CO2}')

    # Calculation of mixture properties: bm, Kij, am
    bm = (.5 * b_H2O) + (.5 * b_CO2)
    Kij = 1 - ((2 * np.sqrt(H2O_crit[0] * CO2_crit[0]))/(H2O_crit[0] + CO2_crit[0]))**((Zc_H2O + Zc_CO2)/2)
    am = (.5 ** 2) * ((a_H2O * a_CO2) ** .5) * (1 - Kij)

    if debug:
        print(f'bm: {bm}')
        print(f'Kij: {Kij}')
        print(f'am: {am}')

    # am, bm, into Peng-Robinson eqn of state (PR-EOS) defined explicitly for Vm, solved with root finder
    VmList = []
    # SOLVE BLOCK
    if debug:
        print("\nVm LOOP/ SOLVE BLOCK")

    for i in range(0, len(list(PArray))):
        def func(v):
            return ((R * T) / (v-bm)) - (am / (v**2 + 2*bm*v - bm**2)) - PArray[i]

        v_roots = fsolve(func, np.asarray([0.01, 1]))
        VmList.append(v_roots[0])

        if debug:
            print(f'Vm[{i}]: {v_roots}')

    VmArray = np.asarray(VmList)

    # Lists to hold Calculation of ln(Phi) at each pressure with corresponding Vm at specified T from PR-EOS for both gaseous components
    LNPhiList_H2O = []
    LNPhiList_CO2 = []
    if debug:
        print("\nLN PHI LOOP:")

    for i in range(0, len(PArray)):
        # A, B, Z parameters for current pressure
        A = (am * PArray[i])/((R * T) ** 2)
        B = (bm * PArray[i])/(R * T)
        Z = (PArray[i] * VmArray[i])/(R * T)

        # ln(Phi) for each species at current pressure
        LNPhiH2O = ((b_H2O / bm) * (Z - 1)) - np.log(Z - B) + ((A / (np.sqrt(8) * B)) * ((2 * (.5 * a_H2O + .5 * a_CO2) / am)
                                    - (b_H2O / bm)) * (np.log((Z + (B * (1 - np.sqrt(2)))) / (Z + (B * (1 + np.sqrt(2)))))))

        LNPhiCO2 = ((b_CO2 / bm) * (Z - 1)) - np.log(Z - B) + ((A / (np.sqrt(8) * B)) * ((2 * (.5 * a_H2O + .5 * a_CO2) / am)
                                    - (b_CO2 / bm)) * (np.log((Z + (B * (1 - np.sqrt(2)))) / (Z + (B * (1 + np.sqrt(2)))))))

        LNPhiList_CO2.append(LNPhiCO2)
        LNPhiList_H2O.append(LNPhiH2O)

        if debug:
            print(f'\nLN_Phi_H2O[{i}]: {LNPhiH2O}')
            print(f'LN_Phi_CO2[{i}]: {LNPhiCO2}')

    # Exponentiate to convert to arrays of Phi for each species
    Phi_H2OArray = np.exp(np.asarray(LNPhiList_H2O))
    Phi_CO2Array = np.exp(np.asarray(LNPhiList_CO2))

    if debug:
        print(f'\nH2O Phi Array: {Phi_H2OArray}\n')
        print(f'CO2 Phi Array: {Phi_CO2Array}\n')

    # Remove anomalous readings
    indexToDeleteH2O = np.where(Phi_H2OArray > 5000)[0]
    Phi_H2OArray = np.delete(Phi_H2OArray, indexToDeleteH2O)
    PArray = np.delete(PArray, indexToDeleteH2O)
    Phi_CO2Array = np.delete(Phi_CO2Array, indexToDeleteH2O)

    indexToDeleteCO2 = np.where(Phi_CO2Array > 5000)[0]
    Phi_CO2Array = np.delete(Phi_CO2Array, indexToDeleteCO2)
    PArray = np.delete(PArray, indexToDeleteCO2)
    Phi_H2OArray = np.delete(Phi_H2OArray, indexToDeleteCO2)

    return Phi_H2OArray, Phi_CO2Array


# FUNCTION CALLS
H2O_280, CO2_280 = PR_EOS_MIXTURE(280, debug = False)
H2O_320, CO2_320 = PR_EOS_MIXTURE(320, debug = False)
H2O_360, CO2_360 = PR_EOS_MIXTURE(360, debug = False)
H2O_400, CO2_400 = PR_EOS_MIXTURE(400, debug = False)


# Graphing procedure: Both Phi arrays graphed against PArray, H2O is solid line, CO2 is dashed
H2OTrace280 = go.Scatter(
        x=PArray,  # X-axis data
        y=H2O_280,  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='H2O @ 280 K',  # Name of the trace
        line=dict(color='blue', width=2)  # Line color and width
    )

CO2Trace280 = go.Scatter(
        x=PArray,  # X-axis data
        y=CO2_280,  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='CO2 @ 280 K',  # Name of the trace
        line=dict(color='blue', width=2, dash = 'dash')  # Line color and width
    )


H2OTrace320 = go.Scatter(
        x=PArray,  # X-axis data
        y=H2O_320,  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='H2O @ 320 K',  # Name of the trace
        line=dict(color='black', width=2)  # Line color and width
    )

CO2Trace320 = go.Scatter(
        x=PArray,  # X-axis data
        y=CO2_320,  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='CO2 @ 320 K',  # Name of the trace
        line=dict(color='black', width=2, dash = 'dash')  # Line color and width
    )


H2OTrace360 = go.Scatter(
        x=PArray,  # X-axis data
        y=H2O_360,  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='H2O @ 360 K',  # Name of the trace
        line=dict(color='red', width=2)  # Line color and width
    )

CO2Trace360 = go.Scatter(
        x=PArray,  # X-axis data
        y=CO2_360,  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='CO2 @ 360 K',  # Name of the trace
        line=dict(color='red', width=2, dash = 'dash')  # Line color and width
    )


H2OTrace400 = go.Scatter(
        x=PArray,  # X-axis data
        y=H2O_400,  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='H2O @ 400 K',  # Name of the trace
        line=dict(color='green', width=2)  # Line color and width
    )

CO2Trace400 = go.Scatter(
        x=PArray,  # X-axis data
        y=CO2_400,  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='CO2 @ 400 K',  # Name of the trace
        line=dict(color='green', width=2, dash = 'dash')  # Line color and width
    )

# Figure Layout
layout = go.Layout(
    title='Fugacity vs Pressure at various Temperatures for CO2-H2O Mixture',
    xaxis=dict(title='Pressure'),
    yaxis=dict(title='Fugacity Coeff.'),
)

# Instantiate and Show Plotly Figure
fig = go.Figure(
    data=[CO2Trace280, H2OTrace280, CO2Trace320, H2OTrace320, CO2Trace360, H2OTrace360, CO2Trace400, H2OTrace400],
    layout=layout)

fig.show()






