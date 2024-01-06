# The following calculates fugacity coefficients (Phi) for CO2 gas at a specified input temperature over a pre-defined
# range of pressures using the Peng-Robinson Equation of State (PR-EOS)

import numpy as np
from scipy.optimize import fsolve
import plotly.graph_objects as go

# Constants and critical properties of CO2
R = 8.314 * (10**-5) # m^3/Bar K
Tc = 304.1282 # K
Pc = 73.773 # Bar
W = 0.22394 # Accentric factor (unitless)

# Array of desired range of pressure values
PArray = list(np.linspace(0.1, 100, 100))

def PR_EOS(T):
    '''
    :param T: input temperature (K)
    :return: arrays of fugacity coefficients for carbon dioxide for each pressure in the PArray defined above
    '''

    global PArray

    # Tr, a, b, alpha parameters for CO2
    Tr = T/Tc
    a = (0.45724 * (R ** 2) * (Tc ** 2)) / Pc
    b = (0.0778 * R * Tc) / Pc
    alpha = (1 + ((0.37464 + (1.54226 * W) - (0.26992 * (W ** 2))) * (1 - (Tr ** 0.5))))**2

    # a, b, alpha into Peng-Robinson eqn of state (PR-EOS) defined explicitly for V, solved with root finder
    VList = []
    # SOLVE BLOCK
    for i in range(0, len(list(PArray))):
        def func(v):
            return ((R * T) / (v-b)) - ((a * alpha) / (v**2 + 2*b*v - b**2)) - PArray[i]

        v_roots = fsolve(func, np.asarray([0.01, 1]))

        VList.append(v_roots[0])


    VArray = np.asarray(VList)

    # List to hold Calculation of ln(Phi) at each pressure with corresponding V at specified T from PR-EOS
    LNPhiArray = ((PArray * VArray)/(R * T)) - 1 - np.log(((VArray - b) * PArray)/(R * T)) - ((a * alpha)/ (b * R * T * 2
                                * np.sqrt(2))) * np.log((VArray + (1 + np.sqrt(2)) * b) / (VArray + (1 - np.sqrt(2)) * b))

    PhiArray = np.exp(LNPhiArray)

    # Deletes anomalous readings from Phi and Pressure arrays prior to graphing.
    indexToDelete = np.where(PhiArray > 5000)[0]
    PhiArray = np.delete(PhiArray, indexToDelete)
    PArray = np.delete(PArray, indexToDelete)

    return PhiArray


# FUNCTION CALLS
PHI280 = PR_EOS(280)
PHI320 = PR_EOS(320)
PHI360 = PR_EOS(360)
PHI400 = PR_EOS(400)


# Graphical procedure: Phi array graphed against previously defined PArray
Trace280 = go.Scatter(
        x=PArray,  # X-axis data
        y=PHI280,  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='280 K',  # Name of the trace
        line=dict(color='blue', width=2)  # Line color and width
    )

Trace320 = go.Scatter(
        x=PArray,  # X-axis data
        y=PHI320,  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='320 K',  # Name of the trace
        line=dict(color='green', width=2)  # Line color and width
    )

Trace360 = go.Scatter(
        x=PArray,  # X-axis data
        y=PHI360,  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='360 K',  # Name of the trace
        line=dict(color='red', width=2)  # Line color and width
    )

Trace400 = go.Scatter(
        x=PArray,  # X-axis data
        y=PHI400,  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='400 K',  # Name of the trace
        line=dict(color='orange', width=2)  # Line color and width
    )

# Figure Layout
layout = go.Layout(
    title='Fugacity vs Pressure at various Temperatures for CO2',
    xaxis=dict(title='Pressure'),
    yaxis=dict(title='Fugacity Coeff.'),
)

# Instantiate and Show Plotly Figure
fig = go.Figure(
    data=[Trace280, Trace320, Trace360, Trace400],
    layout=layout)

fig.show()
