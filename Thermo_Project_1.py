# The following simulates temperature and pressure vs time profiles derived from fundamental thermodynamic principles
# and mass/energy balances for a hypothetical airbag canister. The project was a feasibility study of whether the
# design goals were possible within the specified constraints of the problem statement.

# Design constraints and assumptions:
# 1) Canister Volume = 100 mL

# 2) Gas has the properties of N2 @ 500 K and is generated at a constant rate (allowable range of 5 - 100 mmol/s) for
#    the first 0.1 s after which no further gas is generated

# 3) The canister has a number of 20 cm^2 apertures for gas to escape into the bag. Total area of apertures cannot
#    exceed 25% of total canister surface are to maintain structural integrity.

# 4) Maximum allowable pressure = 3 bar. 30 L of gas must exit the canister within .15 sec, and 80 L must exit after 1 sec

# It was determined the pressure and release constraints could not be met while adhering to the other design constraints.

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Instantiation of Constants
R = 8.314  # L Bar/Mol K
Cp = 31.257  # J/mol K eval. at 500C, 0.101 MPa
Cv = 22.939  # J/mol K eval. at 500C, 0.101 MPa
Hin = 22876 # J/mol eval. at 500C, 0.101 MPa
V = 0.0001  # m^3
A = 0.05  # m^2
h0 = 1  # W/m^2 K
CStar = 0.8 * 0.002 * 0.5  # Mol K/sec Pa m^2
gasReleaseRate = 100  # mol/sec
nA = 6  # Num Apertures

# Initial Conditions
P0 = 101000  # Pa
T0 = 300  # K

# Timesteps
tMax = 1  # sec
timeSteps = 200
timeStep = tMax/timeSteps # sec
t_eval = np.linspace(0, tMax, timeSteps)

# Vectors of initial conditions and parameters
p = [R, Cp, Cv, V, A, h0, CStar, gasReleaseRate, nA, Hin]
w0 = [T0, P0]

# Defines gas generation as stepwise function of time t in the simulation
def RGEN(t):
    if t < 0.1:
        return gasReleaseRate
    else:
        return 0


def VectorField(t, sv, *p):
    """
    :param t: time, as array of timesteps
    :param sv: State Variables (T & P)
    :param p: vector of parameters (constants)
    :return: vector of [dT/dt, dP/dt] in terms of defined parameters
    """

    # Unpacking initial conditions
    T, P = sv
    R, Cp, Cv, V, A, h0, CStar, gasReleaseRate, nA, Hin = p

    # Temperature and Pressure vs time profiles as ODEs
    dTdt = (T/P) * (R/Cv) * (h0 * A/V) * (T0-T) + (T/P) * (R/Cv) * (RGEN(t)/V) * (Hin - Cv * T) - (T/P) * (R/Cv) * (R * nA * CStar/V) * (P-P0)
    dPdt = (R/Cv) * (h0 * A/V) * (T0-T) + (R/V) * (Hin/Cv) * RGEN(t) - (R/V) * (Cp/Cv) * nA * CStar * (P-P0)

    return [dTdt, dPdt]

def VolumeReleased(T, P):
    vr = nA * CStar * timeStep * ((P-P0)/T) # moles gas released per time step

    return vr

# Solve the ODE system using solve_ivp, passing in VectorField, timespan, vector of initial conditions, and array of timesteps
solution = solve_ivp(VectorField, (0, tMax), w0, args=p, t_eval=t_eval)

# Access the solution and convert to dataframe
t = solution.t
T_values = solution.y[0]
P_values = solution.y[1]

results = pd.DataFrame({
    'Time': solution.t,
    'Temperature': solution.y[0],
    'Pressure': solution.y[1]
})

# Data Visualization:

# Create separate plots
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Temperature (K) vs. Time", "Pressure (Pa) vs. Time"))

# Add Temperature vs. Time subplot
fig.add_trace(go.Scatter(x=results['Time'], y=results['Temperature'], mode='lines', name='Temperature'), row=1, col=1)
fig.update_yaxes(title_text='Temperature', row=1, col=1)

# Add Pressure vs. Time subplot
fig.add_trace(go.Scatter(x=results['Time'], y=results['Pressure'], mode='lines', name='Pressure'), row=2, col=1)
fig.update_yaxes(title_text='Pressure ', row=2, col=1)
fig.update_xaxes(title_text='Time (s)', row=2, col=1)


# Show the figure
fig.show()



# Create a list to store the volume released at each time step
vrList = []

# Calculate the volume released at each time step and store the cumulative volume
cumulativeVolume = 0
for i in range(len(t)):
    T_current = T_values[i]
    P_current = P_values[i]
    vr = VolumeReleased(T_current, P_current)
    cumulativeVolume += vr
    vrList.append(22.4 * cumulativeVolume) # converts moles to L

# Add the 'Volume Released' column to the dataframe
results['Cumulative Volume Released (L)'] = vrList

# Print the updated dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(results)