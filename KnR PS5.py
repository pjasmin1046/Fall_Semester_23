import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
import pandas as pd


resTime = 100
R = 1.987 # cal/mol K

# PROBLEM 2

# B, ii

def BI(debug = False):
    selectivity = []

    k1 = (10**4) * np.exp(-(10**4) / (R * 300))
    k2 = (10**5) * np.exp(-(10**3) / (R * 300))
    Ca0 = 1.831 # mol/L
    Cc0 = (k1 * Ca0) / (-k1 + k2)
    if debug:
        print(f'k1: {k1}\n')
        print(f'k2: {k2}\n')
        print(f'Cc0: {Cc0}\n')


    Cc = lambda t: (Cc0 * np.exp(-k2 * t)) - (((k1 * Ca0) / (-k1 + k2)) * np.exp(-k1 * t))
    Cd = lambda t: (-Cc0 / k2) * np.exp(-k2 * t) + (Ca0 / (-k1 + k2)) * np.exp(-k1 * t) + (Cc0 / k2) - (Ca0 / (-k1 + k2))

    for i in range (1, resTime):
        if debug:
            print(f'Cc({i}): {Cc(i)}')
            print(f'Cd({i}): {Cd(i)}')
            print(f'Sc({i}): {Cc(i) / (Cc(i) + Cd(i))}\n')

        selectivity.append(Cc(i) / (Cc(i) + Cd(i)))

    # Graphical Output
    ScTrace = go.Scatter(
            x=np.linspace(0, resTime, len(selectivity)),  # X-axis data
            y=selectivity,  # Y-axis data
            mode='lines',  # 'lines' mode for a line plot
            name='Sc',  # Name of the trace
            line=dict(color='blue', width=2)  # Line color and width
        )

    # Figure Layout
    layout = go.Layout(
        title='Selectivity C',
        xaxis=dict(title='Time (t)'),
        yaxis=dict(title='Sc'),
    )

    # Instantiate and Show Plotly Figure
    fig = go.Figure(
        data=[ScTrace],
        layout=layout)

    fig.show()

def BII(debug = False):
    Ca0 = 1.831 # mol/L
    Cc0 = 0
    Cd0 = 0
    T0 = 300 # K
    density = 1
    Cp = 16.8 # cal/mol

    p = [R, Cp, density]
    w0 = [Ca0, Cc0, Cd0, T0]

    t_eval = np.linspace(1, resTime)

    def VectorField(t, sv, *p):
        """
        :param t: time, as array of timesteps
        :param sv: State Variables (CA, CC, CD, T)
        :param p: vector of parameters (constants)
        :return: vector of [dCA/dt, dCC/dt, dCC/dt, dT/dt] in terms of defined parameters
        """
        Ca, Cc, Cd, T = sv
        # print(f'Ca: {Ca}\n Cc: {Cc}\n Cd: {Cd}\n T: {T}\n')
        R, Cp, density = p

        dCadt = Ca * ((10**4) * np.exp((-(10**4))/(R * T)))
        dCcdt = (Ca * ((10**4) * np.exp((-(10**4))/(R * T)))) - (Cc * ((10**5) * np.exp((-(10**3))/(R * T))))
        dCddt = (Cc * ((10**5) * np.exp((-(10**3))/(R * T))))
        dTdt = (4000 * (Ca * ((10**4) * np.exp((-(10**4))/(R * T))))) - (42000 * (Cc * ((10**5) * np.exp((-(10**3))/(R * T)))))

        return [dCadt, dCcdt, dCddt, dTdt]

    solution = solve_ivp(VectorField, (0, resTime), w0, args=p, t_eval=t_eval)

    # Access the solution
    t = solution.t
    CaCalc = solution.y[0]
    CcCalc = solution.y[1]
    CdCalc = solution.y[2]
    TCalc = solution.y[3]


    results = pd.DataFrame({
        'Time': solution.t,
        'Ca': solution.y[0],
        'Cc': solution.y[1],
        'Cd': solution.y[2],
        'T': solution.y[3]
    })

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(results)

    # Create separate plots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Temperature vs. Time", "Species Conc. vs. Time"))

    # Add Temperature vs. Time subplot
    fig.add_trace(go.Scatter(x=results['Time'], y=results['T'], mode='lines', name='Temperature'), row=1, col=1)
    fig.update_yaxes(title_text='Temperature', row=1, col=1)

    # Add Conc. vs. Time subplot
    fig.add_trace(go.Scatter(x=results['Time'], y=results['Ca'], mode='lines', name='Ca'), row=2, col=1)
    fig.update_yaxes(title_text='Ca', row=2, col=1)

    fig.add_trace(go.Scatter(x=results['Time'], y=results['Cc'], mode='lines', name='Cc'), row=2, col=1)
    fig.update_yaxes(title_text='Cc', row=2, col=1)

    fig.add_trace(go.Scatter(x=results['Time'], y=results['Cd'], mode='lines', name='Cd'), row=2, col=1)
    fig.update_yaxes(title_text='Cd', row=2, col=1)

    # Update layout for better visualization
    fig.update_layout(title='Temperature and Species Conc. vs. Time')

    # Show the figure
    fig.show()


# BI(debug = True)
BII()