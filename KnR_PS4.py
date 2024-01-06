import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import plotly.graph_objects as go
import random


def EulerMethod():
    # Rate Constants
    k1 = 1 / 7
    k2 = .49 / 12
    k3 = .51 / 12
    k4 = 1 / 192
    k5 = 1 / 180
    k6 = (.04 ** 2) / 6
    k7 = 2

    M0 = 1
    F0 = 10
    K0 = 0

    p = [k1, k2, k3, k4, k5, k6, k7]
    w0 = [M0, F0, K0]

    tMax = 132 * 12
    t_eval = np.linspace(0, 132 * 12, 123 * 12)

    def VectorField(t, sv, *p):
        """
        :param t: time, as array of timesteps
        :param sv: State Variables (M, F, K)
        :param p: vector of parameters (rate constants)
        :return: vector of [dM/dt, dF/dt, dK/dt] in terms of defined parameters
        """
        M, F, K = sv
        k1, k2, k3, k4, k5, k6, k7 = p

        dMdt = (k2 * K) - (k4 * M)
        dFdt = (k3 * K) - (k5 * F) - 2 * (k6 * (F ** 2))
        dKdt = (2 * k1 * F) - (k2 * K) - (k3 * K) - k7

        return [dMdt, dFdt, dKdt]

    solution = solve_ivp(VectorField, (0, tMax), w0, args=p, t_eval=t_eval)

    # Access the solution
    t = solution.t
    T_values = solution.y[0]
    P_values = solution.y[1]

    results = pd.DataFrame({
        'Time': solution.t,
        'M': solution.y[0],
        'F': solution.y[1],
        'K': solution.y[2]
    })

    # Graphical output in Plotly

    MTrace = go.Scatter(
        x=results['Time'],  # X-axis data
        y=results['M'],  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='Dudes',  # Name of the trace
        line=dict(color='blue', width=2)  # Line color and width
    )

    FTrace = go.Scatter(
        x=results['Time'],  # X-axis data
        y=results['F'],  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='Ladies',  # Name of the trace
        line=dict(color='red', width=2)  # Line color and width
    )

    KTrace = go.Scatter(
        x=results['Time'],  # X-axis data
        y=results['K'],  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='Rug-Rats',  # Name of the trace
        line=dict(color='green', width=2)  # Line color and width
    )

    # Figure Layout
    layout = go.Layout(
        title='Kinetic Goat Population Model',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Goats'),
    )

    # Instantiate and Show Plotly Figure
    fig = go.Figure(
        data=[MTrace, FTrace, KTrace],
        layout=layout)

    fig.show()

def KMCMethod():

    # Rate Constants
    k1 = 1 / 7
    k2 = .49 / 12
    k3 = .51 / 12
    k4 = 1 / 192
    k5 = 1 / 180
    k6 = (.04 ** 2) / 6
    k7 = 2

    M = 1
    F = 10
    K = 0

    t = 0
    tMax = 12 * 132

    tList = []
    MList = []
    FList = []
    KList = []

    while (t < tMax):
        # Compute RXN Rates
        r1 = k1 * F
        r2 = k2 * K
        r3 = k3 * K
        r4 = k4 * M
        r5 = k5 * F
        r6 = k6 * (F**2)
        r7 = k7
        rt = r1 + r2 + r3 + r4 + r5 + r6 + r7
        # Random Number Generation
        rand1 = random.uniform(0, 1)
        rand2 = random.uniform(0, 1)

        # Determine which rxns occur
        if (rand2 <= (r1/rt)):
            # REACTION 1
            K = K+2

        elif (rand2 <= ((r1 + r2)/rt)):
            # REACTION 2
            K = K-1
            M = M+1

        elif (rand2 <= ((r1 + r2 + r3)/rt)):
            # REACTION 3
            K = K-1
            F = F+1

        elif (rand2 <= ((r1 + r2 + r3 + r4)/rt)):
            # REACTION 4
            M = M-1

        elif (rand2 <= ((r1 + r2 + r3 + r4 + r5)/rt)):
            # REACTION 5
            F = F-1

        elif (rand2 <= ((r1 + r2 + r3 + r4 + r5 + r6)/rt)):
            # REACTION 6
            F = F-2

        else:
            # REACTION 7
            K = K-1

        # Increment Time
        dt = -1/rt * np.log(rand1)
        t = t + dt
        print(t)

        tList.append(t)
        MList.append(M)
        FList.append(F)
        KList.append(K)

    results = pd.DataFrame({
        'Time': tList,
        'M': MList,
        'F': FList,
        'K': KList
    })


    # Graphical output in Plotly

    MTrace = go.Scatter(
        x=results['Time'],  # X-axis data
        y=results['M'],  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='Dudes',  # Name of the trace
        line=dict(color='blue', width=2)  # Line color and width
    )

    FTrace = go.Scatter(
        x=results['Time'],  # X-axis data
        y=results['F'],  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='Ladies',  # Name of the trace
        line=dict(color='red', width=2)  # Line color and width
    )

    KTrace = go.Scatter(
        x=results['Time'],  # X-axis data
        y=results['K'],  # Y-axis data
        mode='lines',  # 'lines' mode for a line plot
        name='Rug-Rats',  # Name of the trace
        line=dict(color='green', width=2)  # Line color and width
    )


    return MTrace, FTrace, KTrace

# KMC Graphing
# Figure Layout
layout = go.Layout(
    title='Kinetic Goat Population Model',
    xaxis=dict(title='Time'),
    yaxis=dict(title='Goats'),
)

# Instantiate and Show Plotly Figure
fig = go.Figure(
    layout=layout)

for i in range (0, 10):
    M, F, K = KMCMethod()
    fig.add_trace(M)
    fig.add_trace(F)
    fig.add_trace(K)

fig.show()

