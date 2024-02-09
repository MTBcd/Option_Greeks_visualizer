from math import log, sqrt, exp
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots



class BSPricing:
    def __init__(self, S, K, T, r, d, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.d = d
        self.sigma = sigma

    def params(self):
        return {'S' : self.S,
                'K' : self.K,
                'T' : self.T,
                'r' : self.r,
                'd' : self.d,
                'sigma' : self.sigma}

    def N(self, D):
        return norm.cdf(D)

    def n(self, D):
        return norm.pdf(D)

    def D1(self):
        D1 = (log(self. S /self.K) + (self.r - self.d + self.sigma**2 / 2) * self.T) / (self.sigma * sqrt(self.T))
        return D1

    def D2(self):
        D2 = self.D1() - self.sigma * sqrt(self.T)
        return D2

    def Call(self):
        Call = self.S * exp(-self.d * self.T) * self.N(self.D1()) - self.K * exp(-self.r * self.T) * self.N(self.D2())
        return Call

    def Put(self):
        Put = self.K * exp(-self.r * self.T) * self.N(self.D2()) - self.S * exp(-self.d * self.T) * self.N(self.D1())
        return Put


class Greeks(BSPricing):
    def __init__(self, S, K, T, r, d, sigma):
        super().__init__(S, K, T, r, d, sigma)  # Use super() for a cleaner call to the parent class constructor

    def DeltaCall(self):
        # Use self.D1() and self.N() to access the inherited methods
        DeltaCall = exp(-self.d * self.T) * self.N(self.D1())
        return DeltaCall

    def DeltaPut(self):
        DeltaPut = -exp(-self.d * self.T) * self.N(self.D1())
        return DeltaPut

    def GammaCall(self):
        GammaCall = (exp(-self.d * self.T) / (self.S * self.sigma * sqrt(self.T))) * self.n(self.D1())
        return GammaCall

    def GammaPut(self):
        GammaPut = (exp(-self.d * self.T) / (self.S * self.sigma * sqrt(self.T))) * self.n(self.D1())
        return GammaPut

    def VegaCall(self):
        VegaCall = self.S * exp(-self.d * self.T) * self.n(self.D1()) * sqrt(self.T)
        return VegaCall / 100

    def VegaPut(self):
        VegaPut = self.S * exp(-self.d * self.T) * self.n(self.D1()) * sqrt(self.T)
        return VegaPut / 100

    def ThetaCall(self):
        ThetaCall = (-(self.S * self.sigma * exp(-self.d * self.T) * self.n(self.D1())) /
                    (2 * sqrt(self.T))) - self.r * self.K * exp(-self.r * self.T) * self.N(
            self.D2()) + self.d * self.S * exp(-self.d * self.T) * self.N(self.D1())
        return ThetaCall

    def ThetaCall(self):
        partial_time = -(self.S * self.sigma * exp(-self.d * self.T) * self.n(self.D1())) / (2 * sqrt(self.T))
        partial_interest = -self.r * self.K * exp(-self.r * self.T) * self.N(self.D2())
        partial_dividend = self.d * self.S * exp(-self.d * self.T) * self.N(self.D1())
        ThetaCall = partial_time + partial_interest + partial_dividend
        return ThetaCall / 254

    def ThetaPut(self):
        partial_time = -(self.S * self.sigma * exp(-self.d * self.T) * self.n(self.D1())) / (2 * sqrt(self.T))
        partial_interest = self.r * self.K * exp(-self.r * self.T) * self.N(self.D2())
        partial_dividend = -self.d * self.S * exp(-self.d * self.T) * self.N(self.D1())
        ThetaPut = partial_time + partial_interest + partial_dividend
        return ThetaPut / 254

    def RhoCall(self):
        RhoCall = self.K * self.T * exp(-self.r * self.T) * self.N(self.D2())
        return RhoCall / 100

    def RhoPut(self):
        RhoPut = self.K * self.T * exp(-self.r * self.T) * self.N(-self.D2())
        return RhoPut / 100

    def VannaCall(self):
        VannaCall = exp(-self.d * self.T) * self.n(self.D1()) * (1 - self.D1() / (self.sigma * sqrt(self.T)))
        return VannaCall

    def VannaPut(self):
        VannaPut = exp(-self.d * self.T) * self.n(self.D1()) * (1 - self.D1() / (self.sigma * sqrt(self.T)))
        return VannaPut

    def plot_payoffs(self):
        # Utilize np.linspace for generating S_range
        S_range = np.linspace(self.S * 0.8, self.S * 1.2, 100)

        # Calculate payoffs
        call_payoffs = [max(S - self.K, 0) for S in S_range]
        put_payoffs = [max(self.K - S, 0) for S in S_range]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(S_range, call_payoffs, label='Call Payoff')
        plt.plot(S_range, put_payoffs, label='Put Payoff')
        plt.xlabel('Underlying Price (S)')
        plt.ylabel('Payoff')
        plt.title('Option Payoffs')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_greeks(self):
        # Generate a range of underlying prices
        S_range = np.linspace(self.S * 0.8, self.S * 1.2, 100)

        # Temporarily store original stock price
        original_S = self.S

        # Calculate Greeks for each price in the range
        deltas, gammas, vegas, thetas, rhos, vannas = [], [], [], [], [], []
        for S in S_range:
            self.S = S  # Update stock price for calculations
            deltas.append(self.DeltaCall())
            gammas.append(self.GammaCall())
            vegas.append(self.VegaCall())
            thetas.append(self.ThetaCall())
            rhos.append(self.RhoCall())
            vannas.append(self.VannaCall())

        # Reset stock price to original
        self.S = original_S

        # Creating subplots
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))  # Adjust the size as needed
        fig.suptitle('Greeks for Call Option Across Different Underlying Prices')

        # Plotting each Greek in a subplot
        axs[0, 0].plot(S_range, deltas, label='Delta')
        axs[0, 0].set_title('Delta')
        axs[0, 0].grid(True)

        axs[0, 1].plot(S_range, gammas, label='Gamma')
        axs[0, 1].set_title('Gamma')
        axs[0, 1].grid(True)

        axs[1, 0].plot(S_range, vegas, label='Vega')
        axs[1, 0].set_title('Vega')
        axs[1, 0].grid(True)

        axs[1, 1].plot(S_range, thetas, label='Theta')
        axs[1, 1].set_title('Theta')
        axs[1, 1].grid(True)

        axs[2, 0].plot(S_range, rhos, label='Rho')
        axs[2, 0].set_title('Rho')
        axs[2, 0].grid(True)

        axs[2, 1].plot(S_range, vannas, label='Vanna')
        axs[2, 1].set_title('Vanna')
        axs[2, 1].grid(True)

        # Hide empty subplot (if any)
        for ax in axs.flat:
            ax.label_outer()
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the main title
        plt.show()


def plot_greeks_call_with_plotly(option):
    # Define the range of underlying prices for plotting
    S_range = np.linspace(option.S * 0.7, option.S * 1.3, 200)

    # Initialize a 3x2 subplot figure
    fig = make_subplots(rows=3, cols=2, subplot_titles=('Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'Vanna'))

    # Store the original value of S
    original_S = option.S

    # Calculate and plot each Greek
    for i, (row, col) in enumerate([(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)], start=1):
        greek_values = []
        for S in S_range:
            option.S = S  # Temporarily set the stock price to S

            # Dynamically call the Greek method based on i
            if i == 1:
                greek_values.append(option.DeltaCall())
            elif i == 2:
                greek_values.append(option.GammaCall())
            elif i == 3:
                greek_values.append(option.VegaCall())
            elif i == 4:
                greek_values.append(option.ThetaCall())
            elif i == 5:
                greek_values.append(option.RhoCall())
            elif i == 6:
                greek_values.append(option.VannaCall())

        # Reset the stock price to the original value
        option.S = original_S

        # Add the calculated Greek values to the subplot
        fig.add_trace(go.Scatter(x=S_range, y=greek_values, mode='lines',
                                 name=['Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'Vanna'][i - 1]), row=row, col=col)

    # Update layout
    fig.update_layout(height=900, width=800, title_text="Greeks Visualization Across Different Underlying Prices")
    return fig

def plot_greeks_put_with_plotly(option):
    # Define the range of underlying prices for plotting
    S_range = np.linspace(option.S * 0.7, option.S * 1.3, 200)

    # Initialize a 3x2 subplot figure
    fig = make_subplots(rows=3, cols=2, subplot_titles=('Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'Vanna'))

    # Store the original value of S
    original_S = option.S

    # Calculate and plot each Greek
    for i, (row, col) in enumerate([(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)], start=1):
        greek_values = []
        for S in S_range:
            option.S = S  # Temporarily set the stock price to S

            # Dynamically call the Greek method based on i
            if i == 1:
                greek_values.append(option.DeltaPut())
            elif i == 2:
                greek_values.append(option.GammaPut())
            elif i == 3:
                greek_values.append(option.VegaPut())
            elif i == 4:
                greek_values.append(option.ThetaPut())
            elif i == 5:
                greek_values.append(option.RhoPut())
            elif i == 6:
                greek_values.append(option.VannaPut())

        # Reset the stock price to the original value
        option.S = original_S

        # Add the calculated Greek values to the subplot
        fig.add_trace(go.Scatter(x=S_range, y=greek_values, mode='lines',
                                 name=['Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'Vanna'][i - 1]), row=row, col=col)

    # Update layout
    fig.update_layout(height=900, width=800, title_text="Greeks Visualization Across Different Underlying Prices")
    return fig

def plot_payoffs_with_plotly(option):
    # Generate a range of underlying prices
    S_range = np.linspace(option.S * 0.7, option.S * 1.3, 200)

    # Calculate payoffs
    call_payoffs = np.maximum(S_range - option.K, 0)
    put_payoffs = np.maximum(option.K - S_range, 0)

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=call_payoffs, mode='lines', name='Call Payoff'))
    fig.add_trace(go.Scatter(x=S_range, y=put_payoffs, mode='lines', name='Put Payoff'))

    fig.update_layout(title='Option Payoff Diagrams', xaxis_title='Underlying Price (S)', yaxis_title='Payoff', legend_title='Option Type')
    return fig

# Streamlit app layout starts here
st.title('Option Greeks Visualizer')

with st.sidebar:
    st.header("Option Parameters")
    S = st.number_input('Underlying Price (S)', min_value=1.0, max_value=500.0, value=100.0)
    K = st.number_input('Strike Price (K)', min_value=1.0, max_value=500.0, value=100.0)
    T = st.number_input('Time to Maturity (T in years)', min_value=0.01, max_value=5.0, value=1.0)
    r = st.number_input('Risk-Free Rate (r)', min_value=0.0, max_value=0.3, value=0.05)
    d = st.number_input('Dividend Yield (d)', min_value=0.0, max_value=0.3, value=0.0)
    sigma = st.number_input('Volatility (sigma)', min_value=0.01, max_value=1.0, value=0.2)

option = Greeks(S, K, T, r, d, sigma)

calculate = st.button('Calculate Greeks & Payoffs')

if calculate:
    with st.container():
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Call Price", f"{option.Call():.2f}")
        with col2:
            st.metric("Delta", f"{option.DeltaCall():.3f}")
        with col3:
            st.metric("Gamma", f"{option.GammaCall():.3f}")
        with col4:
            st.metric("Vega", f"{option.VegaCall():.3f}")
        with col5:
            st.metric("Theta", f"{option.ThetaCall():.3f}")
        with col6:
            st.metric("Rho", f"{option.RhoCall():.3f}")
#        with col7:
#            st.metric("Vanna", f"{option.VannaCall():.4f}")
#            st.metric("Vanna", f"{option.VannaPut():.4f}")

    greeks_fig = plot_greeks_call_with_plotly(option)
    st.plotly_chart(greeks_fig, use_container_width=True)

    with st.container():
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Put Price", f"{option.Put():.2f}")
        with col2:
            st.metric("Delta", f"{option.DeltaPut():.3f}")
        with col3:
            st.metric("Gamma", f"{option.GammaPut():.3f}")
        with col4:
            st.metric("Vega", f"{option.VegaPut():.3f}")
        with col5:
            st.metric("Theta", f"{option.ThetaPut():.3f}")
        with col6:
            st.metric("Rho", f"{option.RhoPut():.3f}")

    greeks_fig = plot_greeks_put_with_plotly(option)
    st.plotly_chart(greeks_fig, use_container_width=True)

    # Plotting the payoffs
    payoffs_fig = plot_payoffs_with_plotly(option)
    st.plotly_chart(payoffs_fig, use_container_width=True)
