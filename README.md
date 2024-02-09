# Option_Greeks_visualizer

This repository contains Python code for visualizing Option Greeks and payoffs using the Black-Scholes model. It leverages libraries such as NumPy for numerical computations, SciPy for statistical functions, Matplotlib for static plotting, Plotly for interactive visualizations, and Streamlit for web app creation.

## Features

- Calculation of Call and Put prices using the Black-Scholes model.
- Visualization of Option Greeks (Delta, Gamma, Vega, Theta, Rho, and Vanna) across different underlying prices.
- Interactive web app interface using Streamlit for parameter inputs and real-time visualization.
- Plotting of payoff diagrams for Call and Put options.

## Installation

To run this project, you need to have Python installed on your system. It's recommended to use a virtual environment. Follow these steps to set up the environment and run the app:

```bash
# Clone the repository
git clone https://github.com/your-github-username/option-greeks-visualizer.git

# Navigate to the repository directory
cd option-greeks-visualizer

# (Optional) Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate

# Install the required packages
pip install numpy scipy matplotlib plotly streamlit

# Run the Streamlit app
streamlit run app.py
```

# Usage
After starting the Streamlit app, use the sidebar to input the parameters for the Option model:

Underlying Price (S)
Strike Price (K)
Time to Maturity (T in years)
Risk-Free Rate (r)
Dividend Yield (d)
Volatility (sigma)

Click on the "Calculate Greeks & Payoffs" button to view the calculated Option Greeks and payoff diagrams. The results include both Call and Put option prices, Delta, Gamma, Vega, Theta, and Rho for different underlying prices.

# Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change. Please ensure to update tests as appropriate.

# License
MIT
