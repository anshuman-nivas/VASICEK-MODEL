import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Cleaning the RBI interest rate dataset
file_path = "Interest Rates on Central and State Government Dated securities.csv"
data = pd.read_csv(file_path, skiprows=5)
data.columns = ['Index', 'Year', 'Central_Range',
                'Central_Average', 'State_Range', 'State_Average']
interest_rates = data[['Year', 'Central_Average', 'State_Average']]
interest_rates = interest_rates.dropna()
interest_rates = interest_rates.drop(index=2)
interest_rates = interest_rates.reset_index(drop=True)

# Converting interest rate columns to numeric
interest_rates['Year'] = interest_rates['Year'].str.strip()
interest_rates['Central_Average'] = pd.to_numeric(
    interest_rates['Central_Average'])
interest_rates['State_Average'] = pd.to_numeric(
    interest_rates['State_Average'])

# Function to calculate Vasicek log-likelihood


def vasicek_log_likelihood(params, rates):
    kappa, theta, sigma = params
    r0 = rates[0]
    n = len(rates)
    dt = 1  # Assuming yearly data

    likelihood = 0
    for i in range(1, n):
        r_prev = rates[i-1]
        r_curr = rates[i]
        mu = r_prev + kappa * (theta - r_prev) * dt
        var = sigma**2 * dt

        if var == 0:
            var += 1e-6  # Add a small value to avoid division by zero

        likelihood += -0.5 * np.log(2 * np.pi * var) - \
            (r_curr - mu)**2 / (2 * var)

    return -likelihood

# Function to optimize and get Vasicek parameters


def get_vasicek_params(rates):
    initial_params = [0.1, np.mean(rates), np.std(rates)]
    result = minimize(vasicek_log_likelihood, initial_params, args=(
        rates,), bounds=((0, None), (None, None), (0, None)))
    return result.x

# Function to simulate future rates using the Vasicek model


def vasicek_simulation(r0, kappa, theta, sigma, T=120, N=120):
    dt = T / float(N)
    t = np.linspace(0, T, N+1)
    rates = [r0]

    for _ in range(N):
        dr = kappa * (theta - rates[-1]) * dt + \
            sigma * np.sqrt(dt) * np.random.normal()
        rates.append(rates[-1] + dr)

    return t, rates


# Analyze both Central and State interest rates
for gov_type in ['Central', 'State']:
    rates = interest_rates[f'{gov_type}_Average'].values
    kappa, theta, sigma = get_vasicek_params(rates)
    print(f'Estimated parameters for {gov_type} Government: kappa = {
          kappa}, theta = {theta}, sigma = {sigma}')

    # Initial interest rate for simulation (last observed rate in dataset)
    r0 = rates[-1]

    # Simulate future rates
    time, future_rates = vasicek_simulation(r0, kappa, theta, sigma)

    # Starting year for the simulation (last year in the dataset)
    start_year_str = interest_rates['Year'].values[-1]
    start_year = int(start_year_str.split('-')[0])  # Extract the start year

    # Generate future years
    years = np.arange(start_year, start_year + len(time))

    # Plot the future rates with year labels
    plt.plot(years, future_rates, label=f'{gov_type} Government')

# Set x-axis ticks to every 5 years
plt.xticks(np.arange(start_year, start_year + 120 + 1, 5))

plt.xlabel('Year')
plt.ylabel('Interest Rate r(t)')
plt.title('Vasicek Model - 100 Year Prediction for Central and State Government')
plt.legend()
plt.show()
