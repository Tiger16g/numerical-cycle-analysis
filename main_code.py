# Required Libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For basic plots (fallback)
from sklearn.model_selection import train_test_split  # For splitting the data
from sklearn.metrics import mean_absolute_error, mean_squared_error  # For model evaluation
from scipy.fft import fft, ifft  # For Fourier analysis
from sklearn.preprocessing import StandardScaler  # For data scaling
import xgboost as xgb  # For the XGBoost model
from numba import njit  # For speeding up computations
import plotly.express as px  # For interactive plots
import dask.dataframe as dd  # For distributed computation with large data (Optional)

# Added Functions
@njit
def calculate_years_cycle(start_year, end_year):
    """
    Calculate the 9-year cycle based on summing the odd years.
    """
    sum_odds = 0
    for year in range(start_year, end_year + 1):
        if year % 2 != 0:
            sum_odds += year
    cycle_value = sum_odds % 9
    return cycle_value

def extract_cycles(data, threshold=0.05):
    """
    Extract repeating frequencies using Fourier analysis.
    """
    transformed = fft(data)
    freq = np.abs(transformed)
    cycles = ifft(transformed)
    cycle_values = np.real(cycles)
    
    if np.max(freq) > threshold:
        return cycle_values
    else:
        return np.array(data)

def train_and_predict(X_train, X_test, y_train, y_test):
    """
    Train an XGBoost model and make predictions.
    """
    xg_reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=7, random_state=42)
    xg_reg.fit(X_train, y_train)
    predictions = xg_reg.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    
    return predictions, mae, rmse

def plot_predictions(dates, actual_values, predicted_values, title="GDP Predictions"):
    """
    Plot actual vs predicted values using Plotly for interactive visualization.
    """
    # Create the plot using Plotly
    fig = px.line(
        x=dates, 
        y=[actual_values, predicted_values], 
        labels={'x': 'Date', 'y': 'Value'}, 
        title=title,
        line_shape="linear"
    )
    
    # Add hover data for more interactivity
    fig.update_traces(mode="lines+markers", hovertemplate='Date: %{x}<br>Actual: %{y[0]}<br>Predicted: %{y[1]}')
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="GDP",
        legend_title="Legend",
        font=dict(size=12)
    )
    
    # Display the plot
    fig.show()

# Generate Dummy Data (Optional: Replace with real data if available)
np.random.seed(42)
dates = pd.date_range(start='2000-01-01', end='2020-12-31', freq='Y')
data = {
    'GDP': np.random.uniform(100, 500, size=len(dates)),  # GDP between 100 and 500
    'Unemployment': np.random.uniform(3, 10, size=len(dates)),  # Unemployment rate between 3% and 10%
    'Inflation': np.random.uniform(1, 5, size=len(dates)),  # Inflation rate between 1% and 5%
    'Interest_Rate': np.random.uniform(1, 8, size=len(dates))  # Interest rate between 1% and 8%
}

# Create DataFrame
economic_data = pd.DataFrame(data, index=dates)
economic_data.index.name = 'Date'

# Save to CSV (Optional)
economic_data.to_csv('economic_data.csv')

# Load Data using Dask (for large datasets) or Pandas (for small to medium datasets)
# Uncomment the next line for Dask if you're dealing with large datasets
# data = dd.read_csv('economic_data.csv').compute()

# Load data with Pandas for small-medium datasets
data = pd.read_csv('economic_data.csv', parse_dates=['Date'], index_col='Date')

# Clean the data (handle missing values)
data_cleaned = data.dropna()

# Extract features and target
X = data_cleaned.drop('GDP', axis=1)
y = data_cleaned['GDP']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Calculate the 9-year cycle
yearly_cycle_values = [calculate_years_cycle(year, year + 8) for year in range(data_cleaned.index.year.min(), data_cleaned.index.year.max(), 9)]
yearly_cycle_values = np.array(yearly_cycle_values)

# Extract cycles from the target data
cycle_data = extract_cycles(y)

# Combine the cycle data with the features
X_with_cycles = pd.DataFrame(cycle_data, columns=['Cycle_Values'], index=y.index)
X_with_cycles['Year_Cycle_Values'] = np.resize(yearly_cycle_values, len(y))

# Combine the scaled features with the cycle data
X_with_cycles = pd.concat([pd.DataFrame(X_scaled, columns=X.columns), X_with_cycles], axis=1)

# Train the model and make predictions
predictions, mae, rmse = train_and_predict(X_with_cycles.iloc[:len(X_train)], X_with_cycles.iloc[len(X_train):], y_train, y_test)

# Display the results
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# Plot the predictions using Plotly
plot_predictions(data_cleaned.index[len(X_train):], y_test, predictions, title="GDP Predictions")
