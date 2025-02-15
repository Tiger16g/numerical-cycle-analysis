import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.fft import fft, ifft
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Loading the data (Example of economic data)
data = pd.read_csv('economic_data.csv', index_col='Date', parse_dates=True)

# Cleaning the data (Removing missing values)
data_cleaned = data.dropna()

# Extracting GDP data
X = data_cleaned.drop('GDP', axis=1)
y = data_cleaned['GDP']

# Scaling the data using normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Adding the 9-year cycle (Summing the odd years)
def calculate_years_cycle(start_year, end_year):
    """Calculate the 9-year cycle based on summing the odd years"""
    years = list(range(start_year, end_year + 1))
    sum_odds = sum(year for year in years if year % 2 != 0)
    cycle_value = sum_odds % 9  # Modulo 9 to give a result between 0 and 8
    return cycle_value

# Extracting repeating cycles using Fourier analysis
def extract_cycles(data, threshold=0.05):
    """Extract repeating frequencies using Fourier analysis"""
    transformed = fft(data)
    freq = np.abs(transformed)
    cycles = ifft(transformed)
    cycle_values = np.real(cycles)
    
    # Filtering frequencies below the threshold (optional)
    if np.max(freq) > threshold:
        return cycle_values
    else:
        return data  # If no strong cycle is extracted, return the original data

# Merging the 9-year cycle with the data
yearly_cycle_values = [calculate_years_cycle(year, year + 8) for year in range(data_cleaned.index.year.min(), data_cleaned.index.year.max(), 9)]
yearly_cycle_values = np.array(yearly_cycle_values)

# Extracting repeating cycles from GDP using Fourier Transform
cycle_data = extract_cycles(y)

# Merging the cycle with the original data
X_with_cycles = pd.DataFrame(cycle_data, columns=['Cycle_Values'], index=y.index)
X_with_cycles['Year_Cycle_Values'] = np.resize(yearly_cycle_values, len(y))  # Adding the 9-year cycle

# Merging the data with the cycle
X_with_cycles = pd.concat([X_scaled, X_with_cycles], axis=1)

# Creating an XGBoost model to improve predictions
xg_reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=7, random_state=42)
xg_reg.fit(X_with_cycles.iloc[:len(X_train)], y_train)

# Predicting the values using the trained model
predictions = xg_reg.predict(X_with_cycles.iloc[len(X_train):])

# Evaluating the model
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# Predicting using new data
new_data = pd.read_csv('new_economic_data.csv', index_col='Date', parse_dates=True)
new_data_cleaned = new_data.dropna()

# Extracting repeating cycles from the new data
new_cycle_data = extract_cycles(new_data_cleaned['GDP'])

# Merging the 9-year cycle with the new data
new_yearly_cycle_values = [calculate_years_cycle(year, year + 8) for year in range(new_data_cleaned.index.year.min(), new_data_cleaned.index.year.max(), 9)]
new_yearly_cycle_values = np.array(new_yearly_cycle_values)

# Merging the cycle with the new data
new_X_scaled = scaler.transform(new_data_cleaned.drop('GDP', axis=1))
new_X_with_cycles = pd.DataFrame(new_cycle_data, columns=['Cycle_Values'], index=new_data_cleaned.index)
new_X_with_cycles['Year_Cycle_Values'] = np.resize(new_yearly_cycle_values, len(new_data_cleaned))

# Merging the data with the cycle
new_X_with_cycles = pd.concat([new_X_scaled, new_X_with_cycles], axis=1)

# Predicting using the new data
new_predictions = xg_reg.predict(new_X_with_cycles)

# Displaying the predictions
plt.figure(figsize=(10, 6))
plt.plot(new_data_cleaned.index, new_data_cleaned['GDP'], label='Actual GDP', color='blue')
plt.plot(new_data_cleaned.index, new_predictions, label='Predicted GDP', color='red')
plt.title('GDP Prediction Using Numerical Cycle Analysis')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.show()

# Calculating errors on the new data
new_mae = mean_absolute_error(new_data_cleaned['GDP'], new_predictions)
new_rmse = mean_squared_error(new_data_cleaned['GDP'], new_predictions, squared=False)

print(f'New Data Mean Absolute Error: {new_mae}')
print(f'New Data Root Mean Squared Error: {new_rmse}')
