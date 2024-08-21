

```python
# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('data/historical_data.csv')

# Data exploration
print(data.head())
print(data.describe())

# Data preprocessing
# Example: Convert date to datetime and sort data
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Feature and target variable
X = data[['Open', 'High', 'Low', 'Volume']]  # Example features
y = data['Close']  # Target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot results
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='True')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
