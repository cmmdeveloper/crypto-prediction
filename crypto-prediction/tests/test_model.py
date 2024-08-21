import pandas as pd
from src.data_preprocessing import preprocess_data
from src.model import train_model, evaluate_model
from sklearn.model_selection import train_test_split

def test_train_model():
    data = preprocess_data('data/historical_data.csv')
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    model = train_model(X_train, y_train)
    mse, _ = evaluate_model(model, X_test, y_test)
    assert mse < 1000  # Example assertion, adjust based on your model's performance
