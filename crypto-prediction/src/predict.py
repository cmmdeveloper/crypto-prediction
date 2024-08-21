import pandas as pd
from model import train_model, evaluate_model
from data_preprocessing import preprocess_data

def main():
    data = preprocess_data('data/historical_data.csv')
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = train_model(X_train, y_train)
    mse, y_pred = evaluate_model(model, X_test, y_test)

    print(f'Mean Squared Error: {mse}')

if __name__ == '__main__':
    main()
