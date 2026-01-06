#evaluation

from sklearn.metrics import mean_squared_error

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
