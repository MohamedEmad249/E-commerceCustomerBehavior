#traditional_ml

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=500, random_state=100)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    return model.predict(X_test)


def get_all_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=100),
        "Random Forest": RandomForestRegressor(n_estimators=500, random_state=100),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=500, random_state=100),
        "KNN": KNeighborsRegressor(n_neighbors=20)
    }