from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

def train_and_select_model(df, target):
    """
    Trains interpretable ML models and selects
    the best-performing model based on R² score.
    """
    X = df[["Year", "Inflation (%)", "Demand_Index"]]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "model": model,
            "R2": r2_score(y_test, preds),
            "MAE": mean_absolute_error(y_test, preds)
        }

    best_model_name = max(results, key=lambda x: results[x]["R2"])
    return best_model_name, results[best_model_name]["model"], results
