from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

def train_models(df, target):
    X = df[["Year", "Inflation", "Demand_Index"]]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # Decision Tree (Explainable – depth controlled)
    dt = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    return {
        "Linear Regression": {
            "model": lr,
            "r2": r2_score(y_test, lr_pred),
            "mae": mean_absolute_error(y_test, lr_pred)
        },
        "Decision Tree": {
            "model": dt,
            "r2": r2_score(y_test, dt_pred),
            "mae": mean_absolute_error(y_test, dt_pred)
        }
    }
