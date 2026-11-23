import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class TimeSeriesBoosting:
    def __init__(self, aggregation, horizon):
        self.aggregation = aggregation
        self.horizon = horizon
        self.model = HistGradientBoostingRegressor()
        self.lags = [1, 2, 3, 7, 14]

    def make_features(self, y):
        X = pd.DataFrame(index=y.index)

        for lag in self.lags:
            X[f"lag_{lag}"] = y.shift(lag)

        X["dayofweek"] = y.index.dayofweek
        X["month"] = y.index.month
        X["day"] = y.index.day

        return X

    def fit(self, df):
        y = df["Quantity"].resample(self.aggregation).sum().astype(float)
        self.lags = [lag for lag in self.lags if lag < len(y)]
        if not self.lags:
            self.lags = [1]
        X = self.make_features(y)

        data = X.join(y.rename("target")).dropna()
        self.model.fit(data.drop(columns="target"), data["target"])

        self.history = y
        return self

    def predict(self, horizon):
        assert (
            self.horizon == horizon
        ), "Horizon должен совпадать с тем, который использовался при обучении."
        y = self.history.copy()
        future = []

        for _ in range(self.horizon):
            date = y.index[-1] + pd.tseries.frequencies.to_offset(self.aggregation)

            y.loc[date] = 0
            X = self.make_features(y).iloc[[-1]]
            pred = self.model.predict(X)[0]

            y.loc[date] = pred
            future.append(pred)

        return (
            pd.Series(future, index=y.index[-self.horizon :], name="Quantity")
            .round()
            .astype("int64")
        )
