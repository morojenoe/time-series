from statsmodels.tsa.arima.model import ARIMA


class ARIMAForecaster:
    def __init__(self, aggregation, horizon):
        self.aggregation = aggregation
        self.horizon = horizon
        self.model = None

    def fit(self, df):
        ts = df["Quantity"].resample(self.aggregation).sum()
        self.model = ARIMA(ts, order=(1, 1, 1)).fit()

    def predict(self, horizon):
        assert (
            horizon == self.horizon
        ), "Horizon должен совпадать с тем, который использовался при обучении."
        return self.model.forecast(self.horizon)
