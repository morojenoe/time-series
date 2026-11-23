import pandas as pd


class MovingAverageForecaster:
    def fit(self, sales):
        self.sales = sales

    def predict(self, horizon):
        series = self.sales["Quantity"]
        history = list(series)
        preds = []
        for _ in range(horizon):
            pred = sum(history[-horizon:]) / horizon
            preds.append(pred)
            history.append(pred)
        future_index = pd.date_range(
            series.index[-1], periods=horizon + 1, freq=pd.infer_freq(series.index)
        )[1:]
        return pd.Series(preds, index=future_index, name="Quantity")
