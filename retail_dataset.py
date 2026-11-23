import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


class RetailDataset:
    def __init__(self, online_retail):
        self.df = online_retail
        self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"])

    def get_most_popular_product_codes(self, count):
        return self.df["StockCode"].value_counts().head(count)

    def describe(self, column_name):
        return self.df[column_name].describe()

    def set_selected_code(self, code):
        self.selected_code = code
        return self

    def filter_selected_code(self):
        return self.df[self.df["StockCode"] == self.selected_code]

    def get_all_descriptions_for_selected_code(self):
        return self.filter_selected_code()["Description"].value_counts()

    # def aggregate_quantity_by(self, aggregation):
    #     select_code = self.df['StockCode'] == self.selected_code
    #     exclude_cancellation = ~self.df['InvoiceNo'].astype('str').str.startswith('C')
    #     mask = select_code & exclude_cancellation
    #     return self.df[mask].set_index('InvoiceDate')[['Quantity']].resample(aggregation).sum()

    def missing_data(self, aggregation):
        aggregated = self.aggregate_quantity_by(aggregation)

        all_dates = pd.date_range(
            start=aggregated.index.min(), end=aggregated.index.max(), freq=aggregation
        )

        # Даты, которых нет в индексе
        missing_dates = all_dates.difference(aggregated.index)

        print("Количество пропущенных дат:", len(missing_dates))
        print(missing_dates)

        zero_dates = aggregated[aggregated["Quantity"].eq(0)]

        print("Количество дней с Quantity = 0:", len(zero_dates))
        print(zero_dates)

        nan_rows = aggregated[aggregated["Quantity"].isna()]
        print("Количество NaN:", len(nan_rows))
        print(nan_rows)

    def _agg_text(self, aggregation):
        return (
            "дням"
            if aggregation == "D"
            else "неделям" if aggregation == "W" else "месяцам"
        )

    def draw_data(self, aggregation):
        sales = self.aggregate_quantity_by(aggregation)
        plt.figure(figsize=(25, 5))
        plt.plot(sales.index, sales["Quantity"])
        plt.xticks(sales.index[::5], rotation=90)
        plt.title(f"Количество продаж по {self._agg_text(aggregation)}")
        plt.xlabel("Дата")
        plt.ylabel("Количество продаж")
        plt.show()

    def draw_missing_data(self, aggregation):
        sales = self.aggregate_quantity_by(aggregation, 1)
        plt.figure(figsize=(25, 5))
        plt.plot(sales.index, sales["Quantity"].fillna(0))
        empty_quantity_mask = (sales["Quantity"] == 0) | (sales["Quantity"].isna())
        plt.scatter(
            sales[empty_quantity_mask].index,
            sales[empty_quantity_mask]["Quantity"].fillna(0),
            color="red",
        )
        plt.xticks(sales.index[::5], rotation=90)
        plt.title(f"Количество продаж по {self._agg_text(aggregation)}")
        plt.xlabel("Дата")
        plt.ylabel("Количество продаж")
        plt.show()

    def draw_missing_hist(self):
        sales = self.aggregate_quantity_by("D", 1)
        empty_quantity_mask = (sales["Quantity"] == 0) | (sales["Quantity"].isna())
        dates = pd.to_datetime(sales[empty_quantity_mask].index)

        days_ru = [
            "Понедельник",
            "Вторник",
            "Среда",
            "Четверг",
            "Пятница",
            "Суббота",
            "Воскресенье",
        ]

        counts = Counter(dates.weekday)

        plt.bar(days_ru, [counts[i] for i in range(7)])
        plt.xlabel("День недели")
        plt.ylabel("Количество")
        plt.title("Количество отсутствующих значений по дням недели")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

    def draw_sales_by_weekday(self):
        sales = self.aggregate_quantity_by("D")
        s = sales["Quantity"]

        df_weekly = s.rename("sales").to_frame()
        df_weekly["weekday"] = df_weekly.index.dayofweek
        df_weekly["week_start"] = df_weekly.index - pd.to_timedelta(
            df_weekly.index.dayofweek, unit="D"
        )

        # Превращаем данные в таблицу:
        # строки — недели, столбцы — дни недели
        weekly_pivot = df_weekly.pivot_table(
            index="week_start", columns="weekday", values="sales", aggfunc="sum"
        )

        # Сохраняем правильный порядок дней
        days = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
        weekly_pivot = weekly_pivot.reindex(columns=range(7))
        weekly_pivot.columns = days

        # Строим наслоенный график
        plt.figure(figsize=(14, 6))

        for week, row in weekly_pivot.iterrows():
            plt.plot(days, row.values, alpha=0.25, linewidth=1)

        plt.xlabel("День недели")
        plt.ylabel("Количество проданных единиц")
        plt.title("Наслоенный график продаж по дням неделям")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    #####

    def aggregate_quantity_by(self, aggregation, min_count=0):
        select_code = self.df["StockCode"].eq(self.selected_code)
        exclude_cancellation = ~self.df["InvoiceNo"].astype(str).str.startswith("C")
        data = self.df.loc[select_code & exclude_cancellation]

        # min_count=1 сохраняет NaN, если в интервале не было записей
        return (
            data.set_index("InvoiceDate")["Quantity"]
            .resample(aggregation)
            .sum(min_count=min_count)
            .to_frame()
        )

    def diagnose_missing_data(self, aggregation):
        series = self.aggregate_quantity_by(aggregation)
        missing = series["Quantity"].isna()

        print(f"Интервалов без записей: {missing.sum()}")
        print("Их даты:")
        print(series.index[missing].tolist())

        zero = series["Quantity"].eq(0)
        print(f"\nИнтервалов с суммарным Quantity = 0: {zero.sum()}")
        print(series.loc[zero])

        return series

    def detect_iqr_outliers(self, aggregation):
        series = self.aggregate_quantity_by(aggregation)["Quantity"].dropna()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = series[(series < lower) | (series > upper)]

        print(f"Границы по IQR: [{lower:.2f}, {upper:.2f}]")
        print(f"Потенциальных выбросов: {len(outliers)}")
        print(outliers)

        return outliers

    def plot_series(self, aggregation, outliers=None):
        series = self.aggregate_quantity_by(aggregation)["Quantity"]

        plt.figure(figsize=(16, 5))
        plt.plot(series.index, series, linewidth=1, label="Количество")
        plt.axhline(0, color="gray", linewidth=0.8)

        if outliers is not None and not outliers.empty:
            plt.scatter(
                outliers.index,
                outliers,
                color="red",
                label="Потенциальные выбросы",
                zorder=3,
            )

        plt.title(f"Продажи товара по {self._agg_text(aggregation)}")
        plt.xlabel("Дата")
        plt.ylabel("Количество")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def analyze_sales(self, aggregation="D", plot=True):
        """
        Анализирует временной ряд Quantity для выбранного StockCode.

        aggregation: частота агрегации, например "D" (дни),
                     "W" (недели), "MS" (месяцы).
        plot: строить ли график.
        """
        import pandas as pd
        import matplotlib.pyplot as plt

        # Агрегированный ряд выбранного товара
        sales = self.aggregate_quantity_by(aggregation)
        quantity = sales["Quantity"].dropna()

        if quantity.empty:
            print("Нет данных для анализа выбранного товара.")
            return None

        # Описательная статистика
        stats = quantity.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])

        # Пропуски, нулевые значения и отрицательные значения
        missing = sales["Quantity"].isna()
        zero = sales["Quantity"].eq(0)
        negative = sales["Quantity"] < 0

        # Потенциальные выбросы по правилу 1.5 × IQR
        q1 = quantity.quantile(0.25)
        q3 = quantity.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = quantity[(quantity < lower) | (quantity > upper)]

        # Частота периодов в заголовке графика
        period_names = {
            "D": "дням",
            "W": "неделям",
            "MS": "месяцам",
            "M": "месяцам",
        }
        period_name = period_names.get(aggregation, aggregation)

        print("Описательные статистики Quantity:")
        print(stats)
        print(f"\nСуммарное количество: {quantity.sum()}")
        print(f"Интервалов без данных (NaN): {missing.sum()}")
        print(f"Интервалов с Quantity = 0: {zero.sum()}")
        print(f"Интервалов с отрицательным Quantity: {negative.sum()}")
        print(f"Границы выбросов по IQR: [{lower:.2f}, {upper:.2f}]")
        print(f"Потенциальных выбросов: {len(outliers)}")

        if not outliers.empty:
            print("\nПотенциальные выбросы:")
            print(outliers)

        if plot:
            plt.figure(figsize=(16, 5))
            plt.plot(sales.index, sales["Quantity"], linewidth=1, label="Quantity")
            plt.axhline(0, color="gray", linewidth=0.8)

            if not outliers.empty:
                plt.scatter(
                    outliers.index,
                    outliers,
                    color="red",
                    label="Потенциальные выбросы",
                    zorder=3,
                )

            plt.title(f"Количество товара по {period_name}")
            plt.xlabel("Дата")
            plt.ylabel("Количество")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return {
            "sales": sales,
            "statistics": stats,
            "missing_periods": sales.index[missing],
            "zero_periods": sales.index[zero],
            "negative_periods": sales.index[negative],
            "iqr_bounds": (lower, upper),
            "outliers": outliers,
        }

    def train_test_split(self, aggregation="D", horizon=61, step=1):
        data = self.aggregate_quantity_by(aggregation)

        initial_train_size = len(data) // 2

        for split_point in range(initial_train_size, len(data) - horizon + 1, step):
            train = data.iloc[:split_point]
            test = data.iloc[split_point : split_point + horizon]

            yield train, test
