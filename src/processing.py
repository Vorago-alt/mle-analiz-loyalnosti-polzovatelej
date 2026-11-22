import pandas as pd


class DataProcessor:
    """Загрузка и предобработка данных Яндекс Афиши."""

    def __init__(self, orders_path, rates_path):
        self.orders_path = orders_path
        self.rates_path = rates_path
        self.data = None

    def load_and_prepare(self):
        # 1. Загрузка выгрузки из SQL и курсов тенге
        data = pd.read_csv(self.orders_path)
        tenge = pd.read_csv(self.rates_path)

        print('Загружено заказов:', len(data))

        # 2. Приводим даты к datetime
        data['order_dt'] = pd.to_datetime(data['order_dt'])
        data['order_ts'] = pd.to_datetime(data['order_ts'])
        tenge['data'] = pd.to_datetime(tenge['data'])

        # 3. Перевод тенге в рубли по курсу на дату заказа
        data = data.merge(tenge, left_on='order_dt', right_on='data', how='left')
        data['revenue_rub'] = data['revenue'].where(
            data['currency_code'] == 'rub',
            (data['revenue'] / data['nominal'] * data['curs']).round(2)
        )
        data = data.drop(columns=['data', 'nominal', 'curs'])

        # 4. Неявные дубликаты: совпадение по содержательным полям
        content_cols = ['user_id', 'order_ts', 'event_id', 'revenue',
                        'currency_code', 'tickets_count',
                        'device_type_canonical', 'service_name']
        dup_count = data.duplicated(subset=content_cols).sum()
        data = data.drop_duplicates(subset=content_cols).reset_index(drop=True)
        print('Удалено неявных дубликатов:', dup_count)

        # 5. Фильтр выбросов: выручка не выше 99-го перцентиля
        q99 = data['revenue_rub'].quantile(0.99)
        print('99-й перцентиль revenue_rub:', round(q99, 2))
        data = data[data['revenue_rub'] <= q99].copy()
        print('Осталось заказов:', len(data))

        self.data = data
        return data
