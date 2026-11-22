import numpy as np
import pandas as pd
import phik


class DataProcessor:

    def __init__(self, orders_path, rates_path):
        self.orders_path = orders_path
        self.rates_path = rates_path
        self.data = None
        self.profiles = None

    def load_and_prepare(self):
        data = pd.read_csv(self.orders_path)
        tenge = pd.read_csv(self.rates_path)

        print('Загружено заказов:', len(data))

        data['order_dt'] = pd.to_datetime(data['order_dt'])
        data['order_ts'] = pd.to_datetime(data['order_ts'])
        tenge['data'] = pd.to_datetime(tenge['data'])

        data = data.merge(tenge, left_on='order_dt', right_on='data', how='left')
        data['revenue_rub'] = np.where(
            data['currency_code'] == 'rub',
            data['revenue'],
            data['revenue'] / data['nominal'] * data['curs'],
        ).round(2)
        data = data.drop(columns=['data', 'nominal', 'curs'], errors='ignore')
        data['days_since_prev'] = data['days_since_prev'].astype('Int64')

        content_cols = ['user_id', 'order_dt', 'order_ts', 'event_id', 'revenue',
                        'currency_code', 'tickets_count',
                        'device_type_canonical', 'service_name']
        dup_count = data.duplicated(subset=content_cols).sum()
        data = data.drop_duplicates(subset=content_cols).reset_index(drop=True)
        print('Удалено неявных дубликатов:', dup_count)

        q99 = data['revenue_rub'].quantile(0.99)
        print('99-й перцентиль revenue_rub:', round(q99, 2))
        data = data[data['revenue_rub'] <= q99].copy()
        print('Осталось заказов:', len(data))

        self.data = data
        return data

    def build_user_profiles(self):
        data = self.data.sort_values(['user_id', 'order_ts']).reset_index(drop=True)

        first_order = (data.drop_duplicates(subset='user_id', keep='first')
                       [['user_id', 'device_type_canonical',
                         'region_name', 'event_type_main']]
                       .rename(columns={
                           'device_type_canonical': 'first_device',
                           'region_name': 'first_region',
                           'event_type_main': 'first_event_type'}))

        agg = data.groupby('user_id').agg(
            first_order_dt=('order_dt', 'min'),
            last_order_dt=('order_dt', 'max'),
            orders_total=('order_id', 'count'),
            avg_revenue_rub=('revenue_rub', 'mean'),
            avg_days_between_orders=('days_since_prev', 'mean'),
        ).reset_index()

        profile = agg.merge(first_order, on='user_id', how='left')

        purchases_pos = data[data['revenue_rub'] >= 0]
        revenue_agg = (purchases_pos.groupby('user_id')['revenue_rub']
                       .mean().rename('avg_revenue_rub'))
        profile = profile.drop(columns='avg_revenue_rub')
        profile = profile.merge(revenue_agg, on='user_id', how='left')

        only_returns = profile['avg_revenue_rub'].isna().sum()
        print('Пользователей, у которых все заказы — возвраты (исключаем):',
              only_returns)
        profile = profile[profile['avg_revenue_rub'].notna()].reset_index(drop=True)

        profile['is_two'] = (profile['orders_total'] >= 2).astype(int)
        profile['first_order_weekday'] = profile['first_order_dt'].dt.dayofweek

        q99_orders = profile['orders_total'].quantile(0.99)
        print('99-й перцентиль orders_total:', q99_orders)
        before = len(profile)
        profile = profile[profile['orders_total'] <= q99_orders].copy()
        print('Удалено аномальных пользователей:', before - len(profile),
              '(', round((before - len(profile)) / before * 100, 2), '% )')

        self.profiles = profile.reset_index(drop=True)
        return self.profiles

    def calc_retention(self, column):
        profile = self.profiles
        table = (profile.groupby(column)
                 .agg(users=('user_id', 'count'),
                      return_share=('is_two', 'mean')))
        table['users_share_%'] = (table['users'] / len(profile) * 100).round(2)
        table['return_share_%'] = (table['return_share'] * 100).round(2)
        return table[['users', 'users_share_%', 'return_share_%']] \
            .sort_values('users', ascending=False)

    def test_hypotheses(self):
        profile = self.profiles

        np.random.seed(42)
        sport = profile[profile['first_event_type'] == 'спорт']
        concerts = profile[profile['first_event_type'] == 'концерты']

        boot_diffs = []
        for _ in range(1000):
            sport_sample = sport['is_two'].sample(len(sport), replace=True)
            concerts_sample = concerts['is_two'].sample(len(concerts), replace=True)
            boot_diffs.append(sport_sample.mean() - concerts_sample.mean())
        boot_diffs = pd.Series(boot_diffs)

        lower, upper = boot_diffs.quantile([0.025, 0.975]) * 100
        print('Гипотеза 1 (спорт vs концерты):')
        print('  Спорт: %.2f%%, Концерты: %.2f%%'
              % (sport['is_two'].mean() * 100, concerts['is_two'].mean() * 100))
        print('  95%% ДИ разности долей: [%.2f%%, %.2f%%]' % (lower, upper))
        if lower > 0:
            print('  Вывод: гипотеза подтверждается')
        elif upper < 0:
            print('  Вывод: гипотеза не подтверждается, направление противоположное')
        else:
            print('  Вывод: гипотеза не подтверждается, различие не значимо')

        region_ret = self.calc_retention('first_region')
        big = region_ret[region_ret['users'] >= 100]
        corr = big['users'].corr(big['return_share_%'], method='spearman')
        print('\nГипотеза 2 (размер региона vs доля возврата):')
        print('  Корреляция Спирмена (регионы с >= 100 пользователей):',
              round(corr, 3))
        if corr > 0.5:
            print('  Вывод: гипотеза подтверждается')
        else:
            print('  Вывод: гипотеза не подтверждается, связь слабая')

        return boot_diffs, corr

    def corr_analysis(self):
        profile = self.profiles.copy()

        cols = ['orders_total', 'avg_revenue_rub', 'avg_days_between_orders',
                'first_device', 'first_event_type', 'first_region',
                'first_order_weekday']

        phik_matrix = profile[cols].phik_matrix(
            interval_cols=['orders_total', 'avg_revenue_rub',
                           'avg_days_between_orders', 'first_order_weekday'])

        print('Корреляция phi_k с orders_total:')
        print(phik_matrix['orders_total'].sort_values(ascending=False).round(3))

        profile['orders_segment'] = pd.cut(
            profile['orders_total'],
            bins=[0, 1, 2, 4, np.inf],
            labels=['1 заказ', '2 заказа', '3-4 заказа', '5+'])
        print('\nСегменты:')
        print(profile['orders_segment'].value_counts().sort_index())

        cols_seg = ['avg_revenue_rub', 'avg_days_between_orders',
                    'first_device', 'first_event_type', 'first_region',
                    'first_order_weekday']
        phik_seg = profile[cols_seg + ['orders_segment']].phik_matrix(
            interval_cols=['avg_revenue_rub', 'avg_days_between_orders',
                           'first_order_weekday'])
        print('\nКорреляция phi_k с сегментом числа заказов:')
        print(phik_seg['orders_segment'].sort_values(ascending=False).round(3))

        self.phik_matrix = phik_matrix
        self.phik_seg = phik_seg
        return phik_matrix, phik_seg
