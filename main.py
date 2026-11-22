from src.processing import DataProcessor


def main():
    processor = DataProcessor(
        orders_path='data/orders.csv',
        rates_path='data/final_tickets_tenge_df.csv',
    )
    data = processor.load_and_prepare()

    profiles = processor.build_user_profiles()
    print('\nПрофиль пользователя: подтверждение шага 3 первой части')
    print('Пользователей в выборке:', len(profiles))
    print('Средняя выручка с одного заказа:',
          round(profiles['avg_revenue_rub'].mean(), 2), 'руб.')
    print('Доля пользователей с 2+ заказами:',
          round(profiles['is_two'].mean() * 100, 2), '%')

    print('\n--- Retention по сегментам ---')
    print('По жанру первого мероприятия:')
    print(processor.calc_retention('first_event_type'))
    print('\nПо устройству первого заказа:')
    print(processor.calc_retention('first_device'))
    print('\nПо региону первого заказа (топ-10):')
    print(processor.calc_retention('first_region').head(10))

    print('\n--- Гипотезы ---')
    processor.test_hypotheses()

    print('\n--- Корреляционный анализ ---')
    processor.corr_analysis()


if __name__ == '__main__':
    main()
