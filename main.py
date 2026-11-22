from src.processing import DataProcessor


def print_report(processor, profiles):
    p = profiles

    print('\n' + '=' * 60)
    print('ИТОГОВЫЙ ОТЧЁТ: возврат пользователей Яндекс Афиши')
    print('=' * 60)

    print('\n1. Данные и подготовка:')
    print('  - выгрузка из SQL: заказы без фильмов, мобильные и desktop;')
    print('  - тенге переведены в рубли по ежедневному курсу;')
    print('  - удалены неявные дубликаты, отфильтрованы выбросы выручки')
    print('    (выше 99-го перцентиля) и аномальные пользователи по числу заказов.')

    print('\n2. Профиль пользователя:')
    print(f'  - пользователей в выборке: {len(p)}')
    print(f'  - средняя выручка с одного заказа: '
          f'{p["avg_revenue_rub"].mean():.2f} руб.')
    print(f'  - доля вернувшихся (2+ заказа): {p["is_two"].mean() * 100:.1f}%')

    print('\n3. Возврат по признакам первого заказа:')
    device = processor.calc_retention('first_device')
    for dev, row in device.iterrows():
        print(f'  - {dev}: {row["return_share_%"]:.1f}% '
              f'({int(row["users"])} пользователей)')
    genre = processor.calc_retention('first_event_type')
    top_genre = genre.index[0]
    print(f'  - крупнейший сегмент по жанру: {top_genre} '
          f'({genre.loc[top_genre, "return_share_%"]:.1f}% возврата);')
    region = processor.calc_retention('first_region')
    big = region[region['users'] >= 100]
    print(f'  - лучший регион из крупных: {big.index[0]} '
          f'({big["return_share_%"].max():.1f}%);')

    print('\n4. Гипотезы:')
    print('  - «спорт возвращает чаще концертов» — не подтвердилась,')
    print('    возврат спорта ниже концертов (направление противоположное);')
    print('  - «крупные регионы возвращают чаще» — не подтвердилась,')
    print('    связь размера региона с возвратом слабая (Спирмен ~0.1).')

    print('\n5. Что влияет на возврат и число покупок:')
    print('  - из признаков первого заказа заметнее всего устройство')
    print('    (desktop возвращается чаще mobile);')
    print('  - день недели и жанр первого заказа значимого влияния не дают;')
    print('  - по корреляции phi_k число покупок связано с поведением:')
    print('    интервал между заказами (~0.28) и средний чек (~0.22);')
    print('  - вывод: удержание определяется поведением после первого')
    print('    контакта, а не «точкой входа».')

    print('\n6. Рекомендации:')
    print('  - доработать мобильный путь повторной покупки')
    print('    (основной канал с худшим возвратом);')
    print('  - предлагать подборки и купоны после дешёвого первого заказа;')
    print('  - локальные кампании в регионах с низким возвратом;')
    print('  - в модель прогноза возврата брать поведенческие признаки.')


def main():
    processor = DataProcessor(
        orders_path='data/orders.csv',
        rates_path='data/final_tickets_tenge_df.csv',
    )

    data = processor.load_and_prepare()
    profiles = processor.build_user_profiles()

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

    print_report(processor, profiles)


if __name__ == '__main__':
    main()
