from src.processing import DataProcessor


def main():
    processor = DataProcessor(
        orders_path='data/orders.csv',
        rates_path='data/final_tickets_tenge_df.csv',
    )
    data = processor.load_and_prepare()

    print('\nГотово. Заказов после предобработки:', len(data))


if __name__ == '__main__':
    main()
