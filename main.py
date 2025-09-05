import pandas as pd

def step3_analysis():
    # Загрузка данных
    try:
        df = pd.read_csv('ncr_ride_bookings.csv')
        print("Файл успешно загружен!")
        print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
    except FileNotFoundError:
        print("Файл 'uber_pickups_bookings.csv' не найден!")
        print("Убедитесь, что файл находится в той же папке, что и скрипт")
        return

    print("\n" + "=" * 60)
    print("ШАГ 3: ВЫБОРКА И ФИЛЬТРАЦИЯ ДАННЫХ")
    print("=" * 60)

    # 1. Выберите только столбцы Booking ID, booking_datetime,
    # Booking Status, Vehicle Type, Payment Method
    print("\n1. Выбранные столбцы (первые 5 строк):")
    print("-" * 40)
    selected_cols = ['Booking ID', 'booking_datetime', 'Booking Status', 'Vehicle Type', 'Payment Method']

    # Проверяем, что все столбцы существуют в датасете
    available_cols = [col for col in selected_cols if col in df.columns]
    if len(available_cols) != len(selected_cols):
        print("Предупреждение: Не все запрошенные столбцы найдены в датасете")
        print(f"Найдены столбцы: {available_cols}")

    selected_data = df[available_cols]
    print(selected_data.head())
    print(f"Всего записей: {len(selected_data)}")

    # 2. Отфильтруйте все бронирования со статусом "Cancelled by Driver"
    print("\n\n2. Бронирования со статусом 'Cancelled by Driver':")
    print("-" * 60)

    if 'Booking Status' in df.columns:
        cancelled_driver = df[df['Booking Status'] == 'Cancelled by Driver']
        print(cancelled_driver)
        print(f"\nНайдено бронирований: {len(cancelled_driver)}")
    else:
        print("Столбец 'Booking Status' не найден в датасете")

    # 3. Отфильтруйте бронирования с Vehicle Type "Auto" и Booking Value > 500
    print("\n\n3. Бронирования: Vehicle Type 'Auto' AND Booking Value > 500:")
    print("-" * 70)

    conditions_met = True
    if 'Vehicle Type' not in df.columns:
        print("Столбец 'Vehicle Type' не найден")
        conditions_met = False
    if 'Booking Value' not in df.columns:
        print("Столбец 'Booking Value' не найден")
        conditions_met = False

    if conditions_met:
        auto_high_value = df[(df['Vehicle Type'] == 'Auto') & (df['Booking Value'] > 500)]
        print(auto_high_value)
        print(f"\nНайдено бронирований: {len(auto_high_value)}")
    else:
        print("Невозможно выполнить фильтрацию: отсутствуют необходимые столбцы")

    # 4. Отфильтруйте бронирования за март 2024 года
    print("\n\n4. Бронирования за март 2024 года (с 2024-03-01 по 2024-03-31):")
    print("-" * 80)

    if 'booking_datetime' in df.columns:
        # Преобразуем в datetime
        df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])

        # Фильтруем по датам
        march_2024 = df[
            (df['booking_datetime'] >= '2024-03-01') &
            (df['booking_datetime'] <= '2024-03-31')
            ]

        print(march_2024)
        print(f"\nНайдено бронирований за март 2024: {len(march_2024)}")

        # Дополнительная информация о датах
        if len(march_2024) > 0:
            print(f"Первая дата в периоде: {march_2024['booking_datetime'].min()}")
            print(f"Последняя дата в периоде: {march_2024['booking_datetime'].max()}")
    else:
        print("Столбец 'booking_datetime' не найден в датасете")

    print("\n" + "=" * 60)
    print("ШАГ 3 ЗАВЕРШЕН")
    print("=" * 60)


# Запуск анализа
if __name__ == "__main__":
    step3_analysis()