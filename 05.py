import pandas as pd
import numpy as np


def main():
    # Шаг 1: Загрузка и первичный осмотр данных

    # 1. Загрузите датасет uber_pickups_bookings.csv
    try:
        df = pd.read_csv('uber_pickups_bookings.csv')
        print("Файл успешно загружен!")
    except FileNotFoundError:
        print("Файл не найден! Убедитесь, что файл находится в правильной директории.")
        return

    # 2. Выведите первые 5 строк датасета
    print("\n" + "=" * 50)
    print("2. Первые 5 строк датасета:")
    print("=" * 50)
    print(df.head())

    # 3. Выведите общую информацию о датасете
    print("\n" + "=" * 50)
    print("3. Общая информация о датасете:")
    print("=" * 50)
    df.info()

    # 4. Выведите статистическое описание числовых столбцов
    print("\n" + "=" * 50)
    print("4. Статистическое описание числовых столбцов:")
    print("=" * 50)
    print(df.describe())

    # 5. Определите количество строк и столбцов
    print("\n" + "=" * 50)
    print("5. Количество строк и столбцов:")
    print("=" * 50)
    print(f"Количество строк: {df.shape[0]}")
    print(f"Количество столбцов: {df.shape[1]}")

    # Шаг 3: Выборка и фильтрация данных

    # 1. Выберите указанные столбцы и выведите первые 5 строк
    print("\n" + "=" * 50)
    print("1. Выбранные столбцы (первые 5 строк):")
    print("=" * 50)
    selected_columns = df[['Booking ID', 'booking_datetime', 'Booking Status', 'Vehicle Type', 'Payment Method']]
    print(selected_columns.head())

    # 2. Отфильтруйте бронирования с статусом "Cancelled by Driver"
    print("\n" + "=" * 50)
    print("2. Бронирования отмененные водителем:")
    print("=" * 50)
    cancelled_by_driver = df[df['Booking Status'] == 'Cancelled by Driver']
    print(cancelled_by_driver)

    # 3. Отфильтруйте бронирования с Vehicle Type "Auto" и Booking Value > 500
    print("\n" + "=" * 50)
    print("3. Бронирования Auto с Booking Value > 500:")
    print("=" * 50)
    auto_high_value = df[(df['Vehicle Type'] == 'Auto') & (df['Booking Value'] > 500)]
    print(auto_high_value)

    # 4. Отфильтруйте бронирования за март 2024 года
    print("\n" + "=" * 50)
    print("4. Бронирования за март 2024 года:")
    print("=" * 50)
    # Преобразуем booking_datetime в datetime формат
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])

    # Фильтруем по диапазону дат
    march_2024_bookings = df[
        (df['booking_datetime'] >= '2024-03-01') &
        (df['booking_datetime'] <= '2024-03-31')
        ]
    print(march_2024_bookings)


if name == "__main__":
    main()