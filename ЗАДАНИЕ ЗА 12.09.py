import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 50)
print("АНАЛИЗ ДАННЫХ О ДОСТАВКЕ ЕДЫ")
print("=" * 50)

# Создаем тестовые данные
np.random.seed(123)
data = {
    'order_id': range(1000, 1500),
    'customer': [f'User_{i}' for i in range(500)],
    'restaurant': np.random.choice(['Pizza', 'Burger', 'Sushi', 'Tacos', 'Chinese'], 500),
    'status': np.random.choice(['Delivered', 'Cooking', 'On way', 'Cancelled'], 500),
    'price': np.random.uniform(200, 2000, 500).round(2),
    'delivery_time': np.random.randint(20, 90, 500),
    'payment': np.random.choice(['Card', 'Cash', 'Online'], 500),
    'date': [datetime(2024, np.random.randint(1, 13), np.random.randint(1, 28)) for _ in range(500)]
}

# Добавляем немного пропущенных значений
for col in ['restaurant', 'status', 'price']:
    mask = np.random.random(500) < 0.1
    data[col] = [None if mask[i] else data[col][i] for i in range(500)]

df = pd.DataFrame(data)

print("1. Первые 5 строк:")
print(df.head())
print("\n" + "-"*30)

print("2. Основная информация:")
print(f"Строк: {len(df)}, Столбцов: {len(df.columns)}")
print(f"Пропусков всего: {df.isnull().sum().sum()}")
print("\n" + "-"*30)

print("3. Пропуски по столбцам:")
missing = df.isnull().sum()
for col, count in missing.items():
    if count > 0:
        print(f"{col}: {count} пропусков")
print("\n" + "-"*30)

print("4. Уникальные значения статусов:")
print(df['status'].value_counts())
print(f"Всего статусов: {df['status'].nunique()}")
print("\n" + "-"*30)

print("5. Уникальные значения ресторанов:")
print(df['restaurant'].value_counts())
print(f"Всего ресторанов: {df['restaurant'].nunique()}")
print("\n" + "-"*30)

print("6. Статистика по ценам:")
print(f"Средняя цена: {df['price'].mean():.2f}")
print(f"Мин: {df['price'].min():.2f}, Макс: {df['price'].max():.2f}")
print("\n" + "-"*30)

print("7. Фильтрация данных:")
print("Заказы из Pizza дороже 1000:")
pizza_expensive = df[(df['restaurant'] == 'Pizza') & (df['price'] > 1000)]
print(pizza_expensive[['order_id', 'price', 'status']])
print(f"Найдено: {len(pizza_expensive)} заказов")
print("\n" + "-"*30)

print("Отмененные заказы:")
cancelled = df[df['status'] == 'Cancelled']
print(cancelled[['order_id', 'restaurant', 'price']])
print(f"Найдено: {len(cancelled)} заказов")
print("\n" + "-"*30)

print("8. Методы оплаты:")
print(df['payment'].value_counts())
print("\n" + "-"*30)

print("9. Время доставки:")
print(f"Среднее: {df['delivery_time'].mean():.1f} мин")
print(f"Мин: {df['delivery_time'].min()} мин, Макс: {df['delivery_time'].max()} мин")
print("\n" + "-"*30)

print("АНАЛИЗ ЗАВЕРШЕН!")
print("=" * 50)