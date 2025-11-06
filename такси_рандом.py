import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Шаг 1: Генерация реалистичных синтетических данных
np.random.seed(42)


def generate_ride_data(n_samples=10000):
    """Генерация данных о поездках"""

    data = {
        # Основные факторы
        'distance_km': np.random.uniform(1, 50, n_samples),
        'duration_min': np.random.uniform(5, 120, n_samples),

        # Временные факторы
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'month': np.random.randint(1, 13, n_samples),

        # Погодные условия
        'temperature': np.random.uniform(-10, 35, n_samples),
        'precipitation': np.random.uniform(0, 20, n_samples),
        'visibility_km': np.random.uniform(0.1, 20, n_samples),

        # Факторы спроса
        'surge_multiplier': np.random.choice([1.0, 1.2, 1.5, 2.0, 3.0], n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
        'area_demand_index': np.random.uniform(0.5, 3.0, n_samples),

        # Географические факторы
        'start_area_wealth': np.random.uniform(0.5, 2.0, n_samples),
        'end_area_wealth': np.random.uniform(0.5, 2.0, n_samples),
        'cross_city_trip': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),

        # Факторы транспортного средства
        'vehicle_type': np.random.choice(['economy', 'comfort', 'business', 'premium'], n_samples,
                                         p=[0.5, 0.3, 0.15, 0.05]),
        'vehicle_age': np.random.randint(0, 8, n_samples),
        'fuel_efficiency': np.random.uniform(8, 15, n_samples),

        # Факторы водителя
        'driver_rating': np.random.uniform(4.0, 5.0, n_samples),
        'driver_experience_years': np.random.randint(0, 20, n_samples),

        # Дополнительные услуги
        'has_air_conditioning': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'has_wifi': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'extra_luggage': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),

        # Трафик и дорожные условия
        'traffic_index': np.random.uniform(1.0, 3.0, n_samples),
        'road_quality': np.random.uniform(0.5, 1.5, n_samples),
        'num_traffic_lights': np.random.randint(0, 15, n_samples),

        # Экономические факторы
        'fuel_price': np.random.uniform(45, 60, n_samples),
        'operating_cost_index': np.random.uniform(0.8, 1.3, n_samples)
    }

    df = pd.DataFrame(data)

    # Расчет базовой стоимости с учетом всех факторов
    base_price = (
            df['distance_km'] * 12 +  # базовый тариф за км
            df['duration_min'] * 2 +  # тариф за время
            df['distance_km'] * df['fuel_price'] / df['fuel_efficiency'] +  # топливные расходы
            df['traffic_index'] * df['duration_min'] * 0.5 +  # влияние трафика
            df['precipitation'] * 2 +  # влияние погоды
            (df['start_area_wealth'] + df['end_area_wealth']) * 15 +  # влияние района
            df['operating_cost_index'] * 20  # операционные расходы
    )

    # Модификаторы
    vehicle_modifiers = {'economy': 1.0, 'comfort': 1.3, 'business': 1.7, 'premium': 2.5}
    df['vehicle_modifier'] = df['vehicle_type'].map(vehicle_modifiers)

    time_modifier = (
            (df['hour_of_day'].isin([7, 8, 17, 18])).astype(int) * 0.3 +  # час пик
            (df['is_weekend'] == 1) * 0.2 +  # выходные
            ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int) * 0.4  # ночное время
    )

    service_modifier = (
            df['has_air_conditioning'] * 0.1 +
            df['has_wifi'] * 0.15 +
            df['extra_luggage'] * 0.2
    )

    # Финальная стоимость
    df['trip_cost'] = (
                              base_price *
                              df['vehicle_modifier'] *
                              df['surge_multiplier'] *
                              (1 + time_modifier) *
                              (1 + service_modifier) *
                              (1 + (5 - df['driver_rating']) * 0.05)  # влияние рейтинга водителя
                      ) + np.random.normal(0, 10, n_samples)  # случайный шум

    # Ограничение минимальной стоимости
    df['trip_cost'] = np.maximum(df['trip_cost'], 50)

    return df.drop('vehicle_modifier', axis=1)


# Генерация данных
print("🚗 Генерация данных о поездках...")
ride_data = generate_ride_data(10000)
print(f"✅ Сгенерировано {len(ride_data)} записей")
print(f"📊 Столбцы: {list(ride_data.columns)}")

# Шаг 2: Предобработка данных
print("\n🔧 Предобработка данных...")

# Кодирование категориальных переменных
label_encoders = {}
categorical_columns = ['vehicle_type']

for col in categorical_columns:
    le = LabelEncoder()
    ride_data[col] = le.fit_transform(ride_data[col])
    label_encoders[col] = le

# Разделение на признаки и целевую переменную
X = ride_data.drop('trip_cost', axis=1)
y = ride_data['trip_cost']

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Масштабирование числовых признаков
scaler = StandardScaler()
numerical_columns = X.columns.difference(categorical_columns)
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

print(f"📐 Размер тренировочной выборки: {X_train.shape}")
print(f"📐 Размер тестовой выборки: {X_test.shape}")

# Шаг 3: Обучение нескольких моделей
print("\n🤖 Обучение моделей...")

models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression()
}

results = {}

for name, model in models.items():
    print(f"Обучение {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        'model': model,
        'predictions': y_pred,
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }

# Шаг 4: Сравнение моделей
print("\n📊 Сравнение моделей:")
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'MAE': [results[name]['mae'] for name in results.keys()],
    'RMSE': [results[name]['rmse'] for name in results.keys()],
    'R²': [results[name]['r2'] for name in results.keys()]
})

print(comparison_df.round(4))

# Шаг 5: Анализ важности факторов для лучшей модели
best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']

print(f"\n🏆 Лучшая модель: {best_model_name}")

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n🔍 Важность факторов:")
    print(feature_importance.head(15))

# Шаг 6: Визуализация результатов
plt.figure(figsize=(15, 10))

# 1. Важность факторов
plt.subplot(2, 2, 1)
if hasattr(best_model, 'feature_importances_'):
    top_features = feature_importance.head(10)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.title('Топ-10 важнейших факторов')
    plt.xlabel('Важность')
    plt.gca().invert_yaxis()

# 2. Сравнение моделей
plt.subplot(2, 2, 2)
metrics = ['MAE', 'RMSE', 'R²']
x_pos = np.arange(len(metrics))
width = 0.25

for i, (name, result) in enumerate(results.items()):
    values = [result['mae'], result['rmse'], result['r2']]
    plt.bar(x_pos + i * width, values, width, label=name)

plt.xlabel('Метрики')
plt.ylabel('Значение')
plt.title('Сравнение моделей')
plt.xticks(x_pos + width, metrics)
plt.legend()

# 3. Предсказания vs Фактические значения
plt.subplot(2, 2, 3)
plt.scatter(y_test, results[best_model_name]['predictions'], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Фактическая стоимость')
plt.ylabel('Предсказанная стоимость')
plt.title(f'Предсказания {best_model_name}')

# 4. Распределение ошибок
plt.subplot(2, 2, 4)
errors = results[best_model_name]['predictions'] - y_test
plt.hist(errors, bins=50, alpha=0.7)
plt.xlabel('Ошибка предсказания')
plt.ylabel('Частота')
plt.title('Распределение ошибок')

plt.tight_layout()
plt.show()


# Шаг 7: Функция для предсказания стоимости новой поездки
def predict_ride_cost(model, scaler, label_encoders, features):
    """Предсказание стоимости поездки на основе введенных факторов"""

    # Создание DataFrame с теми же столбцами
    feature_df = pd.DataFrame([features])

    # Кодирование категориальных переменных
    for col, encoder in label_encoders.items():
        if features[col] in encoder.classes_:
            feature_df[col] = encoder.transform([features[col]])[0]
        else:
            feature_df[col] = 0  # значение по умолчанию

    # Масштабирование числовых признаков
    numerical_cols = feature_df.columns.difference(categorical_columns)
    feature_df[numerical_cols] = scaler.transform(feature_df[numerical_cols])

    # Предсказание
    prediction = model.predict(feature_df)[0]
    return max(prediction, 0)  # Обеспечиваем неотрицательную стоимость


# Пример использования
print("\n🎯 Пример предсказания стоимости поездки:")
sample_ride = {
    'distance_km': 15.5,
    'duration_min': 35,
    'hour_of_day': 18,
    'day_of_week': 4,
    'is_weekend': 0,
    'month': 6,
    'temperature': 25,
    'precipitation': 0,
    'visibility_km': 10,
    'surge_multiplier': 1.5,
    'area_demand_index': 2.1,
    'start_area_wealth': 1.2,
    'end_area_wealth': 1.5,
    'cross_city_trip': 0,
    'vehicle_type': 'comfort',
    'vehicle_age': 2,
    'fuel_efficiency': 12,
    'driver_rating': 4.8,
    'driver_experience_years': 5,
    'has_air_conditioning': 1,
    'has_wifi': 0,
    'extra_luggage': 0,
    'traffic_index': 2.5,
    'road_quality': 1.2,
    'num_traffic_lights': 8,
    'fuel_price': 52.5,
    'operating_cost_index': 1.1
}

predicted_cost = predict_ride_cost(best_model, scaler, label_encoders, sample_ride)
print(f"💵 Предсказанная стоимость поездки: {predicted_cost:.2f} руб.")

# Анализ чувствительности к ключевым факторам
print("\n📈 Анализ чувствительности стоимости к расстоянию:")
distances = [5, 10, 15, 20, 25]
for dist in distances:
    sample_ride['distance_km'] = dist
    cost = predict_ride_cost(best_model, scaler, label_encoders, sample_ride)

    print(f"  {dist} км → {cost:.2f} руб.")
