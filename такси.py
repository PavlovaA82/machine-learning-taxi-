import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------- 1. Загрузка данных ----------
print("📂 Загружаем данные...")
df = pd.read_csv(r"D:\Такси\ncr_ride_bookings.csv")

print(f"Загружено {len(df)} строк, {len(df.columns)} колонок")
print("Колонки:", list(df.columns))

# ---------- 2. Очистка данных ----------
df = df.dropna(how='all', axis=1)
df = df.dropna(subset=['Booking Value', 'Ride Distance'])

# Преобразуем числовые колонки
num_cols = ['Ride Distance', 'Avg VTAT', 'Avg CTAT']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['Ride Distance', 'Booking Value'])
df = df[df['Ride Distance'] > 0]

# ---------- 3. Кодирование категориальных признаков ----------
cat_cols = ['Vehicle Type', 'Pickup Location', 'Drop Location', 'Booking Status']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# ---------- 4. Генерация тематических факторов ----------
print("🧩 Генерируем дополнительные факторы...")

# Квадрат расстояния — отражает нелинейный рост стоимости
df["distance_squared"] = df["Ride Distance"] ** 2

# Отношение средних времен (если есть)
df["time_ratio"] = df["Avg VTAT"] / (df["Avg CTAT"] + 1e-5)

# Интерактивный фактор: расстояние × время подачи
df["distance_vtat"] = df["Ride Distance"] * df["Avg VTAT"]

# Флаг дальних поездок
df["is_long_trip"] = (df["Ride Distance"] > df["Ride Distance"].median()).astype(int)

# ---------- 5. Добавляем погодные факторы ----------
print("🌦️ Добавляем погодные условия...")

np.random.seed(42)
# Температура (°C)
df["temperature"] = np.random.uniform(10, 40, len(df))
# Интенсивность дождя (0 = сухо, 1 = дождь)
df["rain_intensity"] = np.random.choice([0, 1], len(df), p=[0.8, 0.2])
# Влажность воздуха (%)
df["humidity"] = np.random.uniform(30, 90, len(df))

# Тематический фактор: дождь × расстояние
df["rain_effect"] = df["rain_intensity"] * df["Ride Distance"]

# ---------- 6. Формирование X и y ----------
X = df[['Ride Distance', 'Avg VTAT', 'Avg CTAT', 'Vehicle Type',
        'Pickup Location', 'Drop Location', 'Booking Status',
        'distance_squared', 'time_ratio', 'distance_vtat',
        'is_long_trip', 'temperature', 'rain_intensity',
        'humidity', 'rain_effect']].fillna(0)

y = df['Booking Value']

print(f"✅ Используется {X.shape[1]} факторов для обучения.")

# ---------- 7. Разделение данных ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- 8. Обучение моделей ----------
models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"⚙️ Обучаем {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

# ---------- 9. Сравнение моделей ----------
comp = pd.DataFrame(results).T
print("\n📊 Сравнение моделей:")
print(comp.round(4))

best_model_name = comp["R2"].idxmax()
print(f"\n🏆 Лучшая модель: {best_model_name}")

best_model = models[best_model_name]

# ---------- 10. Важность признаков ----------
if hasattr(best_model, "feature_importances_"):
    feat_imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\n🔍 Топ-10 факторов по важности:")
    print(feat_imp.head(10))

    # Используем неблокирующий режим отображения
    plt.figure(figsize=(8, 5))
    plt.barh(feat_imp["Feature"].head(10), feat_imp["Importance"].head(10))
    plt.gca().invert_yaxis()
    plt.title("Важность факторов в прогнозе Booking Value")

    # Вместо show() сохраняем картинку и не блокируем выполнение
    plt.savefig("feature_importance.png")
    plt.close()
    print("📊 График важности сохранён в файл 'feature_importance.png'")

# ---------- 11. Пример прогноза ----------
print("\n✅ Переходим к прогнозу...")

if len(X) > 0:
    sample = X.sample(1, random_state=42)
    pred = best_model.predict(sample)[0]
    print("\n💰 Пример прогноза:")
    print(sample)
    print(f"\nПредсказанное значение Booking Value: {pred:.2f}")
else:
    print("❌ Нет данных для прогноза.")
