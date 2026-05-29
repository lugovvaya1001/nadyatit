import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# ЗАДАНИЕ 1: Анализ 100 векторов 4-го ранга (4D пространство)
# ============================================================

print("=" * 60)
print("ЗАДАНИЕ 1: 4D векторы -> 3D визуализация")
print("=" * 60)

# Генерация случайных данных для 100 векторов в 4-мерном пространстве
# (замените этот блок на загрузку вашего файла с векторами)
np.random.seed(42)
n_vectors = 100
df_vectors = pd.DataFrame({
    'x': np.random.randn(n_vectors) * 2,  # первая координата
    'y': np.random.randn(n_vectors) * 2,  # вторая координата
    'z': np.random.randn(n_vectors) * 2,  # третья координата
    'w': np.random.randn(n_vectors) * 2   # четвёртая координата (используется для цвета)
})

print(f"Загружено {len(df_vectors)} векторов, размерность: 4D")
print(df_vectors.head())

# Создание 3D графика (отображаем первые три координаты)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Построение scatter plot с цветовой кодировкой по 4-й координате
scatter = ax.scatter(
    df_vectors['x'], 
    df_vectors['y'], 
    df_vectors['z'],
    c=df_vectors['w'],
    cmap='plasma',
    s=50,
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)

# Подписи осей и заголовок
ax.set_xlabel('X (координата 1)', fontsize=12)
ax.set_ylabel('Y (координата 2)', fontsize=12)
ax.set_zlabel('Z (координата 3)', fontsize=12)
ax.set_title('Визуализация 100 векторов в 3D (цвет = 4-я координата)', fontsize=14)
fig.colorbar(scatter, ax=ax, shrink=0.5, label='W (4-я координата)')

plt.tight_layout()
plt.savefig('vectors_3d.png', dpi=150)
plt.show()
print("График векторов сохранён как 'vectors_3d.png'")
print()

# ============================================================
# ЗАДАНИЕ 2: Титаник - построение матрицы корреляций
# ============================================================

print("=" * 60)
print("ЗАДАНИЕ 2: Титаник - матрица корреляций")
print("=" * 60)

# Загрузка датасета Титаник
from seaborn import load_dataset
df_titanic = load_dataset('titanic')

print("Исходные колонки:")
print(df_titanic.columns.tolist())
print(f"Размер: {df_titanic.shape}")
print(f"NaN до обработки:\n{df_titanic.isnull().sum()}")
print()

# Копирование данных для очистки
df_clean = df_titanic.copy()

# Удаление колонок, которые не несут статистической пользы для корреляции
# (имена, номера билетов, дублирующие категориальные признаки и т.д.)
cols_to_drop = ['name', 'ticket', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone', 'class']
df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns], inplace=True)

# Удаление колонки с каютами (слишком много пропусков)
if 'cabin' in df_clean.columns:
    df_clean.drop(columns=['cabin'], inplace=True)

# Заполнение пропусков в возрасте медианным значением
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())

# Заполнение пропусков в порте посадки модальным значением
df_clean['embarked'] = df_clean['embarked'].fillna(df_clean['embarked'].mode()[0])

# Преобразование пола в числовое значение: male -> 0, female -> 1
df_clean['sex'] = df_clean['sex'].map({'male': 0, 'female': 1})

# Кодирование порта посадки в числа (S, C, Q -> 0, 1, 2)
le = LabelEncoder()
df_clean['embarked'] = le.fit_transform(df_clean['embarked'])

print("После обработки данных:")
print(df_clean.dtypes)
print(f"NaN осталось: {df_clean.isnull().sum().sum()}")
print()

# Построение матрицы корреляций Пирсона
corr_matrix = df_clean.corr()

# Вывод корреляции каждого признака с целевой переменной Survived
print("Корреляция признаков с Survived (от наибольшей к наименьшей):")
corr_with_survived = corr_matrix['survived'].drop('survived').sort_values(ascending=False)
print(corr_with_survived.round(3))
print()

# Визуализация матрицы корреляций в виде тепловой карты
plt.figure(figsize=(12, 10))

# Маскируем верхний треугольник для наглядности
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Построение heatmap
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    square=True,
    linewidths=0.8,
    linecolor='white',
    cbar_kws={"shrink": 0.8, "label": "Коэффициент корреляции"},
    annot_kws={"size": 9}
)

plt.title('Матрица корреляций признаков Титаника\n(красный - положительная связь, синий - отрицательная)', 
          fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('titanic_correlation_matrix.png', dpi=200, bbox_inches='tight')
plt.show()

print("Матрица корреляций сохранена как 'titanic_correlation_matrix.png'")
print()
print("Ключевые выводы по корреляции с выживаемостью:")
print(f"  * Пол (sex = 1 женский):        {corr_with_survived['sex']:.3f} - женщины выживали чаще")
print(f"  * Класс (pclass):               {corr_with_survived['pclass']:.3f} - чем выше класс, тем выше выживаемость")
print(f"  * Стоимость билета (fare):      {corr_with_survived['fare']:.3f} - положительная связь")
print(f"  * Количество родственников (sibsp + parch): слабая связь")
print(f"  * Возраст (age):                {corr_with_survived['age']:.3f} - очень слабая отрицательная связь")
