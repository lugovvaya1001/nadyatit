import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score)

# ============================================================================
# 1. ЗАГРУЖАЕМ ДАННЫЕ
# ============================================================================
df = pd.read_csv('titanic.csv')

# ============================================================================
# 2. СОЗДАЁМ НОВЫЕ ПРИЗНАКИ ДЛЯ УЛУЧШЕНИЯ ПРЕДСКАЗАНИЙ
# ============================================================================

# Размер семьи (пассажир + сиблинги/супруг + родители/дети)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Одинок ли пассажир (размер семьи = 1)
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Извлекаем обращение из имени (Mr, Mrs, Miss, Master и т.д.)
# Это помогает определить социальный статус пассажира
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Превращаем обращения в числа
title_mapping = {
    'Mr': 0,           # мужчина
    'Miss': 1,         # незамужняя женщина
    'Mrs': 2,          # замужняя женщина
    'Master': 3,       # мальчик
    'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 1,
    'Countess': 2, 'Ms': 1, 'Lady': 2, 'Jonkheer': 4,
    'Don': 4, 'Dona': 2, 'Mme': 2, 'Capt': 4, 'Sir': 4
}
df['Title'] = df['Title'].map(title_mapping).fillna(4).astype(int)

# Разбиваем возраст на категории (дети, подростки, взрослые, пожилые)
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 45, 60, 100], 
                       labels=[0, 1, 2, 3, 4, 5])

# Логарифм от стоимости билета (чтобы сгладить выбросы)
df['LogFare'] = np.log1p(df['Fare'])

# Женщина в первом или втором классе? (очень сильный признак выживания)
df['FemaleHighClass'] = ((df['Sex'] == 'female') & (df['Pclass'] <= 2)).astype(int)

# Ребёнок до 12 лет?
df['IsChild'] = (df['Age'] <= 12).fillna(0).astype(int)

# Удаляем колонки, которые больше не нужны
df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'SibSp', 'Parch', 'Age', 'Fare'], axis=1)

# ============================================================================
# 3. ВЫБИРАЕМ ПРИЗНАКИ ДЛЯ ОБУЧЕНИЯ
# ============================================================================
features = ['Pclass', 'Sex', 'Embarked', 'FamilySize', 'IsAlone', 
            'Title', 'AgeBin', 'LogFare', 'FemaleHighClass', 'IsChild']
X = df[features]          # Матрица признаков
y = df['Survived']        # Целевая переменная (1 - выжил, 0 - погиб)

# ============================================================================
# 4. РАЗДЕЛЯЕМ ДАННЫЕ НА ОБУЧАЮЩУЮ (80%) И ТЕСТОВУЮ (20%) ВЫБОРКИ
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================================================
# 5. ПРЕОБРАЗУЕМ ПРИЗНАКИ В ФОРМАТ, ПОНЯТНЫЙ НЕЙРОСЕТИ
# ============================================================================

# Числовые признаки (их нужно просто масштабировать)
numeric_features = ['FamilySize', 'IsAlone', 'Title', 'LogFare', 'FemaleHighClass', 'IsChild']

# Категориальные признаки (их нужно превратить в 0 и 1)
categorical_features = ['Pclass', 'Sex', 'Embarked', 'AgeBin']

# Создаём пайплайн для числовых признаков
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),   # заполняем пропуски средним
    ('scaler', StandardScaler())                     # приводим к одному масштабу
])

# Создаём пайплайн для категориальных признаков
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # заполняем пропуски самым частым значением
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))  # превращаем в 0 и 1
])

# Объединяем оба пайплайна
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Применяем преобразования к данным
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ============================================================================
# 6. СОЗДАЁМ И ОБУЧАЕМ НЕЙРОСЕТЬ (MLP)
# ============================================================================

# MLP = Multi-Layer Perceptron (многослойный перцептрон)
# Это простая нейросеть прямого распространения
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),   # два скрытых слоя: 100 и 50 нейронов
    activation='relu',               # функция активации ReLU
    solver='adam',                   # оптимизатор Adam
    max_iter=500,                    # максимум эпох обучения
    random_state=42                  # фиксируем случайность для воспроизводимости
)

# Обучаем нейросеть на подготовленных данных
mlp.fit(X_train_processed, y_train)

# ============================================================================
# 7. ДЕЛАЕМ ПРЕДСКАЗАНИЯ НА ТЕСТОВОЙ ВЫБОРКЕ
# ============================================================================
y_pred = mlp.predict(X_test_processed)                    # класс (0 или 1)
y_pred_proba = mlp.predict_proba(X_test_processed)[:, 1]  # вероятность (от 0 до 1)

# ============================================================================
# 8. СТРОИМ МАТРИЦУ ОШИБОК (Confusion Matrix)
# ============================================================================
# Показывает: сколько правильных и неправильных предсказаний по каждому классу
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Погиб', 'Выжил'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Матрица ошибок')
plt.show()

# ============================================================================
# 9. СТРОИМ ROC-КРИВУЮ
# ============================================================================
# Показывает, насколько хорошо модель разделяет классы
# Чем ближе AUC к 1, тем лучше
plt.figure(figsize=(6, 5))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', label='Случайное угадывание')
plt.xlabel('Ложноположительные срабатывания (1 - специфичность)')
plt.ylabel('Истинноположительные срабатывания (чувствительность)')
plt.title('ROC-кривая')
plt.legend()
plt.show()

# ============================================================================
# 10. СТРОИМ МАТРИЦУ МЕТРИК (Accuracy, Precision, Recall, F1, ROC-AUC)
# ============================================================================
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),   # Общая точность
    'Precision': precision_score(y_test, y_pred), # Точность предсказания "выжил"
    'Recall': recall_score(y_test, y_pred),       # Полнота (сколько выживших нашли)
    'F1-Score': f1_score(y_test, y_pred),         # Среднее гармоническое Precision и Recall
    'ROC-AUC': roc_auc                            # Площадь под ROC-кривой
}

plt.figure(figsize=(8, 5))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), hue=list(metrics.keys()), legend=False, palette='viridis')
plt.ylim(0, 1)
plt.title('Матрица метрик качества модели')
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
plt.show()

# ============================================================================
# 11. ВЫВОДИМ РЕЗУЛЬТАТЫ В КОНСОЛЬ
# ============================================================================
print(f"\n{'='*50}")
print(f"ТОЧНОСТЬ МОДЕЛИ: {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.1f}%)")
print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
print(f"{'='*50}\n")

print("ПОДРОБНЫЙ ОТЧЕТ ПО КАЖДОМУ КЛАССУ:")
print(classification_report(y_test, y_pred, target_names=['Погиб (0)', 'Выжил (1)']))
