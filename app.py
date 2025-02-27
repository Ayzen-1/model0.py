import streamlit as st
import pandas as pd
import joblib

# Настройки страницы
st.set_page_config(page_title="ML Model Deployment", page_icon="🔮", layout="centered")

# 1. Загрузка модели
def load_model():
    # Загружает модель из файла model.pkl
    model = joblib.load("model.pkl")
    return model

model = load_model()

# 2. Загрузка датасета
columns = [
    'WSR0', 'WSR1', 'WSR2', 'WSR3', 'WSR4', 'WSR5', 'WSR6', 'WSR7', 'WSR8', 'WSR9', 'WSR10',
    'WSR11', 'WSR12', 'WSR13', 'WSR14', 'WSR15', 'WSR16', 'WSR17', 'WSR18', 'WSR19', 'WSR20', 'WSR21', 'WSR22', 'WSR23',
    'WSR_PK', 'WSR_AV',
    'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17',
    'T18', 'T19', 'T20', 'T21', 'T22', 'T23',
    'T_PK', 'T_AV', 'T85', 'RH85', 'U85', 'V85', 'HT85',
    'T70', 'RH70', 'U70', 'V70', 'HT70',
    'T50', 'RH50', 'U50', 'V50', 'HT50',
    'KI', 'TT', 'SLP', 'SLP_', 'Precp',
]

df = pd.read_excel("output.xlsx", names=columns)
st.write("✅ Датасет успешно загружен!")

# 3. Основной интерфейс приложения
st.title('🔮 Деплой ML-модели')
st.write("Введите данные через ползунки, чтобы получить предсказание.")

# Боковая панель для ввода признаков
st.sidebar.header("Ввод признаков")

# Определяем диапазоны для признаков T14 и T13 из датасета
t14_min = float(df['T14'].min())
t14_max = float(df['T14'].max())
t14_mean = float(df['T14'].mean())

t13_min = float(df['T13'].min())
t13_max = float(df['T13'].max())
t13_mean = float(df['T13'].mean())

# Слайдеры для ввода значений
feat_t14 = st.sidebar.slider("Выберите значение для T14", t14_min, t14_max, t14_mean)
feat_t13 = st.sidebar.slider("Выберите значение для T13", t13_min, t13_max, t13_mean)

# Формирование DataFrame с введёнными значениями
input_data = pd.DataFrame({
    'T14': [feat_t14],
    'T13': [feat_t13]
})

# Функция для отображения графиков
def show_graphs():
    st.subheader("Гистограмма T14")
    # Отображаем гистограмму для T14
    st.bar_chart(df['T14'].value_counts().sort_index())

    st.subheader("Гистограмма T13")
    # Отображаем гистограмму для T13
    st.bar_chart(df['T13'].value_counts().sort_index())

    st.subheader("Scatter plot: T14 vs T13")
    # Отображаем scatter plot для T14 и T13
    st.scatter_chart(df[['T14', 'T13']])

# Функция для отображения статистики по признакам
def display_statistics():
    st.subheader("Статистика по признакам")
    st.write("#### T14")
    st.write(df['T14'].describe())
    st.write("#### T13")
    st.write(df['T13'].describe())

# Кнопка для получения предсказания
if st.button("🔍 Предсказать"):
    prediction = model.predict(input_data)
    st.success(f"📢 Результат предсказания: {prediction[0]}")

# Возможность показать датасет
if st.checkbox("Показать датасет"):
    st.dataframe(df)

# Разворачиваем блок с графиками
with st.expander("Показать графики"):
    show_graphs()

# Разворачиваем блок со статистикой
with st.expander("Показать статистику"):
    display_statistics()
