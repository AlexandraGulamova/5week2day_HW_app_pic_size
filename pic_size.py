import numpy as np
import streamlit as st
from skimage import io, color


# Загрузка изображения
uploaded_pic = st.sidebar.file_uploader('Загрузи изображение', type=['png', 'jpg'])

if uploaded_pic is None:
    st.write('Изображение не было загружено')
    st.stop()

# Чтение изображения через skimage
image_raw = io.imread(uploaded_pic)
image2 = color.rgb2gray(image_raw)  # Преобразуем в черно-белое изображение

# Отображаем изображение и его размер
st.image(image2, caption='Оригинальное изображение', use_column_width=True, clamp=True)
st.write(f'Размер изображения: {image2.shape}')

# Ввод количества сингулярных чисел
top_k = st.sidebar.number_input('Введите топ k сингулярных чисел (чем меньше число, тем сильнее сжатие)', min_value=1, max_value=min(image2.shape), value=10)

# SVD разложение и сжатие
U, sing_vals, V = np.linalg.svd(image2, full_matrices=False)
sigma = np.zeros((U.shape[1], V.shape[0]))
np.fill_diagonal(sigma, sing_vals)

# Обрезаем до top_k компонент
trunc_U = U[:, :top_k]
trunc_sigma = sigma[:top_k, :top_k]
trunc_V = V[:top_k, :]

# Восстановленное изображение после сжатия
compressed_image = np.dot(trunc_U, np.dot(trunc_sigma, trunc_V))

# Отображаем восстановленное изображение
st.image(compressed_image, caption=f'Сжатое изображение с {top_k} сингулярных чисел ', use_column_width=True, clamp=True)


