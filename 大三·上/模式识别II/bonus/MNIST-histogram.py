import streamlit as st
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Load dataset
@st.cache_resource
def load_data():
    data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
    return data

# Use cache to load data
pixel_values, _ = load_data()
pixel_array = pixel_values.to_numpy()

# Cache calculation results and print specific coordinates
@st.cache_data
def calculate_histograms(pixel_array):
    histograms = np.zeros((28, 28, 256))
    with open('coordinates.txt', 'w') as file:
        for image in pixel_array:
            for i in range(28):
                for j in range(28):
                    pixel_value = image[i * 28 + j]
                    histograms[i, j, int(pixel_value)] += 1

        # Find coordinates where 95% of grayscale is concentrated in 0-42 and write to file
        for i in range(28):
            for j in range(28):
                total_pixels = np.sum(histograms[i, j, :])
                cumulative_frequency_0_42 = np.sum(histograms[i, j, :30])
                if cumulative_frequency_0_42 / total_pixels >= 0.99:
                    file.write(f"{i},{j}\n")

    return histograms

# Calculate and cache the histogram
histograms = calculate_histograms(pixel_array)

# Streamlit's interactive part
st.title('MNIST Dataset Pixel Intensity Histogram')

# Position range
pixel_row = st.slider('Row', 0, 27, 0)
pixel_col = st.slider('Column', 0, 27, 0)

# Draw histogram
fig, ax = plt.subplots()
ax.bar(range(256), histograms[pixel_row, pixel_col, :], width=1)
st.pyplot(fig)
