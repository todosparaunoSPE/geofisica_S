# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:07:39 2025

@author: jperezr
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from obspy.io.segy.segy import _read_segy
from obspy.signal.filter import bandpass
from scipy.signal import coherence, welch
from scipy.fft import fft, fftfreq
#from scipy.signal import cwt, morlet
from scipy.signal.wavelets import cwt, morlet


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

# Estilo de fondo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background:
radial-gradient(black 15%, transparent 16%) 0 0,
radial-gradient(black 15%, transparent 16%) 8px 8px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
background-color:#282828;
background-size:16px 16px;
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Función para aplicar filtro de paso de banda
def apply_bandpass_filter(data, sampling_rate, freqmin, freqmax):
    if freqmin <= 0 or freqmax <= 0:
        raise ValueError("Las frecuencias críticas del filtro deben ser mayores que 0.")
    
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = bandpass(data[i, :], freqmin=freqmin, freqmax=freqmax, df=sampling_rate, corners=4, zerophase=True)
    return filtered_data

# Función para plotear datos sísmicos en forma de mapa de calor de amplitud
def plot_seismic_heatmap(data, sampling_rate, cmap='seismic'):
    fig, ax = plt.subplots()
    im = ax.imshow(data.T, aspect='auto', cmap=cmap)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trace number')
    fig.colorbar(im, ax=ax, label='Amplitude')
    st.pyplot(fig)

# Función para calcular y plotear la coherencia entre trazas sísmicas seleccionadas
def plot_coherence(data, trace1, trace2, sampling_rate):
    f, Cxy = coherence(data[trace1, :], data[trace2, :], fs=sampling_rate)
    fig, ax = plt.subplots()
    ax.semilogy(f, Cxy, color='blue')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Coherence')
    ax.set_title(f'Coherence between trace {trace1} and trace {trace2}')
    st.pyplot(fig)

# Función para plotear el espectrograma de wavelet
def plot_wavelet_spectrogram(data, trace_idx, sampling_rate):
    dt = 1 / sampling_rate
    widths = np.arange(1, 31)
    cwt_matrix = cwt(data[trace_idx, :], morlet, widths)
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(cwt_matrix), aspect='auto', extent=[0, len(data[trace_idx, :])*dt, 1, 31], cmap='jet')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Scale')
    plt.title(f'Wavelet Spectrogram for trace {trace_idx}')
    st.pyplot(plt)

# Función para plotear la PSD promediada en el tiempo
def plot_time_averaged_psd(data, trace_idx, sampling_rate, window_size, overlap, method):
    nperseg = int(window_size)
    noverlap = int(overlap * nperseg)
    f, Pxx = welch(data[trace_idx, :], fs=sampling_rate, nperseg=nperseg, noverlap=noverlap, average=method)
    plt.figure(figsize=(8, 4))
    plt.semilogy(f, Pxx, color='blue')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title(f'Time-Averaged PSD for trace {trace_idx}')
    st.pyplot(plt)

# Función para calcular la energía total de cada traza
def calculate_total_energy(data):
    total_energy = np.sum(np.square(data), axis=1)
    return total_energy

# Función para plotear histograma de amplitudes máximas
def plot_max_amplitude_histogram(data):
    max_amplitudes = np.max(np.abs(data), axis=1)
    plt.figure(figsize=(8, 6))
    plt.hist(max_amplitudes, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Max Amplitude')
    plt.ylabel('Frequency')
    plt.title('Distribution of Maximum Amplitudes across Traces')
    st.pyplot(plt)

# Función para plotear visualización 3D de datos sísmicos
def plot_3d(data):
    fig = px.scatter_3d(x=data[:, 0], y=data[:, 1], z=data[:, 2], color=data[:, 2],
                        labels={'x': 'X', 'y': 'Y', 'z': 'Z'},
                        title='Visualización 3D de Datos Sísmicos')
    fig.update_layout(width=900, height=700)  # Ajustar el tamaño del gráfico
    st.plotly_chart(fig)

# Función para realizar análisis de componentes principales (PCA)
def plot_pca_analysis(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    pca = PCA()
    pca.fit(scaled_data)
    
    components = pca.transform(scaled_data)
    explained_variance_ratio = pca.explained_variance_ratio_
    
    pca_results = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(components.shape[1])])
    pca_results['Explained Variance Ratio'] = explained_variance_ratio[:components.shape[0]]  # Ajusta la longitud aquí
    
    st.subheader('Resultados del Análisis de Componentes Principales (PCA)')
    st.write(pca_results)

# Función para mostrar la sección de Ayuda
def show_help():
    st.sidebar.subheader('Ayuda')
    st.sidebar.markdown("""
        Este es un ejemplo de aplicación para visualización y análisis de datos sísmicos desde archivos SEGY.
        
        ### Funcionalidades:
        - **Visualización 2D como Mapa de Calor:** Muestra los datos sísmicos como un mapa de calor de amplitud.
        - **Filtro de Paso de Banda:** Permite aplicar filtros de paso de banda a los datos sísmicos.
        - **Análisis de Coherencia:** Calcula y muestra la coherencia entre trazas sísmicas seleccionadas.
        - **Análisis Espectral Avanzado:** Incluye PSD promediada en el tiempo y espectrograma de wavelet.
        - **Análisis de Amplitud y Energía:** Visualiza la energía total y la distribución de amplitudes máximas.
        - **Visualización 3D:** Muestra una visualización 3D de los datos sísmicos.
        - **Análisis de Componentes Principales (PCA):** Realiza un análisis de componentes principales.
        
        Utiliza las opciones en la barra lateral para interactuar con las diferentes funcionalidades.
    """)

# Función principal
def main():
    st.title('Visualización y Análisis de Datos Sísmicos desde archivo SEGY')
    
    uploaded_file = st.file_uploader("Cargar archivo SEGY", type=["segy", "sgy"])
    if uploaded_file is not None:
        st.write("Archivo cargado exitosamente:", uploaded_file.name)
        
        # Procesar archivo SEGY
        st.write("Cargando datos sísmicos...")
        st.write("(Esto puede tomar unos momentos dependiendo del tamaño del archivo)")
        
        # Usar ObsPy para leer el archivo SEGY
        st.write("Leyendo datos sísmicos...")
        segy_stream = _read_segy(uploaded_file)
        
        # Obtener datos y muestreo del archivo SEGY
        sampling_rate = segy_stream.traces[0].header.sample_interval_in_ms_for_this_trace * 0.001
        data = np.stack([trace.data for trace in segy_stream.traces], axis=0)
        
        # Mostrar el plot 2D utilizando Streamlit como mapa de calor de amplitud
        st.subheader('Visualización 2D como Mapa de Calor de Amplitud')
        plot_seismic_heatmap(data, sampling_rate)
        
        # Mostrar resumen estadístico de los datos sísmicos
        st.subheader('Resumen Estadístico de los Datos Sísmicos')
        df_stats = pd.DataFrame(data.T, columns=['Trace ' + str(i) for i in range(data.shape[0])])
        st.write(df_stats.describe())
        
        # Mostrar sección de Ayuda
        show_help()
        
        # Resto del código como antes...
        
        # Agregar funcionalidades adicionales en la barra lateral
        st.sidebar.title('Opciones de Visualización y Análisis')
        
        # Filtros de Paso de Banda
        st.sidebar.subheader('Filtro de Paso de Banda')
        freqmin = st.sidebar.number_input('Frecuencia Mínima (Hz)', min_value=0.1, max_value=500.0, value=1.0, key='freqmin')
        freqmax = st.sidebar.number_input('Frecuencia Máxima (Hz)', min_value=freqmin, max_value=500.0, value=50.0, key='freqmax')
        
        if st.sidebar.button('Aplicar Filtro'):
            if freqmin <= 0 or freqmax <= 0:
                st.sidebar.error("Las frecuencias críticas del filtro deben ser mayores que 0.")
            else:
                st.sidebar.success("Filtro aplicado correctamente.")
                filtered_data = apply_bandpass_filter(data, sampling_rate, freqmin, freqmax)
                st.subheader(f'Datos Filtrados de {freqmin} - {freqmax} Hz')
                plot_seismic_heatmap(filtered_data, sampling_rate)
        
        # Análisis de Coherencia
        st.sidebar.subheader('Análisis de Coherencia')
        trace1 = st.sidebar.number_input('Seleccione la Trazza 1', min_value=0, max_value=data.shape[0]-1, value=0, key='trace1')
        trace2 = st.sidebar.number_input('Seleccione la Trazza 2', min_value=0, max_value=data.shape[0]-1, value=1, key='trace2')
        
        if st.sidebar.button('Calcular Coherencia'):
            plot_coherence(data, trace1, trace2, sampling_rate)
        
        # Espectrograma de Wavelet
        st.sidebar.subheader('Espectrograma de Wavelet')
        wavelet_trace_idx = st.sidebar.number_input('Seleccione Trazza', min_value=0, max_value=data.shape[0]-1, value=0, key='wavelet_trace_idx')
        
        if st.sidebar.button('Mostrar Espectrograma'):
            plot_wavelet_spectrogram(data, wavelet_trace_idx, sampling_rate)
        
        # PSD promediada en el tiempo
        st.sidebar.subheader('PSD Promediada en el Tiempo')
        psd_trace_idx = st.sidebar.number_input('Seleccione Trazza', min_value=0, max_value=data.shape[0]-1, value=0, key='psd_trace_idx')
        window_size = st.sidebar.slider('Tamaño de la Ventana', min_value=128, max_value=2048, value=256, step=128, key='window_size')
        overlap = st.sidebar.slider('Superposición', min_value=0.0, max_value=0.5, value=0.25, step=0.05, key='overlap')
        method = st.sidebar.selectbox('Método de Promedio', ['mean', 'median'], key='method')
        
        if st.sidebar.button('Calcular PSD'):
            plot_time_averaged_psd(data, psd_trace_idx, sampling_rate, window_size, overlap, method)
        
        # Análisis de Amplitud y Energía
        st.sidebar.subheader('Análisis de Amplitud y Energía')
        if st.sidebar.button('Calcular Energía Total'):
            total_energy = calculate_total_energy(data)
            st.write('Energía Total por Trazza:')
            st.write(pd.DataFrame(total_energy, columns=['Energía']))
        
        if st.sidebar.button('Mostrar Histograma de Amplitudes Máximas'):
            plot_max_amplitude_histogram(data)
        
        # Visualización 3D
        st.sidebar.subheader('Visualización 3D de Datos Sísmicos')
        if st.sidebar.button('Mostrar Visualización 3D'):
            plot_3d(data)
        
        # Análisis de Componentes Principales (PCA)
        st.sidebar.subheader('Análisis de Componentes Principales (PCA)')
        if st.sidebar.button('Realizar PCA'):
            plot_pca_analysis(data)
            
            
# Aviso de derechos de autor
st.sidebar.markdown("""
    ---
    © 2024. Todos los derechos reservados.
    Creado por jahoperi
""")            
            

if __name__ == "__main__":
    main()
