import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Laptop Predictor")


def load_artifacts():
    try:
        with open("pipe.pkl", "rb") as pipe_file:
            pipe = pickle.load(pipe_file)
        with open("df.pkl", "rb") as df_file:
            df = pickle.load(df_file)
        return pipe, df
    except Exception as exc:
        st.error("Model files could not be loaded. This is usually a scikit-learn version mismatch.")
        st.code(
            "pip install -r requirements.txt\n"
            "python train.py",
            language="bash",
        )
        st.exception(exc)
        st.stop()


pipe, df = load_artifacts()

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu brand'].unique())

os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    if weight <= 0:
        st.error("Please enter a laptop weight greater than 0.")
        st.stop()

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size

    query = pd.DataFrame(
        [[company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]],
        columns=[
            'Company',
            'TypeName',
            'Ram',
            'Weight',
            'Touchscreen',
            'Ips',
            'ppi',
            'Cpu brand',
            'HDD',
            'SSD',
            'Gpu brand',
            'os',
        ],
    )

    prediction = int(np.exp(pipe.predict(query)[0]))
    st.success(f"The predicted price of this configuration is {prediction}")

