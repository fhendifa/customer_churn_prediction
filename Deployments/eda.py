import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

def run():
    # Membuat Title
    st.title('Exploratory Data Analysis (EDA)')

    # Membuat Garis Lurus
    st.markdown('---')
    
    # Show dataframe
    data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    st.dataframe(data)

    st.write('#### Churn')
    fig = px.pie(data_frame = data ,names = data.Churn.value_counts().index,values = data.Churn.value_counts().values[0:10],hole = 0.7)
    st.plotly_chart(fig)

    st.write('#### Churn vs Gender')
    fig = plt.figure(figsize = (5,5))
    ax = sns.countplot(x='Churn',hue='gender',data=data)
    st.plotly_chart(fig)

    st.write('#### Churn vs Senior Citizen')
    fig = plt.figure(figsize = (5,5))
    ax = sns.countplot(x='Churn',hue='SeniorCitizen',data=data)
    st.plotly_chart(fig)

    st.write('### Bar Chart')
    opt1 = st.selectbox('Select Columns : ', ('gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'))
    fig = plt.figure(figsize = (13,5))
    ax = sns.countplot(x=data[opt1], data=data)
    ax.bar_label(ax.containers[0])
    st.pyplot(fig)

if __name__ == '__main__':
    run()