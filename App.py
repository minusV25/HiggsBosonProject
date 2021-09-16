import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from PIL import Image



model = tf.keras.models.load_model('higgs_boson_classifier.h5')


#import the dataset
higgs = pd.read_csv('Dataset.csv')

def prediction(model,input):
    prediction = model.predict(input)
    print('prediction successful')
    return 's' if prediction[0][0] >= 0.5 else 'b'

def proba(model,input):
    proba = model.predict(input)
    print('probability successful')
    return proba


col = [['DER_mass_MMC', 'DER_mass_transverse_met_lep', '',
       'DER_pt_h', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep',
       'DER_pt_tot','DER_pt_ratio_lep_tau','DER_met_phi_centrality','PRI_tau_pt',
       'PRI_tau_eta','PRI_tau_phi','PRI_lep_eta','PRI_lep_phi','PRI_met_phi','PRI_jet_leading_phi']]

#correlation matrix
corrMatrix = higgs.corr()

def main():
    primaryColor= "#c7d2d3"
    backgroundColor="##396469"
    secondaryBackgroundColor="#cce1e3"
    textColor="#cadcbf"
    font="sans serif"
    base= "dark"
    
    
    
    st.header('Higgs Boson Event Detection')
    
    
    image = Image.open('C:\Users\Admin\Documents\Higgs Boson\07-aug_higgs.jpg')
    st.image(image, caption='Higgs Boson Collider')
    st.write('Sample of the Higgs Boson Data:')
    st.table(higgs)

    st.write('Correlation between each column:')
    fig, ax = plt.subplots(figsize=(30,30))
    sns.heatmap(corrMatrix, annot=True, linewidths=.5, ax=ax)
    st.pyplot(fig)
    
    
    st.subheader('Input the Data')
    st.write('Please input the data below')

    i = st.number_input('DER_mass_MMC',)
    j = st.number_input('DER_mass_transverse_met_lep',)
    k = st.number_input('DER_pt_h',)
    l = st.number_input('DER_prodeta_jet_jet',)
    m = st.number_input('DER_deltar_tau_lep',)
    n = st.number_input('DER_pt_tot',)
    o = st.number_input('DER_pt_ratio_lep_tau',)
    p = st.number_input('DER_met_phi_centrality',)
    q = st.number_input('PRI_tau_pt',)
    r = st.number_input('PRI_tau_eta',)
    s = st.number_input('PRI_tau_phi',)
    t = st.number_input('PRI_lep_eta',)
    u = st.number_input('PRI_lep_phi',)
    v = st.number_input('PRI_met_phi',)
    w = st.number_input('PRI_jet_leading_phi',)


    input = np.array([[i,j,k,l,m,n,o,p,q,r,s,t,u,v,w]])
    print(type(i))
    print(input)
    
    
    if st.button('Detect Event'):
        pred = prediction(model,input)        
        st.success('The event is predicted is ' + pred)

    if st.button('Show Probability'):
        prob = proba(model,input)
        st.success('The probability of the event is {}'.format(prob[0][0]))

if __name__ == '__main__':

    main()
