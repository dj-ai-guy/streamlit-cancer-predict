import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# normally you would import the data but i pasted a function to get the data  
# and clean it 
def get_clean_data():
    data = pd.read_csv("data/data.csv")

    # dropping Unnamed 32 & id because it has no data and id is not needed
    data = data.drop(columns=['Unnamed: 32', 'id'], axis=1)
    
    # Encoding the diagnosis column. M = 1 B = 0 
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    return data

# Adds a sidebar to the webpage 
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    # Calling the get_clean_data function 
    data = get_clean_data()

    # Getting the names of the slider variables from 
    # the columns of the dataset 
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    # create a dictionary to store the dataset column name and the average value 
    # of that column this will be used to chart predictions later
    input_dict = {}

    # loop through all of the labels and column names and create one slider for each of them 
    for label, key in slider_labels:
        # this puts the slider in the sidebar 
        input_dict[key] = st.sidebar.slider(
            label,
            min_value= float(0),
            max_value=float(data[key].max()),# the key is the 2nd arg in the loop and will be the associated column
            value=float(data[key].mean()) # sets all the sliders to the average value
        )
    return input_dict

# Create a function to scale the slider numbers to get them in a range between 0-1
def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict

# defining the function to create the radar chart using plotly
# copied format from https://plotly.com/python/radar-chart/ - Multiple Trace Radar Chart
def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig

# create a function to make predictions based on the input data 
# this will predicit if the tumor is cancerous or not 
def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb")) # import the model from pickle
    scaler = pickle.load(open("model/scaler.pkl", "rb")) # same with the scaler

    # Taking the input array dictionary and creating a numpy array of only the values 
    input_array = np.array(list(input_data.values())).reshape(1,-1)

    # scale the values of the input data 
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")
    
    if prediction[0] == 0:
        st.write("Benign")
    else:
        st.write("Malicious")

    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def main():
    # set the page configuration 
    st.set_page_config(
        page_title='Breast Cancer Predictor',
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
        )
    
    

    # create the function to add a sidebar 
    input_data = add_sidebar()

    # this is just to test to see if the input data is collecting the data 
    # as the user slides the sliders
    # st.write(input_data)

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")
    
    # create columns 
    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data) # creating a variable to hold the radar chart that is returned from the function
        st.plotly_chart(radar_chart) # passing the variable to plotly chart so it displays in streamlit
    with col2:
        add_predictions(input_data)



if __name__ == '__main__':
    main()