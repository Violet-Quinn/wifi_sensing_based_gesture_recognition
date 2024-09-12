import streamlit as st
import pandas as pd
import re
from math import sqrt, atan2
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="Gesture Recognition using WiFi Sensing",
    page_icon="📡",  # Antenna icon for WiFi sensing
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the trained model and imputer
model_filename = r'D:\Wifi Sensing Project\Deployment\trained_model.joblib'
imputer_filename = r'D:\Wifi Sensing Project\Deployment\imputer1.joblib'

loaded_classifier = joblib.load(model_filename)
loaded_imputer = joblib.load(imputer_filename)

# Check if the loaded model is a k-NN or Random Forest model
model_type = "Unknown"
if isinstance(loaded_classifier, KNeighborsClassifier):
    model_type = "k-NN Classifier"
elif isinstance(loaded_classifier, RandomForestClassifier):
    model_type = "Random Forest Classifier"

# Display model type and parameters
st.subheader("Model Information:")
st.write(f"Model Type: {model_type}")

if model_type == "k-NN Classifier":
    st.write(f"Number of Neighbors (k): {loaded_classifier.n_neighbors}")
    st.write(f"Distance Metric: {loaded_classifier.metric}")
elif model_type == "Random Forest Classifier":
    st.write(f"Number of Trees: {loaded_classifier.n_estimators}")
    st.write(f"Max Depth of Trees: {loaded_classifier.max_depth}")

# Display information about the imputer
st.subheader("Imputer Information:")
st.write(f"Imputation Strategy: Median (default)")
st.write(f"Number of Features in Imputer: {loaded_imputer.statistics_.shape[0]}")

# Set background color and general styling
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f5f5f5;  /* Light gray background */
            color: #333333;  /* Dark gray text */
        }
        .st-bj {
            padding: 0rem;
        }
        .st-ch {
            background-color: #ffffff;  /* White container background */
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);  /* Light shadow for containers */
        }
        .st-cx {
            max-width: 1200px;  /* Limit the width of the content */
        }
        .st-da {
            padding-top: 2rem;  /* Adjust top padding */
        }
        .st-cq {
            margin-bottom: 2rem;  /* Adjust bottom margin */
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def breathe_rate(filename):
    d = pd.read_csv(filename)

    max_values = []

    for i, j in enumerate(d['data']):
        imaginary = []
        real = []

        csi_string = re.findall(r"\[(.*)]", j)[0]
        csi_raw = [int(x) for x in csi_string.split(",") if x != '']

        for k in range(0, len(csi_raw), 2):
            imaginary.append(csi_raw[k])
            real.append(csi_raw[k + 1])

        amp = [round(sqrt(imaginary[k] ** 2 + real[k] ** 2), 1) for k in range(len(imaginary))]
        max_values.append(max(amp))

    # Plot the graph
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    ax.plot(max_values, marker='o', linestyle='-', color='#3498db')  # Blue line graph with markers

    # Set labels and title
    ax.set_xlabel('Data Row')
    ax.set_ylabel('Maximum Value')
    ax.set_title('Maximum Value from Each Data Row')

    # Customize the plot grid lines and background color
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('#ecf0f1')  # Light gray background

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Show the plot
    st.pyplot(fig)


def process_single_csv(filename):
    d = pd.read_csv(filename)

    amps = []
    phases = []

    for i, j in enumerate(d['data']):
        imaginary = []
        real = []
        amp = []
        ph = []

        csi_string = re.findall(r"\[(.*)]", j)[0]
        csi_raw = [int(x) for x in csi_string.split(",") if x != '']

        for k in range(0, len(csi_raw), 2):
            imaginary.append(csi_raw[k])
            real.append(csi_raw[k + 1])

        for k in range(len(imaginary)):
            amp.append(round(sqrt(imaginary[k] ** 2 + real[k] ** 2), 1))
            ph.append(round(atan2(imaginary[k], real[k])))

        amps.append(amp)
        phases.append(ph)

    # Combine amps and phases into a single list
    result_list = [item for sublist in (amps + phases) for item in sublist]

    return result_list


def classify_new_file(filename, true_label):
    # Process the new file
    result_list = process_single_csv(filename)

    # Convert the result list to a DataFrame
    new_data_df = pd.DataFrame([result_list])

    # Print the number of features in the new data
    st.write("Number of features in the new data:", new_data_df.shape[1])

    # Check if the number of features matches the expectation
    if new_data_df.shape[1] != loaded_imputer.statistics_.shape[0]:
        # Fill missing values with zeros
        new_data_df = new_data_df.reindex(columns=range(loaded_imputer.statistics_.shape[0]), fill_value=0)

    # Impute missing values with the median (you can choose a different strategy based on your data)
    new_data_array = loaded_imputer.transform(new_data_df)

    # Make predictions with the loaded classifier
    prediction = loaded_classifier.predict(new_data_array)

    # Print the predicted label and confidence scores
    # st.subheader("Prediction Information:")
    # st.write("Predicted Label:", prediction[0])

    # Calculate accuracy if true label is provided
    # if true_label is not None:
    #    accuracy = accuracy_score([true_label], [prediction[0]])
    #    st.write(f"Accuracy: {accuracy:.2%}")

    return prediction[0]


def main():
    try:
        st.image("wifi_logo.png", width=150)
    except FileNotFoundError:
        st.warning("Unable to open 'wifi_logo.png'. Using a placeholder image.")
        st.image("placeholder_image.png", width=150)

    st.title("Gesture Recognition using WiFi Sensing")
    st.markdown(
        "WiFi sensing is an innovative approach to gesture recognition. By analyzing the variations in WiFi signals, "
        "this technology can detect and interpret gestures made by users."
        "This Streamlit app allows you to explore WiFi"
        "sensing data, visualize maximum values, and predict gestures using a trained machine learning model."
    )

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        st.subheader("Choose an Action:")
        action = st.radio("Select", ["Show Graph", "Predict Gesture"])

        if action == "Show Graph":
            breathe_rate(uploaded_file)

        elif action == "Predict Gesture":
            # Remove the following line for entering true label
            # true_label = st.text_input("Enter the true label (e.g., 'yes', 'thankyou', 'bye', 'hello'):")

            # Modify the function call to pass None as the true_label
            predicted_label = classify_new_file(uploaded_file, None)

            # Display gif based on predicted label
            if predicted_label == "yes":
                st.image("yes.png", caption="Yes Gesture", width=150)
            elif predicted_label == "thankyou":
                st.image("thankyou.png", caption="Thank You Gesture", width=150)
            elif predicted_label == "bye":
                st.image("bye.png", caption="Bye Gesture", width=150)
            elif predicted_label == "hello":
                st.image("hello.png", caption="Hello Gesture", width=150)


if __name__ == "_main_":
    main()
