import streamlit as st
import PIL
import cv2
import numpy as np
import utils
import io

# Set page configuration at the beginning
st.set_page_config(
    page_title="EmoBuddy",
    page_icon="💚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for a health-focused theme
st.markdown("""
    <style>
    .main {
        background-color: #e6f7e6;
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #4caf50;
        margin-bottom: 20px;
        text-align: center;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #388e3c;
        margin-top: 20px;
        border-bottom: 2px solid #388e3c;
        padding-bottom: 10px;
    }
    .button {
        background-color: #388e3c;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        text-align: center;
        display: inline-block;
        cursor: pointer;
        border: none;
        margin-top: 10px;
    }
    .button:hover {
        background-color: #4caf50;
    }
    .container {
        margin-top: 20px;
    }
    .file-uploader {
        margin: 20px 0;
    }
    .input-section {
        margin-bottom: 20px;
    }
    .info-text {
        color: #2e7d32;
        font-size: 16px;
    }
    .success-message {
        color: #388e3c;
        font-size: 18px;
    }
    .warning-message {
        color: #e74c3c;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Add a header image or logo related to health
st.image("braintest-main/brain2/assets/logo.png", width=200)  # Ensure you have a logo at this path

# Title of the app
st.title("EmoBuddy 💚")
st.write("### Enhancing Emotional Communication with AI for Better Mental Health")

# Sidebar for user interaction
st.sidebar.header("Select Input Type")
source_radio = st.sidebar.radio("Choose Source", ["IMAGE", "VIDEO", "WEBCAM"])

st.sidebar.header("Confidence Level")
conf_threshold = float(st.sidebar.slider(
    "Select the Confidence Threshold", 10, 100, 20)) / 100

def play_video(video_source):
    camera = cv2.VideoCapture(video_source)
    st_frame = st.empty()
    
    while camera.isOpened():
        ret, frame = camera.read()

        if ret:
            visualized_image = utils.predict_image(frame, conf_threshold)
            st_frame.image(visualized_image, channels="BGR")
        else:
            camera.release()
            break

def play_live_camera():
    camera = cv2.VideoCapture(0)  # Start capturing video from webcam
    if not camera.isOpened():
        st.error("Webcam not accessible. Please check your camera settings.")
        return

    st_frame = st.empty()  # Create a placeholder for the video frame

    while True:
        ret, frame = camera.read()  # Read a frame

        if not ret:
            st.error("Failed to capture image from webcam. Please check your camera.")
            break
        
        # Process the frame for emotion detection
        visualized_image = utils.predict_image(frame, conf_threshold)
        
        # Display the processed frame
        st_frame.image(visualized_image, channels="BGR")
        
        # Allow breaking the loop with a button
        if st.button("Stop Webcam"):
            camera.release()
            break
    
    camera.release()

if source_radio == "IMAGE":
    st.sidebar.header("Upload Image")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg", "png"))

    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold)

        st.image(visualized_image, channels="BGR", caption="Processed Image")
        st.image(uploaded_image, caption="Uploaded Image")
    else:
        st.image("braintest-main/brain2/assets/emotions.png", caption="Upload an image to analyze.")

if source_radio == "VIDEO":
    st.sidebar.header("Upload Video")
    input = st.sidebar.file_uploader("Choose a video.", type=("mp4"))

    if input is not None:
        g = io.BytesIO(input.read())
        temporary_location = "upload.mp4"

        with open(temporary_location, "wb") as out:
            out.write(g.read())

        play_video(temporary_location)
        
        if st.button("Replay"):
            play_video(temporary_location)
    else:
        st.video("braintest-main/brain2/assets/emobuddy.mp4)  # Removed caption
        st.write("Upload a video to analyze.")

if source_radio == "WEBCAM":
    st.write("### Live Webcam Feed")
    st.write("Click the button below to start the webcam.")
    play_live_camera()
    
    if st.button("Stop Webcam"):
        st.write("Webcam stopped. Click 'WEBCAM' to start again.")

# Add a footer with information about SDG 3
st.markdown("""
    <footer>
        <p style="text-align: center; color: #4caf50;">Improving emotional communication for better mental health. &copy; [4I1E]</p>
    </footer>
""", unsafe_allow_html=True)
