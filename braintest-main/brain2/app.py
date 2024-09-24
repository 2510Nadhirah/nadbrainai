import streamlit as st
import PIL
import cv2
import numpy as np
import utils
import io

# Set page configuration at the beginning
st.set_page_config(
    page_title="EmoBuddy",
    page_icon="💅",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for pink-themed styling
st.markdown("""
    <style>
    .main {
        background-color: #ffeef8;
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #ff69b4;
        margin-bottom: 20px;
        text-align: center;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #ff1493;
        margin-top: 20px;
        border-bottom: 2px solid #ff1493;
        padding-bottom: 10px;
    }
    .button {
        background-color: #ff1493;
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
        background-color: #ff69b4;
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
        color: #d63e6d;
        font-size: 16px;
    }
    .success-message {
        color: #ff1493;
        font-size: 18px;
    }
    .warning-message {
        color: #e74c3c;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Define the camera_input_live function
def camera_input_live():
    cap = cv2.VideoCapture(0)  # Start capturing video from webcam
    ret, frame = cap.read()     # Read a frame
    cap.release()               # Release the webcam
    if ret:
        _, buffer = cv2.imencode('.jpg', frame)  # Encode the frame as a jpg
        return io.BytesIO(buffer)                 # Return as BytesIO for PIL to open
    else:
        return None

def play_video(video_source):
    camera = cv2.VideoCapture(video_source)
    st_frame = st.empty()
    
    while camera.isOpened():
        ret, frame = camera.read()

        if ret:
            visualized_image = utils.predict_image(frame, conf_threshold)  # Change to emotion detection
            st_frame.image(visualized_image, channels="BGR")
        else:
            camera.release()
            break

def play_live_camera():
    image = camera_input_live()
    if image is not None:
        uploaded_image = PIL.Image.open(image)
        uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold)  # Change to emotion detection
        st.image(visualized_image, channels="BGR")

st.title("EmoBuddy 💅")

st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "VIDEO", "WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider(
    "Select the Confidence Threshold", 10, 100, 20)) / 100

if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg", "png"))

    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold)  # Change to emotion detection

        st.image(visualized_image, channels="BGR")
        st.image(uploaded_image)
    else:
        st.image("braintest-main/brain2/assets/userpage.png")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an image.")

if source_radio == "VIDEO":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose a video.", type=("mp4"))

    if input is not None:
        g = io.BytesIO(input.read())
        temporary_location = "upload.mp4"

        with open(temporary_location, "wb") as out:
            out.write(g.read())

        out.close()
        play_video(temporary_location)
        
        if st.button("Replay"):
            play_video(temporary_location)
    else:
        st.video("braintest-main/brain2/assets/uservidd.mp4")
        st.write("Click on 'Browse Files' in the sidebar to run inference on a video.")

if source_radio == "WEBCAM":
    play_live_camera()

