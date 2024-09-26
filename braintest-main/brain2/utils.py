import openvino as ov
import cv2
import numpy as np
import matplotlib.pyplot as plt

core = ov.Core()

# Load and compile the face detection model
model_face = core.read_model(model='Model/face-detection-adas-0001.xml')
compiled_model_face = core.compile_model(model=model_face, device_name="CPU")

input_layer_face = compiled_model_face.input(0)
output_layer_face = compiled_model_face.output(0)

# Load and compile the emotion recognition model
model_emo = core.read_model(model='Model/emotions-recognition-retail-0003.xml')
compiled_model_emo = core.compile_model(model=model_emo, device_name="CPU")

input_layer_emo = compiled_model_emo.input(0)
output_layer_emo = compiled_model_emo.output(0)

def preprocess(image, input_layer):
    N, input_channels, input_height, input_width = input_layer.shape
    
    resized_image = cv2.resize(image, (input_width, input_height))
    transposed_image = resized_image.transpose(2, 0, 1)
    input_image = np.expand_dims(transposed_image, 0)
    
    return input_image

def find_faceboxes(image, results, confidence_threshold):
    results = results.squeeze()

    scores = results[:, 2]
    boxes = results[:, -4:]

    face_boxes = boxes[scores >= confidence_threshold]
    scores = scores[scores >= confidence_threshold]

    image_h, image_w, image_channels = image.shape
    
    face_boxes = face_boxes * np.array([image_w, image_h, image_w, image_h])
    face_boxes = face_boxes.astype(np.int64)

    return face_boxes, scores

def draw_emotion(face_boxes, image):
    show_image = image.copy()

    EMOTION_NAMES = ['neutral', 'happy', 'sad', 'surprise', 'anger']

    for i in range(len(face_boxes)):
        xmin, ymin, xmax, ymax = face_boxes[i]
        face = image[ymin:ymax, xmin:xmax]
        
        # --- Emotion ---
        input_image = preprocess(face, input_layer_emo)
        results_emo = compiled_model_emo([input_image])[output_layer_emo]
        
        results_emo = results_emo.squeeze()
        index = np.argmax(results_emo)

        text = EMOTION_NAMES[index]
        
        # Draw the text and rectangle around the face
        fontScale = image.shape[1] / 750
        cv2.putText(show_image, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 200, 0), 8)
        cv2.rectangle(img=show_image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 200, 0), thickness=2)

    return show_image

def predict_image(image, conf_threshold):
    input_image = preprocess(image, input_layer_face)
    results = compiled_model_face([input_image])[output_layer_face]
    face_boxes, scores = find_faceboxes(image, results, conf_threshold)
    visualized_image = draw_emotion(face_boxes, image)

    return visualized_image
