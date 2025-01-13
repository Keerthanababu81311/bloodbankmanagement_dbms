import cv2
import mediapipe as mp
from flask import Flask, jsonify, request
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# Function to process the image and return pose landmarks
def get_pose_landmarks(image):
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    landmarks = []
    if result.pose_landmarks:
        for lm in result.pose_landmarks.landmark:
            landmarks.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
    return landmarks


# Route to handle the image input and pose estimation
@app.route('/process', methods=['POST'])
def process_image():
    # Get the image from the POST request (base64 encoded)
    data = request.json.get('image')
    img_data = base64.b64decode(data)

    # Convert the image data to a format that OpenCV can read
    img = Image.open(BytesIO(img_data))
    img = np.array(img)

    # Process the image to extract pose landmarks
    landmarks = get_pose_landmarks(img)

    # Return the landmarks as JSON response
    return jsonify(landmarks)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
