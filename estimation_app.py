import streamlit as st
from PIL import Image
import numpy as np
import cv2

DEMO_IMAGE = 'stand.jpg'

BODY_PARTS = { 
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

width = 368
height = 368
inWidth = width
inHeight = height

net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Sidebar
st.sidebar.title("Pose Estimation Settings")
st.sidebar.markdown("Adjust the settings to customize the results.")

st.title("📷 Human Pose Estimation with OpenCV")
st.markdown("**Upload an image and detect human poses using OpenCV's pre-trained model.**")

img_file_buffer = st.file_uploader(
    "📤 Upload an image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
else:
    st.sidebar.info("Using demo image for now. Upload an image to test with your data.")
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))

st.subheader("Original Image")
st.image(image, caption="This is the uploaded image", use_container_width=True)

thres = st.sidebar.slider(
    "Confidence Threshold for Keypoint Detection",
    min_value=0,
    max_value=100,
    value=20,
    step=5,
    help="Adjust the threshold to control the sensitivity of pose detection."
)

thres = thres / 100

@st.cache_data
def poseDetector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    
    assert len(BODY_PARTS) == out.shape[1]
    points = []
    
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)
    
    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    
    t, _ = net.getPerfProfile()
    cv2.putText(frame, f"Inference time: {t / 1000:.2f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

output = poseDetector(image)

st.subheader("Positions Estimated")
st.image(output, caption="Pose Detection Result", use_container_width=True)

st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
- Upload a clear image of a person for accurate pose detection.
- Adjust the confidence threshold for better results.
""")
