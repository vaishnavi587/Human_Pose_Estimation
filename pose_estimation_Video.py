import cv2
import numpy as np

# Body parts and pose pairs definition
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

# Load pre-trained model
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Function for pose estimation
def pose_estimation(video_path, threshold=0.2, display_scale=0.5):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    
    print("Press 'q' to exit the video.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame for display scaling
        frame = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
        frame_height, frame_width = frame.shape[:2]
        
        # Process the frame through the neural network
        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]
        
        # List to store detected points
        points = []
        
        for i in range(len(BODY_PARTS)):
            heatmap = out[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatmap)
            x = int((frame_width * point[0]) / out.shape[3])
            y = int((frame_height * point[1]) / out.shape[2])
            points.append((x, y) if conf > threshold else None)
        
        # Draw detected key points and lines
        for pair in POSE_PAIRS:
            part_from = pair[0]
            part_to = pair[1]
            
            id_from = BODY_PARTS[part_from]
            id_to = BODY_PARTS[part_to]
            
            if points[id_from] and points[id_to]:
                cv2.line(frame, points[id_from], points[id_to], (0, 255, 0), 3)
                cv2.circle(frame, points[id_from], 5, (0, 0, 255), thickness=-1)
                cv2.circle(frame, points[id_to], 5, (0, 0, 255), thickness=-1)
        
        # Display the frame
        cv2.imshow("Pose Estimation", frame)
        
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Call the function
video_path = "run1.mp4"
pose_estimation(video_path)
