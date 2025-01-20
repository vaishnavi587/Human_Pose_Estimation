import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

width = 368
height = 368
inWidth = width
inHeight = height

net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
thres = 0.2

def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(math.degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    return angle

def poseDetector(frame, line_color=(0, 255, 0), point_color=(0, 0, 255)):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    assert(len(BODY_PARTS) == out.shape[1])
    
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        if conf > thres:
            points.append((int(x), int(y)))
            cv2.putText(frame, list(BODY_PARTS.keys())[i], (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            points.append(None)
    
    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], line_color, 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, point_color, cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, point_color, cv2.FILLED)
    
    # Example: Right Shoulder, Elbow, Wrist
    if points[2] and points[3] and points[4]:
        angle = calculate_angle(points[2], points[3], points[4])
        cv2.putText(frame, f'{int(angle)}°', (points[3][0] + 10, points[3][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    
    return frame

input = cv2.imread('stand.jpg')
output = poseDetector(input)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Pose Detection Output')
plt.show()
