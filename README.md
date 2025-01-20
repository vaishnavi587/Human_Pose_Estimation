# Human Pose Estimation using Machine Learning

This project demonstrates **Human Pose Estimation (HPE)**, which involves detecting and tracking key points on the human body (like joints and limbs) in images or videos using machine learning. 


## Tech Stack

- Python
- OpenCV (for computer vision tasks)
- Streamlit (for interactive web interface)
- TensorFlow (for machine learning model)
- NumPy (for array manipulations)
- PIL (for image handling)

## Features

- Upload an image to detect human poses.
- Adjustable confidence threshold for detecting key points.
- Display detected key points and the skeleton of the human pose.

## Installation

1. Clone the repository:
   git clone https://github.com/your-username/human-pose-estimation.git
    
2. Install required packages:
    pip install -r requirements.txt
    
3. Download the pre-trained TensorFlow model (`graph_opt.pb`) and place it in the project directory.

4. Run the app:
    streamlit run estimation_app.py
   
5. Access the app in your browser.

## How It Works

1. Model Loading: Loads the TensorFlow model (`graph_opt.pb`) using OpenCV.
2. Key Points Detection: Detects key points on the human body.
3. Pose Visualization: Displays key points and connects them to form a human pose skeleton.

## Hosted URL

You can access the live demo at:
URL: https://humanposeestimationusingmachinelearning-idxeaetrc8ufm4stm8xjqs.streamlit.app/

## Applications

- Sports Analytics: Improving athlete performance.
- Healthcare: Tracking rehabilitation progress.
- Human-Computer Interaction: Gesture recognition.
- Security: Monitoring and tracking individuals.
- Entertainment & Gaming: Realistic character animation.

