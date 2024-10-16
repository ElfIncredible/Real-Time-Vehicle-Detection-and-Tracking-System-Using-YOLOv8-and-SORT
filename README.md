# Real-Time Vehicle Detection and Tracking System Using YOLOv8 and SORT
This project implements a real-time vehicle detection and tracking system using the YOLOv8 object detection model and the SORT (Simple Online and Realtime Tracking) algorithm. It processes video footage to identify and track vehicles such as cars, trucks, buses, and motorbikes. The system overlays bounding boxes, displays unique IDs for tracked vehicles, and counts the total number of vehicles crossing a specified line. It is optimized for real-time performance and can be applied to traffic monitoring, surveillance, and smart city applications.

https://github.com/user-attachments/assets/9020f70c-f75e-4dc0-96ea-4d6a89471bff

## Table of contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Computer Vision](#computer-vision)
  - [Library Imports](#library-imports)
  - [Video Capture Initialization](video-capture-initialization)
  - [Class Names Definition](#class-names-definition)
  - [Mask Image Loading](#mask-image-loading)
  - [Tracking Initialization](#tracking-initialization)
  - [Limits for Vehicle Counting](#limits-for-vehicle-counting)
  - [Main Processing Loop](#main-processing-loop)
  - [Visualization](#visualization)
  - [Display Output](#display-output)
- [Results](#results)
- [Impact](#impact)
- [Future Improvements](#future-improvements)

## Project Overview
The Real-Time Vehicle Detection and Tracking System is designed to monitor and analyze traffic using computer vision and deep learning techniques. This project utilizes the YOLOv8 (You Only Look Once) model for real-time object detection, which is highly efficient in recognizing and classifying multiple vehicle types such as cars, trucks, buses, and motorbikes in video streams.

Once vehicles are detected, the SORT (Simple Online and Realtime Tracking) algorithm is employed to track the movement of individual vehicles across frames by assigning unique IDs. This allows for consistent tracking, even when multiple vehicles appear simultaneously. Additionally, the system includes a counting mechanism that tracks the number of vehicles crossing a predefined boundary, which is useful for traffic flow analysis.

Key features of the system include:
- **Real-Time Performance:** The system processes live video streams with minimal latency.
- **Multiple Vehicle Types Detection:** Cars, trucks, buses, and motorbikes are easily recognized and tracked.
- **Visual Overlays:** Bounding boxes, vehicle IDs, and other visual aids overlay the video for intuitive monitoring.
- **Vehicle Counting:** The system counts vehicles passing through a defined virtual line, providing valuable data for traffic monitoring.
- **Custom Masking:** Masking can be used to specify a region of interest and focus the detection on specific areas.

This project has wide-ranging applications, including traffic monitoring systems, smart city infrastructure, and surveillance systems for optimizing road usage and safety.

## Problem Statement
Efficient traffic management and monitoring are critical challenges in urban areas, where increasing vehicle numbers contribute to congestion, accidents, and pollution. Traditional traffic surveillance methods often rely on manual monitoring or static sensors, which can be resource-intensive, prone to human error, and limited in data accuracy.

There is a growing need for an automated, real-time solution that can accurately detect and track vehicles, providing data to improve traffic flow analysis, optimize road usage, and enhance safety measures. Existing solutions may struggle with issues like high latency, inaccurate object tracking, or inability to differentiate between various vehicle types.

The challenge is to develop a system that can reliably detect, track, and count vehicles in real-time from video footage while ensuring high accuracy, scalability, and adaptability to various environments. This system must be able to handle multiple vehicle types, provide real-time analytics, and integrate seamlessly into existing traffic management infrastructures.

## Dataset
The dataset consists of a video stream capturing real-world urban traffic, including various vehicle types such as cars, trucks, buses, and motorbikes. It represents realistic conditions with varying lighting, vehicle speeds, and occasional occlusions.

Key features:
- **Vehicle Variety:** Cars, trucks, buses, and motorbikes are included for comprehensive detection and tracking.
- **Real-World Environment:** The footage captures dynamic traffic scenarios, ideal for testing real-time object detection models.
- **Masked Region:** A region of interest is defined to focus the detection on specific areas.

## Computer Vision
### Library Imports
The code begins by importing essential libraries, including YOLO from the Ultralytics package for object detection, cv2 for image processing, cvzone for overlaying graphics, math for mathematical operations, and SORT for vehicle tracking.

### Video Capture Initialization
The video capture is set up using OpenCV's VideoCapture function, which reads the specified video file containing the traffic scene.

### Model Loading
The YOLOv8 model is loaded with pre-trained weights (yolov8l.pt), enabling the system to detect and classify vehicles in the video frames.

### Class Names Definition
A list of class names is defined, representing various object categories that the model can recognize, including vehicles and other objects.

### Mask Image Loading
A mask image (Mask.png) is loaded to define the area of interest in the video, allowing for focused detection and processing.

### Tracking Initialization
The SORT tracker is initialized to keep track of detected vehicles over multiple frames, with parameters for maximum age, minimum hits, and IoU threshold.

### Limits for Vehicle Counting
A predefined line is set to define the boundaries for vehicle counting. Vehicles crossing this line are counted, providing traffic flow data.

### Main Processing Loop
The core of the program runs in a loop that:
- Reads frames from the video.
- Applies the mask to focus on the region of interest.
- Detects vehicles using the YOLO model and retrieves bounding box coordinates and confidence scores.
- Updates the tracker with the detected vehicles.

### Visualization
For each detected vehicle:
- Bounding boxes and unique IDs are drawn on the frame.
- Vehicle center points are marked with circles.
- The total count of vehicles crossing the predefined line is displayed.

### Display Output
The processed frames are displayed in a window, showing real-time vehicle detection and counting results. The program continues until interrupted by the user.

## Results
The implementation of the Real-Time Vehicle Detection and Tracking System has demonstrated significant results in accurately detecting and tracking vehicles in urban traffic scenarios. Key outcomes include:
- **High Detection Accuracy:** The YOLOv8 model effectively identifies various vehicle types, achieving a high level of accuracy even in challenging conditions such as varying lighting and occlusions.
- **Real-Time Tracking:** The integration of the SORT algorithm allows for consistent tracking of vehicles across frames, with unique IDs assigned to each vehicle, enabling reliable monitoring of their movements.
- **Effective Vehicle Counting:** The system accurately counts the number of vehicles crossing a predefined line, providing valuable data for traffic analysis and management.
- **Visual Analytics:** The overlay of bounding boxes, vehicle IDs, and count displays on the video feed offers intuitive insights into traffic flow, making it easier for operators to monitor and analyze real-time conditions.

## Impact
The deployment of this vehicle detection and tracking system has several impactful applications:
- **Traffic Management:** It can assist city planners and traffic management centers in analyzing traffic patterns, identifying congestion hotspots, and optimizing traffic signal timings.
- **Safety Enhancements:** By providing real-time data on vehicle movement, the system can help in improving road safety measures and reducing accidents through timely interventions.
- **Smart City Integration:** The technology aligns with smart city initiatives, enabling the integration of automated traffic monitoring solutions that enhance urban mobility and efficiency.
- **Research and Development:** This project serves as a foundation for further research in autonomous driving systems, transportation studies, and AI-based traffic analysis tools, contributing to advancements in smart transportation technologies.

Overall, the results indicate a successful application of deep learning and computer vision technologies in addressing urban traffic challenges, paving the way for smarter, safer, and more efficient transportation systems.

## Future Improvements
- **Model Optimization:** Explore advanced object detection models and optimization techniques (e.g., pruning and quantization) to enhance accuracy and reduce inference time, particularly in complex traffic scenarios.
- **Enhanced Multi-Object Tracking:** Implement more sophisticated tracking algorithms, such as Deep SORT or ByteTrack, to improve vehicle tracking accuracy, especially in crowded environments with occlusions.
- **Integration of Additional Sensors:** Combine the system with other data sources like LiDAR or radar to provide comprehensive traffic data and improve overall detection accuracy and situational awareness.
- **Development of a Real-Time Analytics Dashboard:** Create a user-friendly dashboard for real-time traffic analytics and visualization, empowering traffic management authorities with insights and alerts for proactive decision-making.
