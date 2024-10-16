from ultralytics import YOLO  # for object detection
import cv2  # for image/video processing
import cvzone  # for easier OpenCV utilities like drawing rectangles, putting text, etc.
import math  # for calculations
from sort import *  # for object tracking

# Load the video
cap = cv2.VideoCapture("../Videos/cars.mp4")

# Load the YOLO model with the specified weights(we are using the YOLOv8 large model for detection)
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Class names for YOLO detection (list of COCO dataset classes)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# mask image to filter regions of interest
mask = cv2.imread('Mask.png')

# Initialize the tracker using the Sort algorithm using custom parameters
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define crossing limits for counting objects(Coordinates for a horizontal line & List to store unique IDs of counted vehicles)
limits = [400, 297, 673, 297]
totalCount = []

# Main loop for video processing
while True:
    # Read each frame from the video and apply the mask to the frame to focus on a specific region
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    # Overlay graphics on top of the video frame and Run object detection on the masked region using YOLO, stream=True allows streaming results
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))  # Create an empty array to store detections

    # Iterate through the detection results and extract bounding boxes from the detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding box coordinates (x1, y1, x2, y2), Convert to integers and Calculate width and height of the bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Extract confidence score and round it to 2 decimal places
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class name corresponding to the detected object
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Filter detections to only include vehicles with confidence greater than 0.3
            # Create array with bounding box coordinates and confidence
            if (currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike") and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))  # Append the current detection to the array

    # Update tracker with the new detections
    # Draw the counting line on the video
    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # Loop through tracked objects (detections),
    # Extract bounding box coordinates and ID from the tracker,
    # Convert coordinates to integers
    # Print the result (for debugging)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)

        # Draw bounding box with a unique ID
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, offset=10, thickness=3)

        # Calculate center of the bounding box
        # Draw a small circle at the center
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if the object crosses the counting line
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:  # If object hasn't been counted yet
                totalCount.append(id)  # Add object ID to total count
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)  # Change line color

    # Display the total count on the video
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # Show the final image with the overlayed graphics and detected objects
    cv2.imshow("Image", img)
    cv2.waitKey(1)  # Wait for 1 millisecond before displaying the next frame
