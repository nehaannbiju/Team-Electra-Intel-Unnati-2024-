import cv2
import numpy as np
import winsound

# Load YOLOv3 model and initialize variables
net = cv2.dnn.readNet("C:\\Users\\ebin\\OneDrive\\Desktop\\intel  2024\\bed detection\\yolov3.weights", 
                      "C:\\Users\\ebin\\OneDrive\\Desktop\\intel  2024\\bed detection\\yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the COCO class labels
with open("C:\\Users\\ebin\\OneDrive\\Desktop\\intel  2024\\bed detection\\coco.names", 'r') as f:
    classes = f.read().strip().split('\n')

# Constants for abnormal activity detection
MOVEMENT_THRESHOLD = 30  # Adjust as needed based on your scenario

# Function to calculate Euclidean distance between two points
def euclidean_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# Function to play alarm sound
def play_alarm():
    winsound.Beep(1500, 500)  # Adjust frequency and duration as needed

# Initialize variables for patient detection and abnormal activity detection
previous_patient_position = None

# Open the video file
cap = cv2.VideoCapture("C:\\Users\\ebin\\OneDrive\\Desktop\\intel  2024\\video\\patient abnormal video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for detection
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to hold detection results
    boxes = []
    confidences = []
    class_ids = []

    # Process detection results
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'bed':
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to eliminate redundant overlapping boxes with lower confidences
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Process each detected bed
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            # Draw rectangle around the detected bed
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Display label as "Patient" in the bounding box
            cv2.putText(frame, "Patient", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Get coordinates for patient region (adjust as per your requirement)
            patient_box = (x, y, w, h)
            patient_center = (x + w // 2, y + h // 2)

            # Detect movement as abnormal activity
            if previous_patient_position is not None:
                if euclidean_distance(previous_patient_position, patient_center) > MOVEMENT_THRESHOLD:
                    play_alarm()  # Trigger alarm for any movement
                    print("Abnormal activity detected!")  # Print statement for abnormal activity detection
            
            # Update previous position to current
            previous_patient_position = patient_center

    # Display the frame with detected beds and patients
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
