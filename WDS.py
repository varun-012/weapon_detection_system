import cv2
import numpy as np

# Load YOLO model and class names
net = cv2.dnn.readNet(r"D:\BTECH CLASSES\7)SEVENTH SEMESTER\dip project\yolov3_training_2000.weights",
                    r"D:\BTECH CLASSES\5) FIFTH SEMESTER\UROP\weapondetection FINAL\yolov3_testing.cfg")
classes = ["Weapon"]
output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Function to display the welcome message and menu
def display_menu():
    print("Welcome to the Weapon Detection System")
    print("Please select an option : ")
    print("1. Image")
    print("2. Video")
    print("3. Live Webcam Feed")
    choice = input("Enter your choice (1/2/3) : ")
    return choice

# Function to detect weapons in the input frame
def detect_weapon(img):
    # Perform detection
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(
        img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(
                    detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    # Resize the output image to fit a fixed width (e.g., 800 pixels)
    max_width = 800
    scale = max_width / width
    resized_img = cv2.resize(img, (max_width, int(height * scale)))

    # Display the resized image
    cv2.imshow("Weapon Detection", resized_img)


# Main function to handle user input and detection
def main():
    choice = display_menu()

    if choice == '1':
        img_path = input("Enter the path/name of the image file: ")
        img = cv2.imread(img_path)
        if img is not None:
            detect_weapon(img)
            cv2.waitKey(0)
        else:
            print("Error: Could not read the image file.")
    elif choice == '2':
        video_path = input("Enter the path/name of the video file: ")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return
        while True:
            ret, img = cap.read()
            if not ret:
                break
            detect_weapon(img)
            if cv2.waitKey(1) == 27 or cv2.getWindowProperty("Weapon Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
    elif choice == '3':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        while True:
            ret, img = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break
            detect_weapon(img)
            if cv2.waitKey(1) == 27 or cv2.getWindowProperty("Weapon Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
    else:
        print("Invalid choice. Please restart the program and select a valid option.")

    cv2.destroyAllWindows()

# Main Function
if __name__ == "__main__":
    main()

# End of Program