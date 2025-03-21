import cv2
import numpy as np

# Define fire color range in HSV
lower_fire = np.array([0, 120, 200], dtype=np.uint8)
upper_fire = np.array([35, 255, 255], dtype=np.uint8)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for fire color
    mask = cv2.inRange(hsv, lower_fire, upper_fire)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each detected fire contour
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Ignore small detections
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw bounding box around fire
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"Fire: ({x}, {y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Print fire coordinates to terminal
            print(f"🔥 Fire detected at: x={x}, y={y}, width={w}, height={h}")

    # Show output
    cv2.imshow("Fire Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
