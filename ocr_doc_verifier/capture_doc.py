import cv2
import numpy as np

def capture_document():
    url = "http://10.200.11.73:8080/video"  # replace with your phone's URL
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster processing
        frame_resized = cv2.resize(frame, (640, 480))

        # Convert to grayscale & detect edges for document outline
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        # Show live feed and edge preview
        cv2.imshow("Document Capture - Press 'c' to capture, 'q' to quit", frame_resized)
        cv2.imshow("Edges (for alignment)", edged)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cv2.imwrite("captured_document.jpg", frame)
            print("Document captured and saved as captured_document.jpg")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_document()
