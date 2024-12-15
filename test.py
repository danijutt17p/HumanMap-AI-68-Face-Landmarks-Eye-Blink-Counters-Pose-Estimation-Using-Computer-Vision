import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        # Initialize MediaPipe face mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
        self.results = self.faceMesh.process(self.imgRGB)  # Process the image with FaceMesh model
        faces = []
        if self.results.multi_face_landmarks:  # Check if faces are detected
            for faceLms in self.results.multi_face_landmarks:
                if draw:  # If draw is True, draw the landmarks on the image
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                                               self.drawSpec, self.drawSpec)  # Use FACEMESH_TESSELATION for connections
                face = []
                for id, lm in enumerate(faceLms.landmark):  # Loop through landmarks
                    ih, iw, ic = img.shape  # Get image dimensions
                    x, y = int(lm.x * iw), int(lm.y * ih)  # Convert relative coordinates to pixel coordinates
                    face.append([x, y])  # Append each landmark's coordinates
                faces.append(face)  # Append the landmarks for this face
        return img, faces  # Return the image with drawn landmarks and the list of faces' landmarks

def main():
    cap = cv2.VideoCapture('2.mp4')  # Open the video file
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    pTime = 0  # Variable to track the previous time for FPS calculation
    detector = FaceMeshDetector(maxFaces=2)  # Initialize the FaceMeshDetector class

    while True:
        success, img = cap.read()  # Read each frame from the video
        if not success:
            print("Error: Failed to read frame.")
            break

        img, faces = detector.findFaceMesh(img)  # Process the frame and detect facial landmarks

        # Print the landmarks for the first face detected
        if len(faces) != 0:
            print(faces[0])

        # Calculate FPS (frames per second)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        # Display the FPS on the frame
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        # Resize the frame for better visualization
        img = cv2.resize(img, (1280, 720))

        # Display the frame with drawn landmarks
        cv2.imshow("Face Mesh", img)

        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()
