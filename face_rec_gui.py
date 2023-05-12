import cv2
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread
import face_recognition
import pickle
import numpy as np
import os

biden = False

def load_encodings(encoding_dir='encodings'):
    encodings = []
    names = []

    for file in os.listdir(encoding_dir):
        if file.endswith('.pkl'):
            with open(os.path.join(encoding_dir, file), 'rb') as f:
                encodings.append(pickle.load(f))
                names.append(os.path.splitext(file)[0])

    return encodings, names

def save_encoding(name, face_encoding, encoding_dir='encodings'):
    encoding_file = os.path.join(encoding_dir, name + '.pkl')

    # Create the directory if it doesn't exist
    if not os.path.exists(os.path.dirname(encoding_file)):
        os.makedirs(os.path.dirname(encoding_file))

    with open(encoding_file, 'wb') as f:
        pickle.dump(face_encoding, f)


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.cap = cv2.VideoCapture(1)

        # Create a canvas for video display
        self.canvas = tk.Canvas(window, width = 300, height = 300)
        self.canvas.pack()

        # Create buttons
        self.btn_snapshot = tk.Button(window, text="Take references", width=50, command=self.capture_references_wrapper)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        self.btn_recognize = tk.Button(window, text="Recognize face", width=50, command=self.recognize_from_webcam_wrapper)
        self.btn_recognize.pack(anchor=tk.CENTER, expand=True)

        # Start video capture
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (300, 300))

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            self.imgtk = ImageTk.PhotoImage(image=img)  # Store as an attribute of the class
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)

        self.window.after(10, self.update_video)  # Update every 10 ms

    def capture_references(self, num_images=3, delay=1, encoding_file='encodings/biden.pkl'):
        cap = cv2.VideoCapture(1)
        face_encodings = []

        print(num_images)

        faces_captured = 0

        while faces_captured < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                continue

            #cv2.waitKey(delay * 1000)  # Wait for delay seconds

            frame = cv2.imread("biden.jpg")
            frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

            face_locations = face_recognition.face_locations(rgb_frame)

            print("Found {} face locations in image {}".format(len(face_locations), faces_captured+1))

            face_encodings_in_frame = face_recognition.face_encodings(rgb_frame, face_locations)

            print("Found {} face encodings in image {}".format(len(face_encodings_in_frame), faces_captured+1))

            if len(face_encodings_in_frame) > 0:
                h, w, _ = frame.shape
                center = (h / 2, w / 2)
                distances = [(fl[0]-center[0])**2 + (fl[1]-center[1])**2 for fl in face_locations]
                closest_face_index = np.argmin(distances)

                # Add the encoding of the closest face to the list
                face_encodings.append(face_encodings_in_frame[closest_face_index])

                print("Captured image {} of {}".format(faces_captured+1, num_images))

                faces_captured += 1

        cap.release()
        cv2.destroyAllWindows()

        # Average the face encodings
        if face_encodings:
            average_face_encoding = np.mean(face_encodings, axis=0)

            # Create the directory if it doesn't exist
            if not os.path.exists(os.path.dirname(encoding_file)):
                os.makedirs(os.path.dirname(encoding_file))

            # Save the averaged face encoding
            with open(encoding_file, 'wb') as f:
                pickle.dump(average_face_encoding, f)

        pass

    def recognize_from_webcam(self, encodings, names, process_every_n_frames=4):
        cap = cv2.VideoCapture(1)
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Only process every nth frame
            if True: #frame_number % process_every_n_frames == 0:
                global biden
                if biden:
                    frame = cv2.imread("two_people.jpg")

                frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(encodings, face_encoding)

                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = names[first_match_index]

                    print(f"Found {name} in the video stream.")

            frame_number += 1

    def capture_references_wrapper(self):
        print("Capturing references")
        thread = Thread(target=self.capture_references)
        thread.start()

    def recognize_from_webcam_wrapper(self):
        print("Recognizing from webcam")
        print("Loading encodings")
        encodings, names = load_encodings()
        print("Loaded encodings")
        thread = Thread(target=self.recognize_from_webcam, args=(encodings, names))
        thread.start()

def on_keypress(event):
    if event.char == "b":
        global biden
        biden = True
    else:
        biden = False

root = tk.Tk()
root.bind('<KeyPress>', on_keypress)
app = App(root, "Face recognition")
root.mainloop()