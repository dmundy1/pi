from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
from threading import Thread
import face_recognition
import picamera.array
import tkinter as tk
import numpy as np
import picamera
import pickle
import time
import csv
import cv2
import os

encodings = []
names = []
camera = picamera.PiCamera()
global_username = "<undefined>"

def load_encodings(encoding_dir='encodings'):
    global encodings, names

    for file in os.listdir(encoding_dir):
        if file.endswith('.pkl'):
            with open(os.path.join(encoding_dir, file), 'rb') as f:
                encodings.append(pickle.load(f))
                names.append(os.path.splitext(file)[0])

def get_encodings():
    global encodings, names
    return encodings, names

def save_encoding(name, face_encoding, encoding_dir='encodings'):
    encoding_file = os.path.join(encoding_dir, name + '.pkl')

    if not os.path.exists(os.path.dirname(encoding_file)):
        os.makedirs(os.path.dirname(encoding_file))

    with open(encoding_file, 'wb') as f:
        pickle.dump(face_encoding, f)
    
    load_encodings()

def window_size():
    return (800, 480)

def validate_pin(pin):
    return not (pin and (len(pin) > 6 or not pin.isdigit()))

def check_user(username, pin):
    with open("users.txt", "r") as f:
        for line in f:
            if line.split(",")[0] == username or line.split(",")[1].strip() == pin:
                return False
            
    return True

def add_user(username, pin, encoding=None):
    if not check_user(username, pin):
        return False
    
    with open("users.txt", "a") as f:
        f.write("{},{}\n".format(username, pin))

    return True

def get_username_from_pin(pin):
    with open("users.txt", "r") as f:
        for line in f:
            if line.split(",")[1].strip() == pin:
                return line.split(",")[0]
            
    return None

def get_video_frame():
    global camera
    
    with picamera.array.PiRGBArray(camera) as output:
        camera.capture(output, 'rgb')
        original_frame = cv2.resize(output.array, window_size())
        resized_frame = cv2.resize(original_frame, (0, 0), fx=0.25, fy=0.25)
            
        return original_frame, resized_frame

def update_attendance(username):
    filename = 'hours.txt'
    current_time = int(time.time())
    new_rows = []
    user_found = False
    total_seconds = 0

    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == username:
                    user_found = True
                    if row[2] == '':
                        row[2] = current_time
                        total_seconds = int(row[2]) - int(row[1])
                    else:
                        new_rows.append([username, current_time, ''])
                        total_seconds = 0
                new_rows.append(row)
    except FileNotFoundError:
        pass

    if not user_found:
        new_rows.append([username, current_time, ''])
        total_seconds = 0

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    return total_seconds

class App:
    def __init__(self, window, window_title):
        load_encodings()

        self.window = window
        self.window.title(window_title)

        self.container = tk.Frame(window)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, LoginPage, ConfirmLoginPage, SignedInPage, SignupPage, ReferencePage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

        self.active_frame = frame

        if hasattr(frame, "on_show_frame"):
            frame.on_show_frame()

    def periodic(self):
        if type(self.active_frame) == LoginPage or type(self.active_frame) == ReferencePage:
            self.active_frame.update_frame()

        self.window.after(10, self.periodic)

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        container = tk.Frame(self)
        container.place(relx=0.5, rely=0.5, anchor="center")

        login_button = tk.Button(container, text="Go to Login Page", font=("Helvetica", 24),
                            command=lambda: controller.show_frame("LoginPage"))
        
        login_button.pack(fill="x", pady=10)
        
        signup_button = tk.Button(container, text="Go to Signup Page", font=("Helvetica", 24),
                            command=lambda: controller.show_frame("SignupPage"))
        
        login_button.pack(fill="x")
        signup_button.pack(fill="x")

    def on_show_frame(self):
        global global_username
        global_username = "<undefined>"

class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        container = tk.Frame(self)
        container.place(relx=0.5, rely=0.5, anchor="center")        

        self.canvas = tk.Canvas(container, width=window_size()[0] - 100, height=window_size()[1] - 100)
        self.canvas.pack(pady=10)

        label = tk.Label(container, text="Please look at the camera.")
        label.pack()

        home_button = tk.Button(self, text="Home", command=lambda: self.controller.show_frame("StartPage"))
        home_button.place(relx=1-(5 / window_size()[0]), rely=1-(5 / window_size()[1]), anchor="se")

    def update_detection(self, frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(encodings, face_encoding)

            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = names[first_match_index]

            if name != "Unknown":
                global global_username
                global_username = name
                self.controller.show_frame("ConfirmLoginPage")

    def update_frame(self):
        frames = get_video_frame()
        img = Image.fromarray(np.ascontiguousarray(frames[0][:, :, ::-1]))
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        self.canvas.image = imgtk

        self.update_detection(np.ascontiguousarray(frames[1][:, :, ::-1]))

class ConfirmLoginPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        container = tk.Frame(self)
        container.place(relx=0.5, rely=0.5, anchor="center")

        self.label = tk.Label(container, text="Are you <undefined>?", font=("Helvetica", 24), anchor="center", justify="center")
        self.label.pack(fill="x", pady=10)

        yes_button = tk.Button(container, text="Yes",
                               command=self.login)
        no_button = tk.Button(container, text="No",
                              command=self.ask_for_pin)
        yes_button.pack()
        no_button.pack()

        home_button = tk.Button(self, text="Home", command=lambda: controller.show_frame("StartPage"))
        home_button.place(relx=1-(5 / window_size()[0]), rely=1-(5 / window_size()[1]), anchor="se")

    def on_show_frame(self):
        global global_username
        self.label.config(text="Are you {}?".format(global_username))

    def ask_for_pin(self):
        dialog = tk.Toplevel(self)
        dialog.title("Pin")
        label = tk.Label(dialog, text="Enter your 6-digit pin:")
        label.pack(side="top", fill="x", pady=10)
        pin_entry = tk.Entry(dialog, validate="key", validatecommand=(dialog.register(validate_pin), "%P"))
        pin_entry.pack()
        button = tk.Button(dialog, text="OK", command=lambda: self.check_pin(pin_entry.get(), dialog))
        button.pack()

    def login(self):
        self.controller.show_frame("SignedInPage")

    def check_pin(self, pin, dialog):
        print(pin)
        dialog.destroy()
        global global_username
        name = get_username_from_pin(pin)
        if name != None:
            global_username = name
            self.login()
        else:
            dialog = tk.Toplevel(self)
            dialog.title("Error")
            label = tk.Label(dialog, text="Invalid pin.")
            label.pack(side="top", fill="x", pady=10)
            button = tk.Button(dialog, text="OK", command=lambda: (dialog.destroy(), self.ask_for_pin()))
            button.pack()

class SignedInPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        container = tk.Frame(self)
        container.place(relx=0.5, rely=0.5, anchor="center")

        self.label = tk.Label(container, text="Hello <undefined>, you are now <undefined>!", font=("Helvetica", 24), anchor="center", justify="center")
        self.label.pack(fill="x", pady=10)

        home_button = tk.Button(self, text="Home", command=lambda: controller.show_frame("StartPage"))
        home_button.place(relx=1-(5 / window_size()[0]), rely=1-(5 / window_size()[1]), anchor="se")

    def on_show_frame(self):
        global global_username
  
        minutes = update_attendance(global_username) / 60
        signed_in = minutes == 0
        label = "Hello {}, you are now {}!".format(global_username, "signed in" if signed_in else "signed out")
        label += "\n\nYou clocked a total of {} minutes.".format(round(minutes)) if not signed_in else ""
        self.label.config(text=label)

        self.after(3000, lambda: self.controller.show_frame("StartPage"))

class SignupPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        container = tk.Frame(self)
        container.place(relx=0.5, rely=0.5, anchor="center")

        username_label = tk.Label(container, text="Username", font=("Helvetica", 18))
        username_label.grid(row=0, column=0)
        self.username_entry = tk.Entry(container, font=("Helvetica", 28))
        self.username_entry.grid(row=1, column=0)

        pin_label = tk.Label(container, text="Pin", font=("Helvetica", 18))
        pin_label.grid(row=2, column=0)
        self.pin_entry = tk.Entry(container, font=("Helvetica", 28), show="*", validate="key", validatecommand=(self.register(validate_pin), "%P"))
        self.pin_entry.grid(row=3, column=0)

        tk.Label(container).grid(row=4, column=0) # dummy label

        button = tk.Button(container, text="Signup", font=("Helvetica", 20),
                           command=self.signup)
        button.grid(row=5, column=0)

        home_button = tk.Button(self, text="Home", command=lambda: self.controller.show_frame("StartPage"))
        home_button.place(relx=1-(5 / window_size()[0]), rely=1-(5 / window_size()[1]), anchor="se")

    def show_failed_signup(self, reason=None):
        dialog = tk.Toplevel(self)
        dialog.title("Error")
        label = tk.Label(dialog, text=f"Account creation failed: {reason}" if reason else "Account creation failed for an unknown reason")
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(dialog, text="OK", command=dialog.destroy)
        button.pack()

    def signup(self):
        username = self.username_entry.get()
        pin = self.pin_entry.get()

        if not username or not pin:
            self.show_failed_signup("Username or pin is empty")
            return

        if not validate_pin(pin) or len(pin) != 6:
            self.show_failed_signup("Pin is not 6 digits or contains non-numbers")
            return

        if not add_user(username, pin):
            self.show_failed_signup("Username or pin already exists")
            return

        global global_username
        global_username = username

        dialog = tk.Toplevel(self)
        dialog.title("Success (1/2)")
        label = tk.Label(dialog, text="Profile created successfully, moving on to reference capture.")
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(dialog, text="OK", command=dialog.destroy)
        button.pack()

        self.after(2000, lambda: self.controller.show_frame("ReferencePage"))

class ReferencePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        container = tk.Frame(self)
        container.place(relx=0.5, rely=0.5, anchor="center")

        self.canvas = tk.Canvas(container, width=window_size()[0] - 100, height=window_size()[1] - 100)
        self.canvas.pack(pady=10)

        label = tk.Label(container, text="Please look at the camera.")
        label.pack()

        home_button = tk.Button(self, text="Home", command=lambda: self.controller.show_frame("StartPage"))
        home_button.place(relx=1-(5 / window_size()[0]), rely=1-(5 / window_size()[1]), anchor="se")

    def on_show_frame(self):
        self.face_encodings = []

    def update_detection(self, frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings_in_frame = face_recognition.face_encodings(frame, face_locations)

        if len(face_encodings_in_frame) > 0:
            h, w, _ = frame.shape
            center = (h / 2, w / 2)
            distances = [(fl[0]-center[0])**2 + (fl[1]-center[1])**2 for fl in face_locations]
            closest_face_index = np.argmin(distances)

            self.face_encodings.append(face_encodings_in_frame[closest_face_index])

        print(len(self.face_encodings))

        if len(self.face_encodings) != 10:
            return

        global global_username
        save_encoding(global_username, np.mean(self.face_encodings, axis=0))

        dialog = tk.Toplevel(self)
        dialog.title("Success (2/2)")
        label = tk.Label(dialog, text="Reference capture successful, please log in.")
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(dialog, text="OK", command=dialog.destroy)
        button.pack()

        self.after(2000, lambda: self.controller.show_frame("StartPage"))

    def update_frame(self):
        frames = get_video_frame()
        img = Image.fromarray(np.ascontiguousarray(frames[0][:, :, ::-1]))
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        self.canvas.image = imgtk

        self.update_detection(np.ascontiguousarray(frames[1][:, :, ::-1]))

root = tk.Tk()
root.geometry(f"{window_size()[0]}x{window_size()[1]}")
app = App(root, "Face recognition")
root.after(0, app.periodic)
root.mainloop()