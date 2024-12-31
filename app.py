from flask import Flask, request, render_template, Response
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sqlite3
import joblib
from io import StringIO
import cv2
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
import csv
import io
import base64
import matplotlib
import logging

# Use Agg backend for Matplotlib to avoid GUI-related issues
matplotlib.use("agg")
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Configuration parameters
NIMGS = 50
DATETODAY = date.today().strftime("%d_%m_%y")
DATETODAY2 = date.today().strftime("%d-%B-%Y")
FACE_DETECTOR_PATH = "classifiers/haarcascade_frontalface_default.xml"
MODEL_PATH = "models/face_recognition_model.pkl"
METRICS_PATH = "models/metrics.pkl"
DB_PATH = f"Attendance/attendance_{DATETODAY}.db"
LOG_PATH = "logs/app.log"

# Initialize face detector
face_detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)

# Ensure necessary directories exist
for dir_path in ["Attendance", "static", "static/faces", "logs", "models"]:
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

# Initialize database connection
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

# Create attendance table if it doesn't exist
c.execute(
    """
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prenom TEXT NOT NULL,
        emp_id INTEGER NOT NULL,
        arrivee TEXT,
        depart TEXT,
        date TEXT NOT NULL
    )
"""
)
conn.commit()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_PATH,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

def totalreg():
    """Return the total number of registered users."""
    try:
        return len(os.listdir("static/faces"))
    except Exception as e:
        logging.error(f"Error counting registered users: {e}")
        return 0

def extract_faces(img):
    """Extract faces from an image."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except Exception as e:
        logging.error(f"Error extracting faces: {e}")
        return []

def identify_face(facearray):
    """Identify a face using the trained model."""
    try:
        model = joblib.load(MODEL_PATH)
        return model.predict(facearray)
    except Exception as e:
        logging.error(f"Error identifying face: {e}")
        return None

def train_model():
    """Train the face recognition model."""
    try:
        faces = []
        labels = []
        userlist = os.listdir("static/faces")
        for user in userlist:
            for imgname in os.listdir(f"static/faces/{user}"):
                img = cv2.imread(f"static/faces/{user}/{imgname}")
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face.ravel())
                labels.append(user)
        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, labels)

        X_train, X_test, y_train, y_test = train_test_split(
            faces, labels, test_size=0.2, random_state=42
        )

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "matthews_cc": matthews_corrcoef(y_test, y_pred),
        }

        with open(METRICS_PATH, "wb") as f:
            joblib.dump(metrics, f)

        joblib.dump(knn, MODEL_PATH)
    except Exception as e:
        logging.error(f"Error training model: {e}")

def extract_attendance():
    """Extract attendance data from the database."""
    try:
        c.execute(
            "SELECT prenom, emp_id, arrivee, depart FROM attendance WHERE date=?",
            (DATETODAY,),
        )
        rows = c.fetchall()
        names = [row[0] for row in rows]
        rolls = [row[1] for row in rows]
        arrivees = [row[2] for row in rows]
        departs = [row[3] for row in rows]
        l = len(rows)
        return names, rolls, arrivees, departs, l
    except Exception as e:
        logging.error(f"Error extracting attendance: {e}")
        return [], [], [], [], 0

def export_to_csv():
    """Export attendance data to a CSV file."""
    try:
        names, rolls, arrivees, departs, _ = extract_attendance()

        csv_output = StringIO()
        csv_writer = csv.writer(csv_output)
        csv_writer.writerow(["Prénom", "N° Emp", "Temps d'arrivée", "Temps de Départ"])
        for name, roll, arrive, depart in zip(names, rolls, arrivees, departs):
            csv_writer.writerow([name, roll, arrive, depart])

        return Response(
            csv_output.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment;filename=attendance_{DATETODAY}.csv"
            },
        )
    except Exception as e:
        logging.error(f"Error exporting to CSV: {e}")
        return "Error exporting to CSV", 500

def add_attendance(name):
    """Add attendance record for a user."""
    try:
        username = name.split("_")[0]
        userid = name.split("_")[1]
        current_time = datetime.now().strftime("%H:%M:%S")

        c.execute(
            "SELECT arrivee, depart FROM attendance WHERE date=? AND emp_id=?",
            (DATETODAY, userid),
        )
        row = c.fetchone()
        if row is None:
            c.execute(
                "INSERT INTO attendance (prenom, emp_id, arrivee, date) VALUES (?, ?, ?, ?)",
                (username, userid, current_time, DATETODAY),
            )
        else:
            c.execute(
                "UPDATE attendance SET depart=? WHERE date=? AND emp_id=?",
                (current_time, DATETODAY, userid),
            )
        conn.commit()
    except Exception as e:
        logging.error(f"Error adding attendance: {e}")

def getallusers():
    """Get all registered users."""
    try:
        userlist = os.listdir("static/faces")
        names = []
        rolls = []
        l = len(userlist)

        for i in userlist:
            name, roll = i.split("_")
            names.append(name)
            rolls.append(roll)

        return userlist, names, rolls, l
    except Exception as e:
        logging.error(f"Error getting all users: {e}")
        return [], [], [], 0

@app.route("/")
def home():
    """Render the home page with attendance data."""
    names, rolls, arrivees, departs, l = extract_attendance()
    return render_template(
        "home.html",
        names=names,
        rolls=rolls,
        arrivees=arrivees,
        departs=departs,
        l=l,
        totalreg=totalreg(),
        datetoday2=DATETODAY2,
    )

@app.route("/start", methods=["GET"])
def start():
    """Start the face recognition process."""
    names, rolls, arrivees, departs, l = extract_attendance()

    if "face_recognition_model.pkl" not in os.listdir("models/"):
        return render_template(
            "home.html",
            names=names,
            rolls=rolls,
            arrivees=arrivees,
            departs=departs,
            l=l,
            totalreg=totalreg(),
            datetoday2=DATETODAY2,
            mess="Il n'y a pas de modèle entraîné dans le dossier statique. Veuillez ajouter un nouveau visage pour continuer.",
        )

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x + w, y - 40), (86, 32, 251), -1)
            face = cv2.resize(frame[y : y + h, x : x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(
                frame,
                f"{identified_person}",
                (x, y - 15),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                1,
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) in [ord("q"), 27]:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, arrivees, departs, l = extract_attendance()
    return render_template(
        "home.html",
        names=names,
        rolls=rolls,
        arrivees=arrivees,
        departs=departs,
        l=l,
        totalreg=totalreg(),
        datetoday2=DATETODAY2,
    )

@app.route("/metrics")
def metrics():
    """Render the metrics page with model performance metrics."""
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "rb") as f:
                metrics = joblib.load(f)
                metrics = {k: round(v, 2) if isinstance(v, (int, float)) else v
                         for k, v in metrics.items()}
        else:
            metrics = {
                "accuracy": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "f1_score": "N/A",
            }
        keys = list(metrics.keys())
        values = list(metrics.values())

        if all(isinstance(val, (int, float)) for val in values):
            plt.figure(figsize=(10, 5))
            plt.bar(keys, values, color=["blue", "orange", "green", "red"])
            plt.xlabel("Metrics")
            plt.ylabel("Scores")
            plt.title("Model Performance Metrics")

            img = io.BytesIO()
            plt.savefig(img, format="png")
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode("utf8")

            return render_template("metrics.html", metrics=metrics, plot_url=plot_url)
        else:
            logging.error("Non-Numeric Values Detected in Metrics :", values)
            return "Metrics not available", 500
    except Exception as e:
        logging.error(f"Error rendering metrics: {e}")
        return "Metrics not available", 500

@app.route("/add", methods=["GET", "POST"])
def add():
    """Add a new user and train the model."""
    try:
        newusername = request.form["newusername"]
        newuserid = request.form["newuserid"]
        userimagefolder = "static/faces/" + newusername + "_" + str(newuserid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)

        cap = cv2.VideoCapture(0)
        captured_images = 0

        while captured_images < NIMGS:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture image from the camera.")
                break

            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(
                    frame,
                    f"Images Captured: {captured_images}/{NIMGS}",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 20),
                    2,
                    cv2.LINE_AA,
                )
                if captured_images < NIMGS:
                    name = newusername + "_" + str(captured_images) + ".jpg"
                    cv2.imwrite(os.path.join(userimagefolder, name), frame[y : y + h, x : x + w])
                    captured_images += 1

            cv2.imshow("Adding new User", frame)
            if cv2.waitKey(1) in [ord("q"), 27]:
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured_images == NIMGS:
            train_model()
        else:
            logging.warning(f"Captured {captured_images} images, expected {NIMGS}.")

        names, rolls, arrivees, departs, l = extract_attendance()
        return render_template(
            "home.html",
            names=names,
            rolls=rolls,
            arrivees=arrivees,
            departs=departs,
            l=l,
            totalreg=totalreg(),
            datetoday2=DATETODAY2,
        )
    except Exception as e:
        logging.error(f"Error adding new user: {e}")
        return "Error adding new user", 500

@app.route("/export/csv")
def export_csv():
    """Export attendance data to a CSV file."""
    return export_to_csv()

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
