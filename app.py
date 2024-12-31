import cv2
import os
from flask import Flask, request, render_template, Response
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sqlite3
import joblib
from io import StringIO
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv
import io
import base64
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

nimgs = 50

datetoday = date.today().strftime("%d_%m_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if not os.path.isdir("Attendance"):
    os.makedirs("Attendance")
if not os.path.isdir("static"):
    os.makedirs("static")
if not os.path.isdir("static/faces"):
    os.makedirs("static/faces")

db_path = f"Attendance/attendance_{datetoday}.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
c = conn.cursor()

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


def totalreg():
    return len(os.listdir("static/faces"))


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


def identify_face(facearray):
    model = joblib.load("static/face_recognition_model.pkl")
    return model.predict(facearray)


def train_model():
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

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    matthews_cc = matthews_corrcoef(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "matthews_cc": matthews_cc,
    }

    with open("static/metrics.pkl", "wb") as f:
        joblib.dump(metrics, f)

    joblib.dump(knn, "static/face_recognition_model.pkl")


def extract_attendance():
    c.execute(
        "SELECT prenom, emp_id, arrivee, depart FROM attendance WHERE date=?",
        (datetoday,),
    )
    rows = c.fetchall()
    names = [row[0] for row in rows]
    rolls = [row[1] for row in rows]
    arrivees = [row[2] for row in rows]
    departs = [row[3] for row in rows]
    l = len(rows)
    return names, rolls, arrivees, departs, l


def export_to_csv():
    names, rolls, arrivees, departs, _ = extract_attendance()

    csv_output = StringIO()
    csv_writer = csv.writer(csv_output)
    csv_writer.writerow(["Prènom", "N° Emp", "Temps d'arrivée", "Temps de Départ"])
    for name, roll, arrive, depart in zip(names, rolls, arrivees, departs):
        csv_writer.writerow([name, roll, arrive, depart])

    return Response(
        csv_output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment;filename=attendance_{datetoday}.csv"
        },
    )


def add_attendance(name):
    username = name.split("_")[0]
    userid = name.split("_")[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    c.execute(
        "SELECT arrivee, depart FROM attendance WHERE date=? AND emp_id=?",
        (datetoday, userid),
    )
    row = c.fetchone()
    if row is None:
        c.execute(
            "INSERT INTO attendance (prenom, emp_id, arrivee, date) VALUES (?, ?, ?, ?)",
            (username, userid, current_time, datetoday),
        )
    else:
        c.execute(
            "UPDATE attendance SET depart=? WHERE date=? AND emp_id=?",
            (current_time, datetoday, userid),
        )
    conn.commit()


def getallusers():
    userlist = os.listdir("static/faces")
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split("_")
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


@app.route("/")
def home():
    names, rolls, arrivees, departs, l = extract_attendance()
    return render_template(
        "home.html",
        names=names,
        rolls=rolls,
        arrivees=arrivees,
        departs=departs,
        l=l,
        totalreg=totalreg(),
        datetoday2=datetoday2,
    )


@app.route("/start", methods=["GET"])
def start():
    names, rolls, arrivees, departs, l = extract_attendance()

    if "face_recognition_model.pkl" not in os.listdir("static"):
        return render_template(
            "home.html",
            names=names,
            rolls=rolls,
            arrivees=arrivees,
            departs=departs,
            l=l,
            totalreg=totalreg(),
            datetoday2=datetoday2,
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
        datetoday2=datetoday2,
    )


@app.route("/metrics")
def metrics():
    if os.path.exists("static/metrics.pkl"):
        with open("static/metrics.pkl", "rb") as f:
            metrics = joblib.load(f)
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
        print("Non-Numeric Values Detected in Metrics :", values)
        return "Metrics not available", 500


@app.route("/add", methods=["GET", "POST"])
def add():
    newusername = request.form["newusername"]
    newuserid = request.form["newuserid"]
    userimagefolder = "static/faces/" + newusername + "_" + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(
                frame,
                f"Images Captured: {i}/{nimgs}",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 20),
                2,
                cv2.LINE_AA,
            )
            if j % 5 == 0:
                name = newusername + "_" + str(i) + ".jpg"
                cv2.imwrite(userimagefolder + "/" + name, frame[y : y + h, x : x + w])
                i += 1
            j += 1
        if j == nimgs * 5:
            break
        cv2.imshow("Adding new User", frame)
        if cv2.waitKey(1) in [ord("q"), 27]:
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()
    names, rolls, arrivees, departs, l = extract_attendance()
    return render_template(
        "home.html",
        names=names,
        rolls=rolls,
        arrivees=arrivees,
        departs=departs,
        l=l,
        totalreg=totalreg(),
        datetoday2=datetoday2,
    )


@app.route("/export/csv")
def export_csv():
    return export_to_csv()


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
