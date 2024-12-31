Ce code met en place une application web Flask pour un système de reconnaissance faciale pour la gestion de la présence. Voici une explication détaillée de chaque partie du code:

### Importations et Initialisation

```python
import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sqlite3
import joblib
```
- **cv2**: Utilisé pour les opérations de traitement d'image avec OpenCV.
- **os**: Utilisé pour les interactions avec le système de fichiers.
- **flask**: Utilisé pour créer l'application web.
- **datetime**: Pour manipuler et formater les dates et heures.
- **numpy**: Utilisé pour les opérations mathématiques et les tableaux multidimensionnels.
- **KNeighborsClassifier**: Algorithme de classification K-NN de scikit-learn.
- **sqlite3**: Pour gérer la base de données SQLite.
- **joblib**: Pour sauvegarder et charger des modèles de machine learning.

```python
app = Flask(__name__)

nimgs = 10

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
```
- **app = Flask(__name__)**: Initialise l'application Flask.
- **nimgs**: Nombre d'images à capturer pour chaque utilisateur.
- **datetoday** et **datetoday2**: Formats de la date actuelle pour l'utilisation dans les noms de fichiers et l'affichage.

```python
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
```
- **face_detector**: Charge le classificateur en cascade pour la détection des visages.

```python
if not os.path.isdir("Attendance"):
    os.makedirs("Attendance")
if not os.path.isdir("static"):
    os.makedirs("static")
if not os.path.isdir("static/faces"):
    os.makedirs("static/faces")
```
- Vérifie et crée les répertoires nécessaires pour stocker les données de présence et les images des visages si ils n'existent pas.

### Initialisation de la Base de Données

```python
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
```
- **db_path**: Définit le chemin de la base de données SQLite pour la date actuelle.
- **conn** et **c**: Connecte et crée un curseur pour exécuter des commandes SQL.
- **CREATE TABLE IF NOT EXISTS**: Crée une table "attendance" si elle n'existe pas déjà.

### Fonctions Utilitaires

#### `totalreg()`

```python
def totalreg():
    return len(os.listdir("static/faces"))
```
- Retourne le nombre total d'utilisateurs enregistrés en comptant le nombre de dossiers dans "static/faces".

#### `extract_faces(img)`

```python
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []
```
- Convertit une image en niveaux de gris et utilise le détecteur de visage pour trouver les visages.

#### `identify_face(facearray)`

```python
def identify_face(facearray):
    model = joblib.load("static/face_recognition_model.pkl")
    return model.predict(facearray)
```
- Charge le modèle de reconnaissance faciale et prédit l'identité de l'image fournie.

#### `train_model()`

```python
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
    joblib.dump(knn, "static/face_recognition_model.pkl")
```
- Entraîne un modèle K-NN sur les images de visages et les sauvegarde dans un fichier.

#### `extract_attendance()`

```python
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
    print(f"Loaded {l} attendance records.")
    return names, rolls, arrivees, departs, l
```
- Récupère les enregistrements de présence pour la date actuelle de la base de données.

#### `add_attendance(name)`

```python
def add_attendance(name):
    username = name.split("_")[0]
    userid = name.split("_")[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"Recording attendance for {username} with id {userid} at {current_time}.")

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
```
- Ajoute ou met à jour l'enregistrement de présence pour un utilisateur.

#### `getallusers()`

```python
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
```
- Récupère la liste de tous les utilisateurs enregistrés.

### Routes Flask

#### `/`

```python
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
```
- Affiche la page d'accueil avec les informations de présence.

#### `/start`

```python
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
                (255,

 255, 255),
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
```
- Démarre la reconnaissance faciale en capturant des images de la caméra et en enregistrant les présences.

#### `/add`

```python
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
    print("Training Model")
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
```
- Ajoute un nouvel utilisateur en capturant des images de son visage et en réentraînant le modèle de reconnaissance faciale.

### Exécution de l'application

```python
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
```
- Démarre le serveur Flask pour l'application, accessible depuis n'importe quelle adresse IP du réseau local.