# Face Recognition Attendance System

This is a simple web-based face recognition attendance system implemented using Flask and OpenCV.

## Overview

The system captures real-time video feed from the webcam, detects faces using Haar cascade classifier, and recognizes them based on a pre-trained KNN classifier. Users can add new faces to the system, and the attendance of recognized individuals is logged in a SQL Database.

## Prerequisites

- Python 3.x
- OpenCV
- Flask
- NumPy
- scikit-learn
- reportlab

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/scorpionTaj/face-recognition.git
    ```

2. Install dependencies:

    ```bash
    pip install -r config/requirements.txt
    ```

## Usage

1. Run the Flask application:

    ```bash
    python app.py
    ```

2. Open a web browser and go to `http://localhost:5000/`.

3. Click on the "Prendre la Présence" button to begin capturing attendance.

4. To add a new user, click on the "Ajouter un Nouvel Utilisateur" button and follow the instructions.

## Directory Structure
```
Main Project Folder

├── attendance/                               # Directory for attendance-related functionality
│   └── attendance_{datetoday}.db             # Attendance-related files (SQLite Database)
│
├── classifiers/                             # Directory for Haar Cascade classifier files
│   └── haarcascade_frontalface_default.xml  # Haar Cascade file for face detection
│
├── config/                                  # Directory for configuration files
│   └── requirements.txt                     # File listing the project's dependencies
│
├── docker/                                  # Directory for Docker-related files
│   ├── dockerfile                           # Dockerfile for containerizing the application
│   └── dockerfile.txt                       # Additional Docker-related file (if any)
|
├── docs/                                    # Directory for documentation files
│   └── Details.md                           # Additional documentation file
│
├── logs/                                    # Directory for log files
│   └── app.log                              # Application log file
│
├── models/                                  # Directory for serialized model files
│   └── your_model.pkl                       # Serialized Python object file (e.g., machine learning model)
│
├── static/                                  # Directory for static assets
│   └── ...                                  # CSS, JavaScript, images, and other static files
│
├── templates/                               # Directory for HTML templates
│   ├── home.html                            # Home page HTML template
│   └── metrics.html                         # Metrics page HTML template
│
|
├── .gitignore                               # Git ignore file to specify which files and directories to ignore
├── app.py                                   # Main application file
├── LICENSE                                  # License file for the project
├── README.md                                # Readme file for the project overview and instructions
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you encounter any bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
