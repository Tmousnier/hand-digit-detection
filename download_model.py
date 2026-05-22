import urllib.request
import os

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if os.path.exists(model_path):
    print(f"Modele deja present : {model_path}")
else:
    print("Telechargement du modele MediaPipe Hand Landmarker...")
    urllib.request.urlretrieve(url, model_path)
    print(f"Modele telecharge avec succes ({os.path.getsize(model_path)} octets)")
    print(f"Chemin : {model_path}")

