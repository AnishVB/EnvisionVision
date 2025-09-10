import cv2
import pyttsx3
import time

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

face_cascade = cv2.CascadeClassifier(r"detectionxmp\haarcascade_frontalface_default.xml")
fullbody_cascade = cv2.CascadeClassifier(r"detectionxmp\haarcascade_fullbody.xml")

if face_cascade.empty() or fullbody_cascade.empty():
    print("Error loading cascade files!")
    exit()

cap = cv2.VideoCapture(0)

face_start_time = None
body_start_time = None
face_announced = False
body_announced = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    bodies = fullbody_cascade.detectMultiScale(gray, 1.1, 5)

    now = time.time()

    # ---- FACE ----
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, "GAY", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        if face_start_time is None:
            face_start_time = now
        elif not face_announced and (now - face_start_time >= 5):
            engine.say("Face detected")
            engine.runAndWait()
            face_announced = True
    else:
        face_start_time = None
        face_announced = False

    # ---- BODY ----
    if len(bodies) > 0:
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
            cv2.putText(frame, "Full Body", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        if body_start_time is None:
            body_start_time = now
        elif not body_announced and (now - body_start_time >= 5):
            engine.say("Full body detected")
            engine.runAndWait()
            body_announced = True
    else:
        body_start_time = None
        body_announced = False

    cv2.putText(frame, "Press 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 2)

    cv2.imshow("Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

         
    
cap.release()
cv2.destroyAllWindows()

