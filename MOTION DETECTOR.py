import mediapipe as mp
import cv2
import time

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
pTime = 0
cTime = 0

Green = mp_drawing.DrawingSpec(color = (0,255,0), thickness= 2, circle_radius= 4)
Blue = mp_drawing.DrawingSpec(color = (255,0,0), thickness= 2, circle_radius= 4)
Red = mp_drawing.DrawingSpec(color = (0,0,255), thickness= 1, circle_radius= 2)

cap = cv2.VideoCapture(0)
#address = "http://192.168.1.2:8080/video"
#cap.open(address)

#Initiate Holistiic Model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

 while cap.isOpened():
    ret, frame = cap.read()

    #Recolor Feed
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #Make Detection
    results = holistic.process(image)
    #print(results.face_landmarksq)

    #Recolor image back to BGR for rendering
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 1.Face Landmarks
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, Red, Green)
    # 2.Right Hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, Green, Blue)
    # 3.Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, Green, Blue)
    # 4.Pose Detection
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, Green, Blue )


    cTime = time.time()
    fps = 1/( cTime- pTime)
    pTime = cTime
    cv2.putText(image, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),3)
    cv2.imshow("Holistic Model Detections", image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()