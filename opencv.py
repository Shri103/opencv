import mediapipe as mp
import cv2

vid = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp.solutions.hands.Hands()
fingerCoords = [(8, 6), (12, 10), (16, 14)]


while True:

    ret, frame = vid.read()
    frame2 = frame.copy()
    with mp_pose.Pose(static_image_mode=True) as pose:
        result = pose.process(frame2)
    num = 0
    result2 = hands.process(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    if result2.multi_hand_landmarks:
        handPoints = []
        for hand_landmarks in result2.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame2, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(frame, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)
            for idx, lm in enumerate(hand_landmarks.landmark):
                h, w, c, = frame2.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handPoints.append((cx, cy))

            for c in fingerCoords:
                if handPoints[c[0]][1] < handPoints[c[1]][1]:
                    num += 1
            if result.pose_landmarks:
                arm_landmarks = result.pose_landmarks.landmark
                if arm_landmarks[16].y < arm_landmarks[12].y:
                    cv2.putText(frame, "Hand Raised", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0),
                                12)
                    if num == 3:
                        cv2.putText(frame2, "Volunteered as Tribute", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 12)


    red_dot = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=1)
    mp_draw.draw_landmarks(frame2, landmark_list=result.pose_landmarks, landmark_drawing_spec=red_dot)
    mp_draw.draw_landmarks(frame2, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)


    cv2.imshow("hand_raise_Only", frame)
    cv2.imshow("landmarks", frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

