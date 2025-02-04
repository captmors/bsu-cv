import cv2
import os
import mediapipe as mp


VIDEO_PATH = os.path.join(os.getcwd(), "assets/video.mp4")
print(VIDEO_PATH)
if not os.path.exists(VIDEO_PATH):
    print("Видео не найдено по указанному пути!")
    
RECT_COLOR = (255, 0, 0) 
RECT_THICKNESS = 3  
RECT_W, RECT_H = 150, 100  


def move_rectangle(rect_x, rect_y, hand_x, hand_y, frame_width, frame_height):
    rect_x = min(max(0, hand_x - RECT_W // 2), frame_width - RECT_W)
    rect_y = min(max(0, hand_y - RECT_H // 2), frame_height - RECT_H)
    return rect_x, rect_y


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Не удалось открыть видео!")
else:
    print("Видео открылось успешно!")
    

with mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Не удалось получить кадр или видео завершено.")
            break

        # Перевод в RGB для MediaPipe
        frame = cv2.flip(frame, 1)  # Отразить по горизонтали
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Обработка кадра
        results = hands.process(frame_rgb)

        # Обработка всех рук
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Координаты ладони
                hand_x = int(hand_landmarks.landmark[9].x * frame.shape[1])
                hand_y = int(hand_landmarks.landmark[9].y * frame.shape[0])

                # Координаты прямоугольника
                rect_x, rect_y = move_rectangle(0, 0, hand_x, hand_y, frame.shape[1], frame.shape[0])

                # Рисование прямоугольника для каждой руки
                cv2.rectangle(frame, (rect_x, rect_y), (rect_x + RECT_W, rect_y + RECT_H), RECT_COLOR, RECT_THICKNESS)

        # Показ видео
        cv2.imshow("Augmented Reality Game", frame)

        # Прерывание по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
