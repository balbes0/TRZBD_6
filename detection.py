from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Не можем открыть камеру")
else:
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Не удалось получить кадр")
            break

        detect_image = model(frame, imgsz=640, iou=0.4, conf=0.6, verbose=False)

        result = detect_image[0]

        annotated_image = result.plot()

        cv2.imshow('webcamera detection', annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

camera.release()
cv2.destroyAllWindows()