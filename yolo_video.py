from ultralytics import YOLO
import cv2
import os

def videoCheck(path, save_path='output.mp4'):
    model = YOLO('best.pt')
    video = cv2.VideoCapture(path)

    if not video.isOpened():
        print("Не удалось открыть видео.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    window_name = 'Object Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model(frame, imgsz=640, conf=0.8, iou=0.4, verbose=False)
        annotated_frame = results[0].plot()

        out.write(annotated_frame)
        frame_count += 1

        cv2.imshow(window_name, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Видео сохранено: {os.path.abspath(save_path)}")

if __name__ == '__main__':
    videoCheck('mouse0.mp4', save_path='output.mp4')