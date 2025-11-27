from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if conf > 0.5:
                area = (x2 - x1) * (y2 - y1)
                detections.append((x1, y1, x2, y2, area))

    detections.sort(key=lambda x: x[4], reverse=True) # 4 -> area

    if len(detections) >= 2:
        d1, d2 = detections[0], detections[1]

        c1 = ((d1[0] + d1[2]) // 2, (d1[1] + d1[3]) // 2)
        c2 = ((d2[0] + d2[2]) // 2, (d2[1] + d2[3]) // 2)

        cv2.rectangle(frame, (d1[0], d1[1]), (d1[2], d1[3]), (0, 140, 255), 2)
        cv2.rectangle(frame, (d2[0], d2[1]), (d2[2], d2[3]), (0, 140, 255), 2)

        cv2.line(frame, c1, c2, (0, 255, 0), 2)

        mid_x = (c1[0] + c2[0]) // 2
        frame_center = frame.shape[1] // 2

        if mid_x < frame_center - 30:
            direction = "Sola Yönel"
        elif mid_x > frame_center + 30:
            direction = "Sağa Yönel"
        else:
            direction = "Düz İlerle"

        cv2.putText(frame, direction, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        cv2.line(frame, (frame_center, 0), (frame_center, frame.shape[0]), (255, 255, 255), 1)

    cv2.imshow("YOLO Buoy Navigation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
