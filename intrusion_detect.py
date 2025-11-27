import cv2 as cv
import numpy as np
from ultralytics import YOLO    

model = YOLO("yolov8n.pt")

cap = cv.VideoCapture("video.mp4")

region_points = np.array([[100,100], [500,100], [500, 400], [100,400]], np.uint32)
region_points = region_points.reshape(-1, 1, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (640,480))

    results = model(frame, stream=True, verbose=False)

    intrusion_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:
                x1,y1,x2,y2 = map(int, box.xyxy[0])

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                result = cv.pointPolygonTest(region_points, (center_x, center_y), False)

                if result >= 0:
                    intrusion_detected = True

                    cv.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv.putText(frame, "IHLAL", (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

                else:
                    cv.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                
                cv.circle(frame, (center_x, center_y), 5, (255,0,0), -1)
        
    color = (0,0,255) if intrusion_detected else (0,255,0)
    cv.polylines(frame, [region_points], isClosed=True, color=color, thickness=2)

    if intrusion_detected:
        cv.putText(frame, "YASAKLI BOLGE", (20,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv.putText(frame, "GUVENLI BOLGE", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("Guvenlik Kamerasi", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
