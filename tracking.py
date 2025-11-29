import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("video.mp4")

LINE_START = (0, 250)
LINE_END = (640, 250)
LINE_Y = 250 

count_in = 0
count_out = 0

track_history = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))

    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None:
        
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            
            center_y = int((y1 + y2) / 2)
            center_x = int((x1 + x2) / 2) 

            if track_id not in track_history:
                track_history[track_id] = []
            
            track_history[track_id].append(center_y)

            if len(track_history[track_id]) > 2:
                track_history[track_id].pop(0)

            if len(track_history[track_id]) == 2:
                prev_y = track_history[track_id][0] # Önceki konum
                curr_y = track_history[track_id][1] # Şimdiki konum

                if prev_y < LINE_Y and curr_y >= LINE_Y:
                    count_in += 1
                    # Çizgiyi anlık olarak Yeşil yak
                    cv2.line(frame, LINE_START, LINE_END, (0, 255, 0), 4)

                if prev_y > LINE_Y and curr_y <= LINE_Y:
                    count_out += 1
                    # Çizgiyi anlık olarak Kırmızı yak
                    cv2.line(frame, LINE_START, LINE_END, (0, 0, 255), 4)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    # Referans çizgisini çiz
    cv2.line(frame, LINE_START, LINE_END, (0, 255, 255), 2)

    # Sayaçları ekrana yaz
    cv2.putText(frame, f"Giren: {count_in}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Cikan: {count_out}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Kisi Sayar", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
