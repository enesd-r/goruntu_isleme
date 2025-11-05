import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_yellow = np.array([15,80,80])
    upper_yellow = np.array([35,255,255])
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((7,7), np.uint8)
    mask = cv.GaussianBlur(mask, (7,7), 0)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        biggest = max(contours, key=lambda c: cv.contourArea(c))
        area = cv.contourArea(biggest)

        if 500< area < 100000:
            x,y,w,h = cv.boundingRect(biggest)

            cx, cy = (x+w//2), (y+h//2)
            
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv.circle(frame, (cx, cy), 5, (0,0,255), -1)

            frame_center = frame.shape[1] // 2

            if cx < frame_center:
                print("Sağa yönel")
            else:
                print("Sola yönel")
    
    cv.imshow("Sarı Nesne Takıbı", frame)

    if cv.waitKey(1) & 0XFF == 27:
        break

cap.release()
cv.destroyAllWindows()
