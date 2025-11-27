import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, _ = frame.shape 
    
    frame = cv2.flip(frame, 1) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        if i < 2 and area > 1000: 
            x, y, w, h = cv2.boundingRect(cnt)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)  
            
            cx = x + w // 2
            cy = y + h // 2
            centers.append((cx, cy))

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  

    if len(centers) >= 2:
        cv2.line(frame, centers[0], centers[1], (0, 255, 0), 3)  

    cv2.imshow("Mavi Nesne Tespiti", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
