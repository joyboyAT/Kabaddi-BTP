import cv2

cap = cv2.VideoCapture("test_raid.mp4")
print(cap.get(cv2.CAP_PROP_FPS))
cap.release()
