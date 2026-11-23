import cv2
from ultralytics import YOLO

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    res = model.predict(frame, conf=0.657, agnostic_nms=True, verbose=False)[0]
    cv2.imshow('Coin Detector (press Q to quit)', res.plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()