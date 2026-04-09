import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")

results = model.train(
    data="dogs.yaml", epochs=1, batch=8,
    project='dogs', val=True, verbose=True, workers=0
)

trained_model = YOLO('dogs/train/weights/best.pt')

input_dir = "images/test"
output_dir = "result"
os.makedirs(output_dir, exist_ok=True)

test_images = [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")]

for img_name in test_images:
    img_path = os.path.join(input_dir, img_name)
    
    results = trained_model(img_path, conf=0.5, iou=0.45)
    
    result_img = results[0].plot()
    
    save_path = os.path.join(output_dir, img_name)
    cv2.imwrite(save_path, result_img)

video_path = "dogs.webm"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        break
    
    results = trained_model(frame)
    
    cv2.imshow("Dog Detection - Trained Model", results[0].plot())
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()