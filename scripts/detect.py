from ultralytics import YOLO
import cv2
import sys

def detect_gun(image_path, model_path = 'runs/detect/train/weights/best.pt'):
    model = YOLO(model_path)
    results = model.predict(source = image_path, conf = 0.5)
    annotated = results[0].plot()

    cv2.imwrite('result.jpg', annotated)

    return results
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("WRong format, usage: python detect.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    detect_gun(image_path)

