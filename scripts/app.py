import gradio as gr 
from ultralytics import YOLO 
from PIL import Image

model = YOLO('runs/detect/train/weights/best.pt')

def detect_gun(image):
    results = model.predict(source = image, conf = 0.5)
    annotated = results[0].plot()

    annotated = annotated[:, :, :: -1]
    return Image.fromarray(annotated)

demo = gr.Interface(fn = detect_gun, inputs = gr.Image(type = "pil"), 
                    outputs = gr.Image(type = "pil"), title = "Mini Gun detection app", description = 'Upload an image containing ' \
                    'a gun')

if __name__ == '__main__':
    demo.launch()