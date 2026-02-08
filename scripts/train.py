import os
from ultralytics import YOLO 
import torch 

def train_model(data_path: str, epochs: int = 100, batch: int = 16, model_name: str = 'yolo11n.pt'):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    model = YOLO(model_name, task='detect')
    
    results = model.train(
        data=data_path, 
        epochs=epochs, 
        batch=batch, 
        device=device, 
        project='../runs', 
        name='gun_detection'
    )
    
    return results

if __name__ == '__main__':
    # Configuration
    data_path = '../data/data.yaml'
    epochs = 100
    batch = 16
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Error: data.yaml not found at {data_path}")
        print("Run preprocessing.py first")
        exit(1)
    
    # Train model
    print("Training started")
    results = train_model(data_path, epochs=epochs, batch=batch)
    print("Training complete!")
    print(f"Results saved to: {results.save_dir}")

