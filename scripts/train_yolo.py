from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Set multiprocessing method untuk Windows
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='D:\Quant_ML_Project\ML.py\EasyPark\configs\data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,  # Kurangi batch size juga
        name='parking_detection',
        patience=50,
        save=True,
        device=0,
        plots=True,
        workers=0  # Disable multiprocessing
    )
    
    print(results)