import wandb
from ultralytics import YOLO

def train_baseline():
    wandb.init(
        project="yolo-blind-detection",
        name="baseline-epoch32",
        notes="Starting point for hyperparameter sweep"
    )
    
    model = YOLO('yolov11n.pt')
    results = model.train(
        data='data/dataset.yaml',
        epochs=50,
        imgsz=640,
        device=0,
        optimizer='AdamW',
        lr0=0.005,
        lrf=0.01,
        warmup_epochs=3,
        box=7.5,
        cls=0.7,
        dfl=1.5,
        mosaic=1.0,
        mixup=0.1,
        degrees=10,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        weight_decay=0.001,
        cos_lr=True,
        patience=30,
        cache='disk',
        workers=16,
        amp=True
    )
    
    wandb.log({"best_mAP50": results.results_dict.get('metrics/mAP50(B)', 0)})
    wandb.finish()

if __name__ == "__main__":
    train_baseline()
