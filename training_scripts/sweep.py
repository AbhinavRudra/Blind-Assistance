import wandb
from ultralytics import YOLO

def train_optimum_sweep():
    wandb.init()
    
    model = YOLO('yolo11n.pt')
    results = model.train(
        data='dataset/d2/data.yaml',
        epochs=30,  # Match your baseline epoch count for fair comparison
        imgsz=640,
        device=0,
        optimizer='AdamW',
        
        # SWEEP PARAMETERS (vary)
        cls=wandb.config.cls,
        lr0=wandb.config.lr0,
        mixup=wandb.config.mixup,
        hsv_s=wandb.config.hsv_s,
        
        # FIXED PARAMETERS (baseline)
        box=7.5,
        dfl=1.5,
        lrf=0.01,
        warmup_epochs=3,
        degrees=10,
        scale=0.5,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_v=0.4,
        weight_decay=0.001,
        cos_lr=True,
        patience=30,
        workers=16,
        amp=True,
        verbose=True
    )
    
    # Log metrics
    best_mAP50 = results.results_dict.get('metrics/mAP50(B)', 0)
    best_precision = results.results_dict.get('metrics/precision(B)', 0)
    best_recall = results.results_dict.get('metrics/recall(B)', 0)
    
    wandb.log({
        "final_mAP50": best_mAP50,
        "final_precision": best_precision,
        "final_recall": best_recall
    })
    wandb.finish()

if __name__ == "__main__":
    train_optimum_sweep()