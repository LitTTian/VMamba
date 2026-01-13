import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import wandb
import os
import pandas as pd
from datetime import datetime
# 自定义模块导入
import logging
from models import VMambaT
from models.data import build_cifar10_dataloaders
from trainer import train_one_epoch, validate, save_checkpoint, load_checkpoint

RUN_ID = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def setup_logger(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, f'train_{RUN_ID}.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="VMamba Model Training with WandB")
    # 基础训练参数
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation set split ratio")
    # 设备与路径参数
    parser.add_argument("--save_interval", type=int, default=10, help="Interval (in epochs) to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Training device (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint path")
    # WandB 配置参数
    parser.add_argument("--wandb_project", type=str, default="VMamba-Training", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity (username/team)")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name")
    # 模型与数据配置（可根据你的任务调整）
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classification classes")
    parser.add_argument("--dataset_path", type=str, default="./cifar", help="Root directory of the dataset")
    parser.add_argument("--resize_size", type=int, nargs=2, default=(64, 64), help="Input image resize size")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger()
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or f"VMamba-Train-{RUN_ID}",
        config=vars(args)
    )
    logger.info(f"WandB run initialized: {wandb.run.name}")
    device = torch.device(args.device)
    logger.info(f"Using training device: {device}")
    logger.info("Preparing dataset...")
    
    train_loader, val_loader = build_cifar10_dataloaders(
        dataset_path=args.dataset_path,
        dataset_name="CIFAR10",
        batch_size=args.batch_size,
        resize_size=tuple(args.resize_size),
        device=args.device
    )
    
    logger.info(f"Dataset prepared: Train size={len(train_loader.dataset)}, Val size={len(val_loader.dataset)}")
    logger.info("Initializing model, criterion and optimizer...")
    model = VMambaT(
        out_features=args.num_classes,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    start_epoch = 0
    if args.resume:
        start_epoch, best_val_acc = load_checkpoint(model, optimizer, 
                                                    resume=args.resume,
                                                    device=device)
    logger.info("Starting training...")
    best_val_acc = 0.0
    worst_val_acc = 1.0
    training_metrics = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
        "epoch_duration": [],
        "epoch": [] 
    }
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = datetime.now()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        epoch_end_time = datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
        # NT: 记录每个epoch的指标
        current_lr = optimizer.param_groups[0]["lr"]
        training_metrics["epoch"].append(epoch + 1)
        training_metrics["train_loss"].append(train_loss)
        training_metrics["train_accuracy"].append(train_acc)
        training_metrics["val_loss"].append(val_loss)
        training_metrics["val_accuracy"].append(val_acc)
        training_metrics["learning_rate"].append(current_lr)
        training_metrics["epoch_duration"].append(epoch_duration)
        scheduler.step()
        log_msg = (
            f"Epoch [{epoch+1}/{args.epochs}] completed in {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )
        logger.info(log_msg)
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr,
            "epoch_duration": epoch_duration
        })

        if val_acc > best_val_acc:
            save_checkpoint(model, optimizer, epoch, val_acc, save_optimizer=False, save_dir=args.save_dir, args=args)
            best_val_acc = val_acc
            print(f"New best model saved with Val Acc: {best_val_acc:.4f}")
        if val_acc < worst_val_acc:
            worst_val_acc = val_acc
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, val_acc, save_optimizer=True, 
                            save_name=f"checkpoint_epoch_{epoch+1}.pth", save_dir=args.save_dir, args=args)
            metricsdf = pd.DataFrame(training_metrics)
            metricsdf.to_csv(os.path.join(args.save_dir, "training_metrics.csv"), index=False)
    metricsdf = pd.DataFrame(training_metrics)
    metricsdf.to_csv(os.path.join(args.save_dir, "training_metrics.csv"), index=False)
    avg_epoch_duration = sum(training_metrics["epoch_duration"]) / len(training_metrics["epoch_duration"])
    logger.info(f"Training completed!")
    logger.info(f"Best Val Acc: {best_val_acc:.4f}, Worst Val Acc: {worst_val_acc:.4f}")
    logger.info(f"Average Epoch Duration: {avg_epoch_duration:.2f}s")
    wandb.summary.update({
        "best_val_acc": best_val_acc,
        "worst_val_acc": worst_val_acc,
        "average_epoch_duration": avg_epoch_duration
    })
    wandb.finish()

if __name__ == "__main__":
    main()