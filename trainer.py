import os
import torch
from tqdm import tqdm

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """单轮训练函数"""
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Training")
    for batch_idx, (data, labels) in enumerate(pbar):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)  # VMambaT 前向传播（需确保模型输出与任务匹配）
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        _, preds = torch.max(outputs.data, 1)
        total_correct += torch.sum(preds == labels.data).item()
        total_samples += labels.size(0)
        batch_loss = total_loss / total_samples
        batch_acc = total_correct / total_samples
        pbar.set_postfix({"Loss": batch_loss, "Acc": batch_acc})
    
    avg_train_loss = total_loss / total_samples
    avg_train_acc = total_correct / total_samples
    return avg_train_loss, avg_train_acc

def validate(model, val_loader, criterion, device):
    """验证函数（无梯度计算）"""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    pbar = tqdm(val_loader, desc=f"Validation")
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels.data).item()
            total_samples += data.size(0)
            batch_loss = total_loss / total_samples
            batch_acc = total_correct / total_samples
            pbar.set_postfix({"Loss": batch_loss, "Acc": batch_acc})
    avg_val_loss = total_loss / total_samples
    avg_val_acc = total_correct / total_samples
    
    return avg_val_loss, avg_val_acc

def save_checkpoint(model, optimizer, epoch, val_acc, save_optimizer=True, save_name="best_model.pth", save_dir="./checkpoints", args=None):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_accuracy": val_acc,
    }
    if args is not None:
        checkpoint["args"] = args
    if save_optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    latest_path = os.path.join(save_dir, save_name)
    torch.save(checkpoint, latest_path)

def load_checkpoint(model, optimizer=None, resume=None, device="cpu"):
    """加载断点续训"""
    if not os.path.exists(resume):
        raise FileNotFoundError(f"Checkpoint file not found: {resume}")
    
    checkpoint = torch.load(resume, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint.get("val_accuracy", 0.0)
    
    print(f"Resumed training from checkpoint: {resume}")
    print(f"Start epoch: {start_epoch}, Best val acc so far: {best_val_acc:.4f}")
    
    return start_epoch, best_val_acc