import torchvision.transforms as transforms

# ImageNet 固定归一化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

def build_transforms(resize_size=(224, 224), val_mode=False):
    train_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(resize_size[0], padding=16),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    if val_mode:
        return val_transform
    return train_transform, val_transform