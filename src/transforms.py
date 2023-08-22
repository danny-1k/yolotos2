from torchvision.transforms import transforms
from torchvision.transforms import functional as F

detection_transforms = lambda image_size: {
    'train': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomAdjustSharpness(2),
        transforms.RandomGrayscale(),
        transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(.1, 5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}