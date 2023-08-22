import socket
from datetime import datetime
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

def dataloader_collate_fn(data):
    """Creates padded batch of irregular sequences
    """

    # image, classes, coords, lengths

    data.sort(key=lambda x: x[-1], reverse=True)
    images, class_, x_, y_, bb_width, bb_height, seq_info_, lengths = zip(*data)

    images = torch.stack(images, 0)

    # print(images.shape)

    batch_size = images.shape[0]
    max_length = max(lengths)

    class_tokens = torch.zeros(batch_size, max_length).fill_(1) # <p/> token is @ index #1 in vocab
    x_tokens = torch.zeros(batch_size, max_length).fill_(1) # <p/> token is @ index #1 in vocab
    y_tokens = torch.zeros(batch_size, max_length).fill_(1) # <p/> token is @ index #1 in vocab
    bb_width_tokens = torch.zeros(batch_size, max_length).fill_(1) # <p/> token is @ index #1 in vocab
    bb_height_tokens = torch.zeros(batch_size, max_length).fill_(1) # <p/> token is @ index #1 in vocab
    seq_info_tokens = torch.zeros(batch_size, max_length).fill_(1) # <p/> token is @ index #1 in vocab

    for i, (classes_seq, x_seq, y_seq, width_seq, height_seq, seq_info_seq) in enumerate(zip(class_, x_, y_, bb_width, bb_height, seq_info_)):
        end = lengths[i] # length of this sequence

        class_tokens[i, :end] = classes_seq
        x_tokens[i, :end] = x_seq
        y_tokens[i, :end] = y_seq
        bb_width_tokens[i, :end] = width_seq
        bb_height_tokens[i, :end] = height_seq
        seq_info_tokens[i, :end] = seq_info_seq

    return images, class_tokens, x_tokens, y_tokens, bb_width_tokens, bb_height_tokens, seq_info_tokens, lengths

def denormalize(image):
    """Perform Inverse transforms on image

    Args:
        image : Image Tensor of shape (3, H, W)
    """

    mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1).to(image.device) # (3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1).to(image.device) # (3, 1, 1)

    image = (image*std) + mean

    return image

def get_hostname_and_time_string():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    hostname = socket.gethostname()
    string = f"{current_time}_{hostname}"
    return string


def write_to_tb(writer: SummaryWriter, global_index: int, net: nn.Module, scalars: dict, images: dict = {}, images_with_bbs: dict = {}):
    """Write data from training loop to tensorboard

    Args:
        writer : SummaryWriter
        global_index (int): Global Index
        net (nn.Module): Model
        lr (float): Learning rate
        scalars (dict): Dict of array of dict of scalars. E.g ->
        {
            "Loss": {
                "train": 0.9,
                "test": 0.5,
                ...
            },
            ...
        }

        images (dict): Dict of array of dict Images. Eg -> 
        {
            "FeatureMaps": {
                    "layer1.conv1": Tensor of size (C, H, W)
                    "layer2.conv2": Tensor of size (C, H, W)
                    ...
            },
            ...
        }

        images_with_bbs (dict): Dict of dicts of Images with bounding boxes. Eg ->
        {
            "Detections": {
                "0": {
                    "labels": List of length N,
                    "image": Tensor of size (C, H, W),
                    "bbs": Tensor of shape (N, 4)
                }
                ...
            },
        }
    """

    # write scalars

    for tag in scalars.keys():
        for name in scalars[tag].keys():
            value = scalars[tag][name]
            tb_tag = f"Scalars/{tag}/{name}"

            writer.add_scalar(tb_tag, value, global_index)

    # write images

    for tag in images.keys():
        for name in images[tag].keys():
            value = images[tag][name]
            tb_tag = f"Images/{tag}/{name}"

            writer.add_image(tb_tag, value, global_index)

    for tag in images_with_bbs.keys():
        for name in images_with_bbs[tag].keys():
            labels = images_with_bbs[tag][name]["labels"]
            image = images_with_bbs[tag][name]["image"]
            bbs = images_with_bbs[tag][name]["bbs"]

            tb_tag = f"Detections/{tag}/{name}"

            writer.add_image_with_boxes(
                tag=tb_tag, img_tensor=image,
                box_tensor=bbs, global_step=global_index,
                labels=labels
            )

    # parameters & gradients

    for name, parameter in net.named_parameters():
        if parameter.requires_grad and not isinstance(parameter.grad, type(None)):
            writer.add_histogram(name, parameter, global_index)
            writer.add_histogram(f"{name}.grad", parameter.grad, global_index)