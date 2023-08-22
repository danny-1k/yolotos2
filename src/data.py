import json
import torch
from torch.utils.data import Dataset
from transforms import detection_transforms
from torchvision.datasets.voc import VOCDetection
import random
from vocab import Encoder, Binning, Classes

def label_from_voc(vocab, annotations, width, height, shuffle=True):
    """Convert VOC annotations to YOLOTOS format for training

    Args:
        vocab : Vocab class
        annotations : PascalVOC annotations
        width : Width of image (Used for normalization)
        height : Height of image (Used for normalization)
        shuffle: Randomly shuffle if True
    """
    class_tokens = []
    x_center_tokens = []
    y_center_tokens = []
    bb_width_tokens = []
    bb_height_tokens = []
    seq_info = []

    if shuffle:
        random.shuffle(annotations)

    for idx, annotation in enumerate(annotations):
        obj_class = annotation['name']
        obj_bbox = annotation['bndbox']

        xmin = int(obj_bbox['xmin'])
        xmax = int(obj_bbox['xmax'])
        ymin = int(obj_bbox['ymin'])
        ymax = int(obj_bbox['ymax'])

        x_center = ((xmin + xmax)//2)/width  # normalized x_center
        y_center = ((ymin + ymax)//2)/height  # normalized y_center

        bb_width = (xmax - xmin)/width  # normalized bb width
        bb_height = (ymax - ymin)/height  # normalized bb height

        x_center = min(x_center, 1)
        y_center = min(y_center, 1)
        bb_width = min(bb_width, 1)
        bb_height = min(bb_height, 1)
        
        class_tokens.append(vocab.encode_class([obj_class])[0])
        x_center_tokens.append(vocab.encode_pos([x_center])[0])
        y_center_tokens.append(vocab.encode_pos([y_center])[0])
        bb_width_tokens.append(vocab.encode_pos([bb_width])[0])
        bb_height_tokens.append(vocab.encode_pos([bb_height])[0])
        seq_info.append(0 if idx == 0 else 1 if (len(annotations)-1) else 2) # 0 signifies START, 1 signifies STOP, 2 signifies CONTINUE/ THERES STILL ANOTHER OBJECT


    return class_tokens, x_center_tokens, y_center_tokens, bb_width_tokens, bb_height_tokens, seq_info, len(annotations)


class PascalVocDataset(Dataset):
    def __init__(self, vocab, image_size, root="../data/voc", download=False, year="2007", image_set="val", shuffle=None) -> None:
        self.vocab = vocab
        self.download = download
        self.year = year
        self.image_set = image_set
        self.train = True if "train" in image_set else "test"
        self.transform = detection_transforms(image_size)["train" if self.train else "test"]
        self.shuffle = shuffle or self.train

        self.dataset = VOCDetection(
            root=root, year=year,
            image_set=image_set,
            transform=None,
            download=download

        )

    def __getitem__(self, index):
        image, annotation = self.dataset[index]

        annotation = annotation["annotation"]

        w = int(annotation['size']['width'])
        h = int(annotation['size']['height'])

        annotations = annotation['object']

        class_tokens, x_center_tokens, y_center_tokens, bb_width_tokens, bb_height_tokens, seq_info, length = label_from_voc(
            vocab=self.vocab,
            annotations=annotations,
            width=w, height=h,
            shuffle=self.shuffle
        )

        class_tokens = torch.Tensor(class_tokens).long()
        x_center_tokens = torch.Tensor(x_center_tokens).long()
        y_center_tokens = torch.Tensor(y_center_tokens).long()
        bb_width_tokens = torch.Tensor(bb_width_tokens).long()
        bb_height_tokens = torch.Tensor(bb_height_tokens).long()
        seq_info = torch.Tensor(seq_info).long()

        image = self.transform(image)

        return (image, class_tokens, x_center_tokens, y_center_tokens, bb_width_tokens, bb_height_tokens, seq_info, length)

    def __len__(self):
        return len(self.dataset)


def build_dataset(name, split, args):
    if name == "voc":
        vocab = Encoder(
            classes=Classes(list(json.load(open(f"./voc.json")).keys())),
            binning=Binning(bins=args.bins)
        )

        data = PascalVocDataset(
            image_size=args.input_size,
            vocab=vocab,
            root=args.data_root,
            year=args.voc_year,
            image_set=split
        )

        return data