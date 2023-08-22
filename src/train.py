import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from models import build_model
from data import build_dataset
from utils import dataloader_collate_fn, write_to_tb
from tqdm import tqdm
from loss import Loss


def train_one_epoch(net, criterion, data, optimizer, device, iteration, writer):
    net.train()
    criterion.train()

    net.to(device)
    criterion.to(device)

    average_loss = 0
    average_closs_loss = 0
    average_x_loss = 0
    average_y_loss = 0
    average_w_loss = 0
    average_h_loss = 0
    average_seq_loss = 0

    average_accuracy = 0
    average_class_accuracy = 0
    average_x_accuracy = 0
    average_y_accuracy = 0
    average_w_accuracy = 0
    average_h_accuracy = 0
    average_seq_accuracy = 0


    for image, target_class, target_x, target_y, target_w, target_h, target_seq, _ in tqdm(data):
        image = image.to(device)
        target_class = target_class.to(device).long()
        target_x = target_x.to(device).long()
        target_y = target_y.to(device).long()
        target_w = target_w.to(device).long()
        target_h = target_h.to(device).long()
        target_seq = target_seq.to(device).long()

        optimizer.zero_grad()

        out_class, out_x, out_y, out_w, out_h, out_isdone = net(
            image,
            (target_class, target_x, target_y, target_w, target_h, target_seq),

        )

        loss, (class_loss, x_loss, y_loss, w_loss, h_loss, isdone_loss) = criterion(
            p=(out_class, out_x, out_y, out_w, out_h, out_isdone),
            y=(target_class, target_x, target_y, target_w, target_h, target_seq)
        )

        loss.backward()
        optimizer.step()

        average_loss = .2*average_loss + .8*loss.item()
        average_class_accuracy = .2*average_class_accuracy + .8*class_loss.item()
        average_x_loss = .2*average_x_loss + .8*x_loss.item()
        average_y_loss = .2*average_y_loss + .8*y_loss.item()
        average_w_loss = .2*average_w_loss + .8*w_loss.item()
        average_h_loss = .2*average_h_loss + .8*h_loss.item()
        average_seq_loss = .2*average_seq_loss + .8*isdone_loss.item()


        class_accuracy = (out_class.view(-1, out_class.shape[-1]).argmax(-1) == target_class.view(-1)).float().mean()
        x_accuracy = (out_x.view(-1, out_x.shape[-1]).argmax(-1) == target_x.view(-1)).float().mean()
        y_accuracy = (out_y.view(-1, out_y.shape[-1]).argmax(-1) == target_y.view(-1)).float().mean()
        w_accuracy = (out_w.view(-1, out_w.shape[-1]).argmax(-1) == target_w.view(-1)).float().mean()
        h_accuracy = (out_h.view(-1, out_h.shape[-1]).argmax(-1) == target_h.view(-1)).float().mean()
        seq_accuracy = (out_isdone.view(-1, out_isdone.shape[-1]).argmax(-1) == target_seq.view(-1)).float().mean()

        accuracy = (class_accuracy + x_accuracy + y_accuracy + w_accuracy + h_accuracy + seq_accuracy) / 6

        print(accuracy, average_loss)

        average_accuracy = .6*average_accuracy + accuracy.item()
        average_class_accuracy = .6*average_class_accuracy + .4*class_accuracy
        average_x_accuracy = .6*average_x_accuracy + .4*x_accuracy
        average_y_accuracy = .6*average_y_accuracy + .4*y_accuracy
        average_w_accuracy = .6*average_w_accuracy + .4*w_accuracy
        average_h_accuracy = .6*average_h_accuracy + .4*h_accuracy
        average_seq_accuracy = .6*average_seq_accuracy + .4*seq_accuracy

    write_to_tb(
        writer=writer,
        global_index=iteration,
        net=net,
        scalars={
            "Loss/train": {
                "average_loss": average_loss,
                "average_class": average_closs_loss,
                "average_x": average_x_loss,
                "average_y": average_y_loss,
                "average_w": average_w_loss,
                "average_h": average_h_loss,
                "average_seq": average_seq_loss
            },

            "Accuracy/train": {
                "average_accuracy": average_accuracy,
                "average_class": average_class_accuracy,
                "average_x": average_x_accuracy,
                "average_y": average_y_accuracy,
                "average_w": average_w_accuracy,
                "average_h": average_h_accuracy,
                "average_seq": average_seq_accuracy
            }
        },
    )

    return iteration, average_loss

@torch.no_grad()
def eval_model(net, criterion, data, device, iteration, writer):
    net.eval()
    criterion.eval()

    net.to(device)
    criterion.to(device)

    average_loss = 0
    average_closs_loss = 0
    average_x_loss = 0
    average_y_loss = 0
    average_w_loss = 0
    average_h_loss = 0
    average_seq_loss = 0

    average_accuracy = 0
    average_class_accuracy = 0
    average_x_accuracy = 0
    average_y_accuracy = 0
    average_w_accuracy = 0
    average_h_accuracy = 0
    average_seq_accuracy = 0

    for image, target_class, target_x, target_y, target_w, target_h, target_seq, _ in tqdm(data):
        image = image.to(device)
        target_class = target_class.to(device).long()
        target_x = target_x.to(device).long()
        target_y = target_y.to(device).long()
        target_w = target_w.to(device).long()
        target_h = target_h.to(device).long()
        target_seq = target_seq.to(device).long()

        out_class, out_x, out_y, out_w, out_h, out_isdone = net(
            image,
            (target_class, target_x, target_y, target_w, target_h, target_seq),

        )

        loss, (class_loss, x_loss, y_loss, w_loss, h_loss, isdone_loss) = criterion(
            p=(out_class, out_x, out_y, out_w, out_h, out_isdone),
            y=(target_class, target_x, target_y, target_w, target_h, target_seq)
        )

        average_loss = .2*average_loss + .8*loss.item()
        average_class_accuracy = .2*average_class_accuracy + .8*class_loss.item()
        average_x_loss = .2*average_x_loss + .8*x_loss.item()
        average_y_loss = .2*average_y_loss + .8*y_loss.item()
        average_w_loss = .2*average_w_loss + .8*w_loss.item()
        average_h_loss = .2*average_h_loss + .8*h_loss.item()
        average_seq_loss = .2*average_seq_loss + .8*isdone_loss.item()


        class_accuracy = (out_class.view(-1, out_class.shape[-1]).argmax(-1) == target_class.view(-1)).float().mean()
        x_accuracy = (out_x.view(-1, out_x.shape[-1]).argmax(-1) == target_x.view(-1)).float().mean()
        y_accuracy = (out_y.view(-1, out_y.shape[-1]).argmax(-1) == target_y.view(-1)).float().mean()
        w_accuracy = (out_w.view(-1, out_w.shape[-1]).argmax(-1) == target_w.view(-1)).float().mean()
        h_accuracy = (out_h.view(-1, out_h.shape[-1]).argmax(-1) == target_h.view(-1)).float().mean()
        seq_accuracy = (out_isdone.view(-1, out_isdone.shape[-1]).argmax(-1) == target_seq.view(-1)).float().mean()

        accuracy = (class_accuracy + x_accuracy + y_accuracy + w_accuracy + h_accuracy + seq_accuracy) / 6

        average_accuracy = .6*average_accuracy + accuracy.item()
        average_class_accuracy = .6*average_class_accuracy + .4*class_accuracy
        average_x_accuracy = .6*average_x_accuracy + .4*x_accuracy
        average_y_accuracy = .6*average_y_accuracy + .4*y_accuracy
        average_w_accuracy = .6*average_w_accuracy + .4*w_accuracy
        average_h_accuracy = .6*average_h_accuracy + .4*h_accuracy
        average_seq_accuracy = .6*average_seq_accuracy + .4*seq_accuracy


    write_to_tb(
        writer=writer,
        global_index=iteration,
        net=net,
        scalars={
            "Loss/test": {
            "average_loss": average_loss,
            "average_class": average_closs_loss,
            "average_x": average_x_loss,
            "average_y": average_y_loss,
            "average_w": average_w_loss,
            "average_h": average_h_loss,
            "average_seq": average_seq_loss
        },

        "Accuracy/test": {
            "average_accuracy": average_accuracy,
            "average_class": average_class_accuracy,
            "average_x": average_x_accuracy,
            "average_y": average_y_accuracy,
            "average_w": average_w_accuracy,
            "average_h": average_h_accuracy,
            "average_seq": average_seq_accuracy
        }
    })


    return average_loss


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dataset == "voc":
        args.num_classes = 20

    device = args.device

    net = build_model(args)

    criterion = Loss()

    net.backbone.to(device)

    net.to(device)
    criterion.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=args.factor, mode="min")

    # create necessary folders

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if not os.path.exists(os.path.join(args.checkpoint_dir, args.run_name)):
        os.makedirs(os.path.join(args.checkpoint_dir, args.run_name))


    writer = SummaryWriter(log_dir=f"{args.logdir}/{args.run_name}")


    train = build_dataset(
        name=args.dataset,
        split="val" if args.no_eval else "train",
        args=args
    )

    train = DataLoader(
        train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=dataloader_collate_fn
    )

    if not args.no_eval:

        test = build_dataset(
            name=args.dataset,
            split="val",
            args=args
        )

        test = DataLoader(
            test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataloader_collate_fn
        )


    number_of_train_iters = 0

    print("Started Training.")

    for epoch in range(args.epochs):

        number_of_train_iters, average_train_loss = train_one_epoch(
            net=net,
            criterion=criterion,
            data=train,
            optimizer=optimizer,
            device=device,
            iteration=number_of_train_iters,
            writer=writer,
        )

        for checkpoint_path in [f"{args.checkpoint_dir}/{args.run_name}/checkpoint.pth", f"{args.checkpoint_dir}/{args.run_name}/checkpoint{epoch: 04}.pth"]:
            torch.save({
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "loss": average_train_loss,
            }, checkpoint_path)


        if not args.no_eval:
            average_test_loss = eval_model(
                net=net,
                criterion=criterion,
                data=test,
                device=device,
                iteration=number_of_train_iters,
                writer=writer,
            )

        lr_scheduler.step(average_test_loss)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from utils import get_hostname_and_time_string

    unique_string = get_hostname_and_time_string()

    parser = ArgumentParser()

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument("--lr_drop", default=100, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--factor", default=.1, type=float) # for decay

    # Network
    parser.add_argument("--bins", default=100)
    parser.add_argument("--input_size", default=(224, 224))
    parser.add_argument('--backbone', default='vgg16', type=str,
                        help="Encoder. `vgg16`, `vgg19`, `resnet50`")

    parser.add_argument('--hidden_size', default=128, type=int,
                        help="Hidden size of decoder GRU")
    
    parser.add_argument('--dropout', default=0,
                        type=float, help="Dropout")
    
    # Misc
    parser.add_argument('--dataset', default='voc', type=str)
    parser.add_argument('--voc_year', default='2007', type=str)
    parser.add_argument("--data_root", default="../data/voc", type=str)
    parser.add_argument("--download_dataset", action="store_true")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--seed', default=3417, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument("--logdir", default="../logs",
                        type=str, help="Logdir for tensorboard")
    
    parser.add_argument("--run_name", default=unique_string,
                         type=str, help="Unique string for saving checkpoints and tensorboard")
    
    parser.add_argument("--checkpoint_dir", default="../checkpoints/")

    parser.add_argument("--no_eval", action="store_true")

    args = parser.parse_args()


    run(args)
