import torch
from torch import nn
from torchvision.models import vgg16, vgg16, vgg19, resnet50
from functools import reduce


class Backbone(nn.Module):
    def __init__(self, model, input_size):
        super().__init__()
        self.model = model
        self.input_size = input_size

        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        return x

    @classmethod
    def get_backbone(cls, name, input_size=(224, 224)):
        if name == "vgg16":
            model = vgg16(pretrained=True).features
            x = torch.zeros((1, 3, *input_size))
            p = model(x)
            shape = p.shape[1:]
            num_el = reduce(lambda x, y: x * y, shape)
            return cls(model=model, input_size=input_size), num_el

        if name == "vgg19":
            model = vgg19(pretrained=True).features
            x = torch.zeros((1, 3, *input_size))
            p = model(x)
            shape = p.shape[1:]
            num_el = reduce(lambda x, y: x * y, shape)
            return cls(model=model, input_size=input_size), num_el

        if name == "resnet":
            model = resnet50(pretrained=True)
            model.fc = nn.Identity()

            x = torch.zeros((1, 3, *input_size))
            p = model(x)
            shape = p.shape[1:]
            num_el = reduce(lambda x, y: x * y, shape)
            return cls(model=model, input_size=input_size), num_el


class YOLOTOS(nn.Module):
    def __init__(self, input_size, num_classes, num_bins, backbone_name, hidden_size, dropout=0):
        super().__init__()
        self.backbone, self.backbone_output_dim = Backbone.get_backbone(name=backbone_name, input_size=input_size)
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.bin_embed = nn.Embedding(num_bins+2, hidden_size)
        self.class_embed = nn.Embedding(num_classes+2, hidden_size)
        self.backbone_out = nn.Linear(self.backbone_output_dim, hidden_size)

        self.class_out = nn.Linear(hidden_size, num_classes+2)
        self.pos_out = nn.Linear(hidden_size, (num_bins+2)*4)
        self.isdone_out = nn.Linear(hidden_size, 3)

        self.decoder = nn.GRUCell(input_size=hidden_size*5, hidden_size=hidden_size)
        
        self.attn = nn.MultiheadAttention(embed_dim=1, num_heads=1, dropout=dropout, batch_first=True)


    def forward(self, x, y):
        # y of shape (N, S, 5)
        y_class, y_x, y_y, y_w, y_h, y_isdone = y


        timesteps = y_class.shape[1]

        decoder_hidden = None

        image_x = self.backbone(x)
        image_x = self.backbone_out(image_x)

        batch_size = image_x.shape[0]

        out_class = torch.zeros(batch_size, timesteps, self.num_classes+2)
        out_x = torch.zeros(batch_size, timesteps, self.num_bins+2)
        out_y = torch.zeros(batch_size, timesteps, self.num_bins+2)
        out_w = torch.zeros(batch_size, timesteps, self.num_bins+2)
        out_h = torch.zeros(batch_size, timesteps, self.num_bins+2)
        out_isdone = torch.zeros(batch_size, timesteps, 3)

        for t in range(timesteps):
            if t == 0:
                class_t = torch.zeros((x.shape[0])).long() # SOS
                x_center_t = torch.zeros((x.shape[0])).long() # SOS
                y_center_t = torch.zeros((x.shape[0])).long() # SOS
                bb_width_t = torch.zeros((x.shape[0])).long() # SOS
                bb_height_t = torch.zeros((x.shape[0])).long() # SOS

                x1 = self.class_embed(class_t) # (N, E)
                x2 = self.bin_embed(x_center_t) # (N, E)
                x3 = self.bin_embed(y_center_t) # (N, E)
                x4 = self.bin_embed(bb_width_t) # (N, E)
                x5 = self.bin_embed(bb_height_t) # (N, E)

                x = torch.cat((x1, x2, x3, x4, x5), dim=1)
            
            else: # teacher forcing
                # x1 = out_class[:, t-1].argmax(-1).long()
                # x2 = out_x[:, t-1].argmax(-1).long()
                # x3 = out_y[:, t-1].argmax(-1).long()
                # x4 = out_w[:, t-1].argmax(-1).long()
                # x5 = out_h[:, t-1].argmax(-1).long()

                x1 = y_class[:, t-1]
                x2 = y_x[:, t-1]
                x3 = y_y[:, t-1]
                x4 = y_w[:, t-1]
                x5 = y_h[:, t-1]

                x1 = self.class_embed(x1) # (N, E)
                x2 = self.bin_embed(x2) # (N, E)
                x3 = self.bin_embed(x3) # (N, E)
                x4 = self.bin_embed(x4) # (N, E)
                x5 = self.bin_embed(x5) # (N, E)

                x = torch.cat((x1, x2, x3, x4, x5), dim=1)


            decoder_hidden = self.decoder(x, decoder_hidden)

            decoder_hidden, attention_weights = self.attn(
                key=image_x.view(batch_size, -1, 1),
                value=image_x.view(batch_size, -1, 1),
                query=decoder_hidden.view(batch_size, -1, 1)
            )
            decoder_hidden = decoder_hidden.view(batch_size, -1)

            classes = self.class_out(decoder_hidden)
            out_positions = self.pos_out(decoder_hidden)
            is_last = self.isdone_out(decoder_hidden)

            out_positions = out_positions.view(batch_size, 4, -1)

            out_class[:, t, :] = classes
            out_x[:, t, :] = out_positions[:, 0]
            out_y[:, t, :] = out_positions[:, 1]
            out_w[:, t, :] = out_positions[:, 2]
            out_w[:, t, :] = out_positions[:, 3]
            out_isdone[:, t, :] = is_last

        return out_class, out_x, out_y, out_w, out_h, out_isdone
    

def build_model(args):
    backbone = args.backbone
    dropoout = args.dropout
    hidden_size = args.hidden_size
    num_bins = args.bins
    num_classes = args.num_classes
    input_size = args.input_size

    net = YOLOTOS(
        input_size=input_size,
        num_classes=num_classes,
        num_bins=num_bins,
        backbone_name=backbone,
        hidden_size=hidden_size,
        dropout=dropoout,
    )

    return net
    


if __name__ == "__main__":
    net = YOLOTOS(
        input_size=(224, 224),
        num_classes=3,
        num_bins=100,
        backbone_name="vgg16",
        hidden_size=128,
        dropout=0
    )

    x = torch.zeros((1, 3, 224, 224))
    y = (
        torch.zeros((1, 5)).long(), #classes
        torch.zeros((1, 5)).long(), # x
        torch.zeros((1, 5)).long(), # y
        torch.zeros((1, 5)).long(), # w
        torch.zeros((1, 5)).long(), # h
        torch.zeros((1, 5)).long(), # isdone
    )

    out_class, out_x, out_y, out_w, out_h, out_isdone = net(x, y)

    print(out_class.shape)