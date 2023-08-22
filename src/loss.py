from torch import nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, p, y):
        predicted_class, predicted_x, predicted_y, predicted_w, predicted_h, predicted_isdone = p
        truth_class, truth_x, truth_y, truth_w, truth_h, truth_isdone = y

        class_loss = self.lossfn(predicted_class.view(-1, predicted_class.shape[-1]), truth_class.view(-1))
        x_loss = self.lossfn(predicted_x.view(-1, predicted_x.shape[-1]), truth_x.view(-1))
        y_loss = self.lossfn(predicted_y.view(-1, predicted_y.shape[-1]), truth_y.view(-1))
        w_loss = self.lossfn(predicted_w.view(-1, predicted_w.shape[-1]), truth_w.view(-1))
        h_loss = self.lossfn(predicted_h.view(-1, predicted_h.shape[-1]), truth_h.view(-1))
        isdone_loss = self.lossfn(predicted_isdone.view(-1, predicted_isdone.shape[-1]), truth_isdone.view(-1))

        loss = class_loss + x_loss + y_loss + w_loss + h_loss + isdone_loss

        return loss, (class_loss, x_loss, y_loss, w_loss, h_loss, isdone_loss)