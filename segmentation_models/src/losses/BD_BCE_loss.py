# encoding : utf-8
import torch.nn as nn
import torch
import segmentation_models_pytorch as segu

"""
# custom metric shoul have self.__name__ 
"""


class BoundarySensitiveLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2] + 2, target.shape[-1] + 2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(),
                                      padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target, weight=None, sigmoid=False):
        if sigmoid:
            inputs = torch.sigmoid(inputs)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        BD_loss = self._adaptive_size(inputs[:, 0], target[:, 0])
        return BD_loss


class Mask_BD_and_BCE_loss(nn.Module):
    def __init__(self, pos_weight, bd_weight=0.5):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_BD_and_BCE_loss, self).__init__()
        self.__name__ = 'BD_BCE'
        self.ce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bd = BoundarySensitiveLoss()
        self.bd_weight = bd_weight

    def forward(self, net_output, target):
        low_res_logits = net_output
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            low_res_logits = low_res_logits.view(-1, low_res_logits.shape[2], low_res_logits.shape[3],
                                                 low_res_logits.shape[4])
        loss_ce = self.ce(low_res_logits, target)
        loss_dice = self.bd(low_res_logits, target, sigmoid=True)
        loss = (1 - self.bd_weight) * loss_ce + self.bd_weight * loss_dice
        return loss
