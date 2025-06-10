import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

# Try reducing the BCE to 1/2


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()

        dice_loss = 1 - (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        # inputs = torch.sigmoid(inputs)
        #
        # # Flatten inputs and targets
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        #
        # print(f"Min: {inputs.min()}, Max: {inputs.max()}")
        #
        # # 使用 torch.clamp 将输入值裁剪到 [0, 1] 范围内
        # inputs_clipped = torch.clamp(inputs, 0, 1)
        #
        # # 再次打印裁剪后的数据的范围
        # print(f"Clipped Min: {inputs_clipped.min()}, Clipped Max: {inputs_clipped.max()}")
        #
        # # 检查裁剪后的输入值是否在 [0, 1] 范围内
        # assert torch.all(inputs_clipped >= 0) and torch.all(inputs_clipped <= 1), "Inputs are not in the range [0, 1]"
        # inputs = torch.clamp(inputs, 0, 1)
        # assert torch.all(inputs >= 0) and torch.all(inputs <= 1), "Inputs are not in the range [0, 1]"
        # assert torch.all(targets >= 0) and torch.all(targets <= 1), "Targets are not binary"
        #
        #
        # # Calculate the intersection and number of non-zero elements in inputs and targets
        # intersection = (inputs * targets).sum()
        # num_inputs = inputs.sum()
        # num_targets = targets.sum()
        #
        # # Calculate the Dice coefficient
        # dice_coeff = (2. * intersection + smooth) / (num_inputs + num_targets + smooth)
        #
        # # Calculate the Dice loss
        # dice_loss = 1 - dice_coeff
        #
        # # Calculate the Binary Cross Entropy loss
        # BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        #
        # # Combine the losses
        # Dice_BCE = BCE + dice_loss

        return Dice_BCE

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# class FocalDiceLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
#         super(FocalDiceLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce
#
#     def forward(self, inputs, targets,smooth=1):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(
#                 inputs, targets, reduce=False)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
#
#         inputs = torch.sigmoid(inputs)
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         intersection = (inputs * targets).sum()
#
#         dice_loss = 1 - (2. * intersection + smooth) / \
#                     (inputs.sum() + targets.sum() + smooth)
#
#         FocalDice = F_loss + dice_loss
#
#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return FocalDice

class FocalDiceLoss(nn.Module):
    def __init__(self,  alpha=0.25, gamma=2):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()

        dice_loss = 1 - (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)


        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        Dice_BCE = F_loss + dice_loss

        return Dice_BCE


# class BceDiceLoss(nn.Module):
#     def __init__(self, wb=1, wd=1):
#         super(BceDiceLoss, self).__init__()
#         self.bce = BCELoss()
#         self.dice = DiceLoss()
#         self.wb = wb
#         self.wd = wd
#
#     def forward(self, pred, target):
#         bceloss = self.bce(pred, target)
#         diceloss = self.dice(pred, target)
#
#         loss = self.wd * diceloss + self.wb * bceloss
#         return loss


class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = DiceBCELoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3,
                                                                                                           target) * 0.3 + self.bcedice(
            gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
        return bcediceloss + gt_loss
