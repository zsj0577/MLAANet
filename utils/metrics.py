import numpy as np
import torch


class Metric:
    def __init__(self, mode):
        self.base_dice_scores = []
        self.iou_scores = []

    def compare(self, target, output):
        smooth = 1e-5
        output = torch.sigmoid(output).data.cpu().numpy()
        target = target.data.cpu().numpy()
        output = output > 0.5
        target = target > 0.5
        intersection = (output * target).sum() + smooth
        union = (output.sum() + target.sum() + 2*smooth) - intersection
        return intersection, union

    def calculate_dice_coef(self, intersection, union):
        """
        F1 or Dice Score :: 2|AB|/ |A| + |B|
        """
        dice_coef = (2. * intersection) / (intersection + union)
        return dice_coef

    def calculate_iou_score(self, intersection, union):
        """
        IOU or Jaccard Score :: |AB|/|A U B|
        """
        iou_score = (intersection)/(union)
        return iou_score

    def update(self, target, output):
        intersection, union = self.compare(target, output)
        self.base_dice_scores.append(
            self.calculate_dice_coef(intersection, union)
        )
        self.iou_scores.append(
            self.calculate_iou_score(intersection, union)
        )

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        iou = np.nanmean(self.iou_scores)
        return dice, iou

    def log(self, mode, epoch, epoch_loss):
        dice, iou = self.get_metrics()
        print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f" % (epoch_loss, iou, dice))
