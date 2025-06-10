import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff
from medpy import metric

# 输入（单通道预测图，单通道预测图，评价类型）
# 注意，预测图与原图保持，通道，大小一致
# 参考参数（字符串）：iou,dice_coefficient,accuracy,precision,recall,sensitivity,f1,specificity
def calculate_metrics(predict_image, gt_image):

    # 将图像转换为二进制数组
    predict_image = np.array(predict_image, dtype=bool)
    gt_image = np.array(gt_image, dtype=bool)

    # 计算True Positive（TP）
    tp = np.sum(np.logical_and(predict_image, gt_image))

    # 计算True Negative（TN）
    tn = np.sum(np.logical_and(np.logical_not(predict_image), np.logical_not(gt_image)))

    # 计算False Positive（FP）
    fp = np.sum(np.logical_and(predict_image, np.logical_not(gt_image)))

    # 计算False Negative（FN）
    fn = np.sum(np.logical_and(np.logical_not(predict_image), gt_image))

    # 计算IOU（Intersection over Union）
    iou = tp / (tp + fn + fp + 1e-7)

    # 计算Dice Coefficient（Dice系数）
    dice_coefficient = 2 * tp / (2 * tp + fn + fp + 1e-7)

    # 计算Accuracy（准确率）
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-7)

    # 计算precision（精确率）
    precision = tp / (tp + fp + 1e-7)

    # 计算recall（召回率）
    recall = tp / (tp + fn + 1e-7)

    # 计算Sensitivity（敏感度）
    sensitivity = tp / (tp + fn + 1e-7)

    # 计算F1-score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    # 计算Specificity（特异度）
    specificity = tn / (tn + fp + 1e-7)


    # 计算 Hausdorff 距离
    def hausdorff_distance(set_a, set_b):
        """计算两个点集之间的Hausdorff距离"""
        if set_a.size == 0 and set_b.size == 0:
            return 0  # 如果两个点集都为空，返回 0
        elif (set_a.size == 0 and set_b.size != 0) or (set_b.size == 0 and set_a.size != 0):
            return float('inf')  # 如果其中一个点集为空，返回无穷大

        # 计算两个方向的Hausdorff距离
        d_ab = directed_hausdorff(set_a, set_b)[0]
        d_ba = directed_hausdorff(set_b, set_a)[0]

        return max(d_ab, d_ba)

    def calculate_metric_percase(pred, gt):
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        if pred.sum() > 0 and gt.sum() > 0:
            hd95 = metric.binary.hd95(pred, gt)
            return hd95
        elif pred.sum() > 0 and gt.sum() == 0:
            return 0
        else:
            return 0

    hd95 = calculate_metric_percase(predict_image.astype(np.uint8), gt_image.astype(np.uint8))
    # 提取预测图像和真实图像中的坐标点
    predict_coords = np.argwhere(predict_image)
    gt_coords = np.argwhere(gt_image)

    # 计算Hausdorff距离
    hausdorff_dist = hausdorff_distance(predict_coords, gt_coords)

    return hd95, iou, dice_coefficient, accuracy, precision, recall, sensitivity, f1, specificity

# def calculate_metrics(predict_image, gt_image, spacing=(1.0, 1.0, 1.0)):
#     # 计算True Positive（TP）
#     tp = np.sum(np.logical_and(predict_image, gt_image))
#
#         # 计算True Negative（TN）
#     tn = np.sum(np.logical_and(np.logical_not(predict_image), np.logical_not(gt_image)))
#
#         # 计算False Positive（FP）
#     fp = np.sum(np.logical_and(predict_image, np.logical_not(gt_image)))
#
#         # 计算False Negative（FN）
#     fn = np.sum(np.logical_and(np.logical_not(predict_image), gt_image))
#
#         # 计算IOU（Intersection over Union）
#     iou = tp / (tp + fn + fp + 1e-7)
#
#         # 计算Dice Coefficient（Dice系数）
#     dice_coefficient = 2 * tp / (2 * tp + fn + fp + 1e-7)
#
#         # 计算Accuracy（准确率）
#     accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-7)
#
#         # 计算precision（精确率）
#     precision = tp / (tp + fp + 1e-7)
#
#         # 计算recall（召回率）
#     recall = tp / (tp + fn + 1e-7)
#
#         # 计算Sensitivity（敏感度）
#     sensitivity = tp / (tp + fn + 1e-7)
#
#         # 计算F1-score
#     f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
#
#         # 计算Specificity（特异度）
#     specificity = tn / (tn + fp + 1e-7)
#     # 显式二值化 + 类型转换
#     predict_image = (np.asarray(predict_image) > 0.5).astype(bool)
#     gt_image = (np.asarray(gt_image) > 0.5).astype(bool)
#
#     # 坐标点降采样 (降低计算量)
#     def downsample(coords, factor=2):
#         return coords // factor if factor > 1 else coords
#
#     predict_coords = downsample(np.argwhere(predict_image))
#     gt_coords = downsample(np.argwhere(gt_image))
#
#     # 计算Hausdorff距离（带物理单位）
#     def hausdorff_95(set_a, set_b):
#         if set_a.size == 0 and set_b.size == 0:
#             return 0.0
#         elif set_a.size == 0 or set_b.size == 0:
#             return float('inf')
#
#         # 应用物理间距
#         set_a = set_a.astype(float) * np.array(spacing)
#         set_b = set_b.astype(float) * np.array(spacing)
#
#         # 计算95% Hausdorff距离
#         dist_matrix = np.sqrt(((set_a[:, None] - set_b) ** 2).sum(axis=2))
#         d_ab = np.percentile(np.min(dist_matrix, axis=1), 95)
#         d_ba = np.percentile(np.min(dist_matrix, axis=0), 95)
#         return max(d_ab, d_ba)
#
#     hausdorff_dist = hausdorff_95(predict_coords, gt_coords)
#
#
#
#     return hausdorff_dist, iou, dice_coefficient, accuracy, precision, recall, sensitivity, f1, specificity

    # if evaluate == "iou":
    #     return iou
    #
    # if evaluate == "dice_coefficient":
    #     return dice_coefficient
    #
    # if evaluate == "accuracy":
    #     return accuracy
    #
    # if evaluate == "precision":
    #     return precision
    #
    # if evaluate == "recall":
    #     return recall
    #
    # if evaluate == "sensitivity":
    #     return sensitivity
    #
    # if evaluate == "f1":
    #     return f1
    #
    # if evaluate == "specificity":
    #     return specificity
