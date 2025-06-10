from utils.metrics import Metric
from utils.calculate_metrics import calculate_metrics
from utils.utils import (
    seeding,
    create_dir,
    plot,
)

from losses.diceloss import DiceBCELoss
from models.MLAANet import MLAANet
from data.dataloader import GlasDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from glob import glob
import csv
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d_%H%M")

def exponential_lr_decay(optimizer, epoch, decay_rate=0.9, min_lr=1e-6):

    if epoch % 5 == 0:
        lr = optimizer.param_groups[0]['lr']
        new_lr = lr * decay_rate
        new_lr = max(new_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr
    else:
        return optimizer.param_groups[0]['lr']

class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model,datasetype):
        self.datasetype = datasetype
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        train_path = 'dataset/' + datasetype + '/train/imgs/*'
        test_path = 'dataset/' + datasetype + '/test/imgs/*'
        self.train_x = sorted(glob(train_path))
        self.valid_x = sorted(glob(test_path))
        self.train_data = GlasDataset(self.train_x)
        self.valid_data = GlasDataset(self.valid_x)

        self.num_workers = 0
        self.batch_size = {"train": 4, "val": 4}
        self.lr = 1e-4
        self.modes = ["train", "val"]
        self.num_epochs = 3
        self.criterion = DiceBCELoss()

        self.best_loss = float("inf")
        self.net = model
        self.net = self.net.to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)
        self.dataloaders = {
            "train": DataLoader(dataset=self.train_data, batch_size=self.batch_size['train'], shuffle=False, drop_last=True, num_workers=self.num_workers),
            "val": DataLoader(dataset=self.valid_data, batch_size=self.batch_size['val'], shuffle=False, drop_last=True, num_workers=self.num_workers)
        }
        # Defining the Containers
        self.losses = {mode: [] for mode in self.modes}
        self.iou_scores = {mode: [] for mode in self.modes}
        self.dice_scores = {mode: [] for mode in self.modes}

    def iterate(self, epoch, mode):
        running_loss = 0.0
        counter = 0
        dataloader = self.dataloaders[mode]
        total_batches = len(dataloader)
        metrics = Metric(mode)

        self.net.train() if mode == "train" else self.net.eval()

        for i, (x, y) in enumerate(dataloader):
            counter += 1
            image, mask = x.float().to(device), y.float().to(device)
            if mode == "train":
                self.optimizer.zero_grad()
            if self.net.name == 'PSPNet':
                outputs,_ = self.net(image)[0]
            else:
                outputs = self.net(image)
            loss = self.criterion(outputs, mask)
            running_loss += loss.item()
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item()
            outputs = outputs.detach()
            metrics.update(mask, outputs)

        dice, iou = metrics.get_metrics()
        epoch_loss = running_loss / counter
        metrics.log(mode, epoch, epoch_loss)



        self.losses[mode].append(epoch_loss)
        self.dice_scores[mode].append(dice)
        self.iou_scores[mode].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss  # for val loss

    def start(self):
        for epoch in tqdm(range(self.num_epochs)):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            lr = exponential_lr_decay(self.optimizer, epoch)
            print(f"Epoch {epoch}, Learning Rate: {lr}")
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
            if val_loss < self.best_loss:
                print("############ Improved Model --- Saving.. ############")
                print("Epoch = %d" %(epoch))
                state["best_loss"] = self.best_loss = val_loss
                model_path = "results\\models_Bestfile"
                os.makedirs(model_path, exist_ok=True)
                file_name = f"{self.datasetype}_{self.net.name()}_{str(self.criterion)}_{str(self.num_epochs)}_{str(self.batch_size['train'])}_{current_time}_best_model.pth"
                full_model_path = os.path.join(model_path, file_name)
                torch.save(state["state_dict"], full_model_path)


        print("############ Training Completed !! ############ ")

    def dice_coef(y_true, y_pred):
        """
        :param y_true:
        :param y_pred:
        :return:
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        intersection = np.sum(y_true * y_pred)
        return (2. * intersection + 1.0) / (np.sum(y_true) + np.sum(y_pred) + 1.0)
    def OnlyTesting(self):
        HD = np.zeros(len(self.valid_data), dtype=float)
        iou = np.zeros(len(self.valid_data), dtype=float)
        dice_coefficient = np.zeros(len(self.valid_data), dtype=float)
        accuracy = np.zeros(len(self.valid_data), dtype=float)
        precision = np.zeros(len(self.valid_data), dtype=float)
        recall = np.zeros(len(self.valid_data), dtype=float)
        sensitivity = np.zeros(len(self.valid_data), dtype=float)
        f1 = np.zeros(len(self.valid_data), dtype=float)
        specificity = np.zeros(len(self.valid_data), dtype=float)

        PA = np.zeros(len(self.valid_data), dtype=float)
        MPA = np.zeros(len(self.valid_data), dtype=float)
        IOU = np.zeros(len(self.valid_data), dtype=float)
        MIOU = np.zeros(len(self.valid_data), dtype=float)

        # Open the TXT file for writing F1 scores
        f1_dir = f"results/{self.datasetype}_F1_Files"
        os.makedirs(f1_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Define the path for the TXT file
        txt_file_path = f"{f1_dir}/{self.net.name()}_f1_scores.txt"
        with open(txt_file_path, 'w') as f:  # Open file in write mode

            for i in range(len(self.valid_data)):
                data = self.valid_data.__getitem__(i)
                img = data[0].unsqueeze(0).to(device)
                mask = data[1].squeeze(0)

                if self.net.name() == 'PSPNet':
                    output, _ = self.net(img)
                else:
                    output = self.net(img)

                output = torch.squeeze(output)
                output = torch.sigmoid(output)
                output = output > 0.5
                output = output + 0.0

                M = mask.detach().cpu().numpy()
                P = output.detach().cpu().numpy()

                # Calculate metrics
                HD[i], iou[i], dice_coefficient[i], accuracy[i], precision[i], recall[i], sensitivity[i], f1[i], \
                specificity[
                    i] = calculate_metrics(P[:, :], M[:, :])

                # Save images
                rgb = (np.transpose(data[0].detach().cpu() * 255.0, (1, 2, 0)))
                gt = mask.detach().cpu() * 255.0
                pred = output.detach().cpu() * 255.0

                # Save combined images
                cv2.imwrite(f"results/model_pre_result/{self.datasetype}_{self.net.name()}_Files/combined_{i}_1.jpeg",
                            rgb.numpy().astype(np.uint8))
                cv2.imwrite(f"results/model_pre_result/{self.datasetype}_{self.net.name()}_Files/combined_{i}_2.jpeg",
                            gt.numpy().astype(np.uint8))
                cv2.imwrite(f"results/model_pre_result/{self.datasetype}_{self.net.name()}_Files/combined_{i}_3.jpeg",
                            pred.numpy().astype(np.uint8))

                # Save F1 score to TXT file
                image_name = f"{i}.png"  # Adjust this as per your naming convention
                f.write(f"{image_name}: {f1[i]}\n")  # Write F1 score for each image

        # Calculate and print mean metrics
        mean_HD = np.mean(HD)
        mean_iou = np.mean(iou)
        mean_dice_coefficient = np.mean(dice_coefficient)
        mean_accuracy = np.mean(accuracy)
        mean_precision = np.mean(precision)
        mean_recall = np.mean(recall)
        mean_sensitivity = np.mean(sensitivity)
        mean_f1 = np.mean(f1)
        mean_specificity = np.mean(specificity)


        print(
            'HD: %.4f, iou: %.4f, Dice: %.4f, accuracy: %.4f, precision: %.4f, recall: %.4f, specificity: %.4f, f1: %.4f' % (
                mean_HD, mean_iou, mean_dice_coefficient, mean_accuracy, mean_precision, mean_recall, mean_specificity,
                mean_f1))
    def create_plots(self):
        create_dir('result')
        loss_name = f"./result/{self.datasetype}_{self.net.name()}_Loss_" + str(self.criterion) + '_' + str(self.num_epochs) + '_' + str(self.batch_size['train']) + '.csv'
        iou_name = f"./result/{self.datasetype}_{self.net.name()}_IOU_" + str(self.criterion) + '_' + str(self.num_epochs) + '_' + str(
            self.batch_size['train']) + '.csv'
        dice_name = f"./result/{self.datasetype}_{self.net.name()}_Dice_" + str(self.criterion) + '_' + str(self.num_epochs) + '_' + str(
            self.batch_size['train']) + '.csv'

        with open(loss_name, 'a', newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.losses['train'])
        with open(iou_name, 'a', newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.iou_scores['train'])
        with open(dice_name, 'a', newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.dice_scores['train'])

        plot(self.losses, f"{self.datasetype}_{self.net.name()} BCE loss")
        plot(self.dice_scores, f"{self.datasetype}_{self.net.name()} Dice score")
        plot(self.iou_scores, f"{self.datasetype}_{self.net.name()} IoU score")

    def soft_to_hard_pred(pred, channel_axis=1):
        max_value = np.max(pred, axis=channel_axis, keepdims=True)
        return np.where(pred == max_value, 1, 0)





if __name__ == "__main__":



    dataset_arr = ['cpm17','MoNuSeg', 'PanNuke', 'CoNIC', 'NuInsSeg']
    print("torch.cuda GPU:", torch.cuda.is_available())

    # Select dataset from dataset_arr
    Dataset_mode =0

    #'Train' :1 test 0
    Net_mode = 0

    model_name_arr = ['MLAANet']

    params = {'in_chns': 3,
              'class_num': 1,
              'feature_chns': [16, 32, 64, 128],
              'fr_feature_chns': 8,
              'bilinear': True,
              '_deep_supervision': True,
              'do_ds': True,
              'con_op': nn.Conv2d}

    for i in range(len(model_name_arr)):
        model_name = model_name_arr[i]
        seeding(42)
        if model_name == "MLAANet":
            model = MLAANet(3, 1).to(device)

        selected_dataset = dataset_arr[Dataset_mode]
        folder_name = selected_dataset + "_"+ model_name + "_Files"
        base_path = "results\\model_pre_result"
        model_path = "results\\models_Bestfile"
        folder_path = os.path.join(base_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        if Net_mode == 1:
            model_trainer = Trainer(model, dataset_arr[Dataset_mode])
            print("Name： %s , Epoch： %d, BatchSize: %s" %(model_name, model_trainer.num_epochs, model_trainer.batch_size))
            model_trainer.start()
            model_trainer.create_plots()
        elif Net_mode == 0:
            file_name =dataset_arr[Dataset_mode]+'_'+ model_name + '_DiceBCELoss()_3_4_20250610_1343_best_model.pth'
            state_dict = torch.load(os.path.join(model_path, file_name))
            model.load_state_dict(state_dict)
            model_trainer = Trainer(model, dataset_arr[Dataset_mode])
            model_trainer.OnlyTesting()