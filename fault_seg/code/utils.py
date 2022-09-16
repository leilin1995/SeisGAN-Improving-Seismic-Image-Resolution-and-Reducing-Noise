"""
__author__ = 'linlei'
__project__:utils
__time__:2021/9/28 10:39
__email__:"919711601@qq.com"
"""


import torch
import torch.nn as nn
import numpy as np
import logging
import h5py


# balanced cross_entropy_loss
class BBCE(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,pred,label):
    pred = pred.view(pred.shape[0],-1)
    label = label.view(label.shape[0],-1)

    loss = 0.0
    beta = 0.95
    for i in range(pred.shape[0]):
      beta = 1 - torch.sum(label[i]) / label.shape[1]
      x = torch.log(pred[i])
      y = torch.log(1 - pred[i])
      l = -(beta * label[i] * x + (1 - beta) * (1 - label[i]) * y)
      loss += torch.mean(l)
    loss = loss / (label.shape[0])
    return loss

# Dice loss
class Dice_loss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,preds,labels):
    preds = preds.view(preds.shape[0],-1)
    labels = labels.view(labels.shape[0],-1)
    # 平滑变量
    smooth = 1000
    # 计算交集
    intersection = preds * labels
    N_dice_eff = (2 * intersection.sum(1) + smooth) / (preds.sum(1) + labels.sum(1) + smooth)
    # 计算每张图的损失
    loss = 1 - N_dice_eff.sum() / preds.shape[0]
    return loss

# Focal loss
# 学习置信度较低的样本，alpha 控制正负不平衡的样本，gamma控制执行度较低样本的学习
class FocalLoss(nn.Module):
    def __init__(self,alpha = 0.75,gamma = 2,size_average = True):
        """

        Args:
            alpha:Control positive and negative unbalanced samples weights
            gamma:Control the learning of samples with low execution
            size_average:average or not
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average


    def forward(self,preds,labels):
        preds = preds.view(preds.shape[0] ,-1)
        labels = labels.view(labels.shape[0] ,-1)
        # focal loss
        loss = - (self.alpha * ((1 - preds) ** self.gamma) * labels * torch.log(preds+10e-8) + (1 - self.alpha) * (preds ** self.gamma) * (1 - labels) * torch.log(1 - preds+10e-8))
        if self.size_average:
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss)
        return loss


def read_h5(path):
    """
    read .hd5 file and convert to ndarray
    Args:
        path: data path

    Returns:np array

    """
    f = h5py.File(path,"r")
    data = f["/data"]
    return np.array(data)

def save_h5(data,path):
    """
    save ndarray to .hd5 file
    Args:
        data: np array
        path: save path

    Returns:None

    """
    f = h5py.File(path,"w")
    f.create_dataset("/data",data = data)
    f.close()


class ConfusionMatrix():
    def __init__(self,num_classes = 2,threshold = 0.5):
        """
        caulate confusion matrix
        Args:
            num_classes: category,there are two types of fault recognition by default.
            threshold:threshold of fault segmentation,fault > threshold,not fault < threshold
        """

        self.matrix = np.zeros((num_classes,num_classes))
        self.num_classes = num_classes
        self.threshold = threshold

    def update(self,preds,labels):
        """

        Args:
            preds:Fault prediction probability map
            labels:Fault label map

        Returns:

        """
        preds[preds > self.threshold] = 1
        preds[preds <= self.threshold] = 0
        labels = labels.flatten()
        preds = preds.flatten()
        for p,t in zip(preds,labels):
          self.matrix[int(t),int(p)] += 1

    def summary(self):
        # ACC=Accuracy，PR=Precision，RE=Recall
        sum_TP = 0
        for i in range(self.num_classes):
          sum_TP += self.matrix[i,i]

        ACC = sum_TP / np.sum(self.matrix)
        TP = self.matrix[1,1]
        FP = self.matrix[0,1]
        return TP,FP


if __name__ == "__main__":
  loss = FocalLoss()
  a = torch.zeros((10,1,100,100))
  b = torch.rand((10,1,100,100)) / 5
  l = loss(b,a)
  print(l)
