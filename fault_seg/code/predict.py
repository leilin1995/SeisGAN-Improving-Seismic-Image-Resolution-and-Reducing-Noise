"""
__author__ = 'linlei'
__project__:predict
__time__:2021/9/28 10:44
__email__:"919711601@qq.com"
"""
import argparse
import os
import torch
from model import Unet
from utils import read_h5, save_h5
import numpy as np


def main(args):
    # define device
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    # build model
    model = Unet(in_ch=args.in_channel, out_ch=args.out_channel).to(device)
    path = args.state_path

    # load state dict
    loader = torch.load(path, map_location=device)
    model.load_state_dict(loader["model_state_dict"])

    model.eval()
    # prediction
    with torch.no_grad():
        # read data and add channel dim
        data=read_h5(args.input_path).T # transpose data to :(samples,traces)
        data=normal(data)   # normalize data to [0,1]
        data = np.expand_dims(data,axis=0)
        # to tensor
        data = torch.from_numpy(data).type(torch.float).to(device)
        # expand batch dim
        data = torch.unsqueeze(data, dim=0)
        pred = model(data)
        # squeeze batch dim
        pred = torch.squeeze(pred, dim=0).cpu().numpy()
        if args.in_channel == 1:
            pred = np.squeeze(pred, axis=0)
        # save data
        save_h5(data=pred.T, path=args.save_path)

def normal(data):
    d_max=np.max(data)
    d_min=np.min(data)
    return (data-d_min)/(d_max-d_min)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="../sr/pred_fault.h5",
                        help="Save the predicted file")
    parser.add_argument("--state_path", type=str, default="./best_model.pth",
                        help="The path of the well-trained model")
    parser.add_argument("--input_path", type=str, default="../sr/k3_crossline_401_240×400.h5",
                        help="The path of the data to be predicted")
    parser.add_argument("--device", type=int, default=0, help="Gpu id of used eg. 0,1,2")
    parser.add_argument("--in_channel", type=int, default=1, help="The channel number of input image")
    parser.add_argument("--out_channel", type=int, default=1, help="The channel number of input image")
    args = parser.parse_args()
    # run
    main(args)
