import argparse
import os
import time
from math import log10
import pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor,ToPILImage
from model import Generator
from utils import read_h5,save_h5,frequency_distance,ssim,cal_psnr,normal



def main(opt):
    UPSCALE_FACTOR = opt.upscale_factor
    TEST_DATA_PATH = opt.test_data_path
    # TEST_TARGET_PATH = opt.test_target_path
    MODEL_PATH = opt.model_path
    SAVE_PATH = opt.save_path
    if not os.path.exists(SAVE_PATH + "/predicted/"):
        os.makedirs(SAVE_PATH + "/predicted/")
    model = Generator(scale_factor = UPSCALE_FACTOR).eval()
    model.load_state_dict(torch.load(MODEL_PATH,map_location = "cuda:0"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    file_list = [file for file in os.listdir(TEST_DATA_PATH) if ".h5" in file]
    to_tensor = ToTensor()

    for file in file_list:
        data = read_h5(os.path.join(TEST_DATA_PATH,file))
        data_max = np.max(data)
        data_min = np.min(data)
        data_normal = normal(data)
        data_normal = to_tensor(data_normal).type(torch.FloatTensor)
        input = torch.unsqueeze(data_normal,dim = 0)
        input = input.to(device)
        out = model(input).detach().cpu()
        out_renormal = (data_max - data_min) * out + data_min
        out_renormal = np.squeeze(out_renormal.numpy(),axis = (0,1))
        save_h5(out_renormal,SAVE_PATH + "/predicted/" + file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "test the model performance")
    parser.add_argument("--upscale_factor",default = 2,type = int,help = "super resolution upscale factor")
    parser.add_argument("--test_data_path",default = r"../application/k3",type = str,help = "path of test low seismic images with noise")
    # parser.add_argument("--test_target_path",default = "../data/data/SRF_2/test/high",type = str,help = "path of test high seismic images without noise")
    parser.add_argument("--model_path",default = r"../result/SRF_2/model/netG_bestmodel.pth",type = str,help = "pre_trained model use for test")
    parser.add_argument("--save_path",default = r"../application/k3",type = str,help = "save the test result")
    opt = parser.parse_args()
    main(opt)


