import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # font type
mpl.rcParams['axes.unicode_minus'] = False


"read .hd5 file and convert to ndarray"
def read_h5(path):
    f = h5py.File(path, "r")
    data = f["/data"]
    return np.array(data)

def normal(data):
    data_max = np.max(data)
    data_min = np.min(data)
    return (data - data_min) / (data_max - data_min)

def show_result():
    # 1.load data
    seismic_sr = read_h5("./sr/k3_crossline_401_240×400.h5").T
    fault_sr = read_h5("./sr/pred_fault.h5").T
    sr_alpha = np.ones_like(fault_sr)
    sr_alpha[fault_sr < 0.55] = 0.5
    sr_alpha[fault_sr <= 0.45] = 0.25
    sr_alpha[fault_sr <= 0.3] = 0.
    seismic_raw = read_h5("./org/k3_crossline_401_240×400.h5").T
    fault_raw = read_h5("./org/pred_fault.h5").T
    raw_alpha = np.ones_like(fault_raw)
    raw_alpha[fault_raw < 0.55] = 0.5
    raw_alpha[fault_raw <= 0.45] = 0.25
    raw_alpha[fault_raw <= 0.3] = 0.
    seismic_bic = read_h5("./bicubic/k3_crossline_401_240×400.h5").T
    fault_bic = read_h5("./bicubic/pred_fault.h5").T
    bic_alpha = np.ones_like(fault_bic)
    bic_alpha[fault_bic < 0.55] = 0.5
    bic_alpha[fault_bic <= 0.45] = 0.25
    bic_alpha[fault_bic <= 0.3] = 0.
    # normal seismic
    seismic_sr=normal(seismic_sr)
    seismic_raw=normal(seismic_raw)
    seismic_bic=normal(seismic_bic)

    top_left_x = 10
    top_left_y = 250
    height = 100
    width = 150
    edgecolor = "blue"
    linewidth = 1
    shape_x, shape_y = seismic_raw.shape
    # define rectangle

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
    rect = plt.Rectangle((top_left_x, top_left_y), width, height, fill=False, edgecolor=edgecolor,
                         linewidth=linewidth)
    rect_up_bic = plt.Rectangle((top_left_x * 2, top_left_y * 2), width * 2, height * 2, fill=False,
                                edgecolor=edgecolor, linewidth=linewidth)
    rect_up_gan = plt.Rectangle((top_left_x * 2, top_left_y * 2), width * 2, height * 2, fill=False,
                                edgecolor=edgecolor,
                                linewidth=linewidth)
    font = {'family': 'Times New Roman',
            'color': 'black',
            'weight': 'normal',
            'size': 16,
            }

    font1 = {'family': 'Times New Roman',
             'color': 'black',
             'weight': 'normal',
             'size': 16,
             }

    #### complete figure
    # original
    text_x = -16
    text_y = -16
    # aspect:调整图片的纵横比例,"auto"为自动模式，“1”为1:1
    ax[0][0].imshow(seismic_raw, cmap=plt.cm.gray, aspect="auto", vmin=0, vmax=1)
    ax[0][0].imshow(fault_raw, cmap="jet", aspect="auto", vmin=0.15, vmax=0.85, alpha=raw_alpha)
    ax[0][0].add_patch(rect)
    ax[0][0].text(text_x, text_y, "(a)", font1)
    # ax[0][0].tick_params(labelsize=8)
    # ax[0][0].set_title("Original seismic image", font)
    ax[0][0].set_ylabel("Samples", font)

    # bicubic
    ax[0][1].imshow(seismic_bic, cmap=plt.cm.gray, aspect="auto", vmin=0, vmax=1)
    ax[0][1].imshow(fault_bic, cmap="jet", aspect="auto", vmin=0.15, vmax=0.85, alpha=bic_alpha)
    ax[0][1].add_patch(rect_up_bic)
    ax[0][1].text(text_x * 2, text_y * 2, "(b)", font1)
    # ax[0][1].set_title("Reconstruct by bicubic interpolation", font)
    # GAN
    ax[0][2].imshow(seismic_sr, cmap=plt.cm.gray, aspect="auto", vmin=0, vmax=1)
    ax[0][2].imshow(fault_sr, cmap="jet", aspect="auto", vmin=0.15, vmax=0.85, alpha=sr_alpha)
    # print(target.shape)
    ax[0][2].add_patch(rect_up_gan)
    ax[0][2].text(text_x * 2, text_y * 2, "(c)", font1)
    # ax[0][2].set_title("Reconstruct by our GAN", font)

    #### subpatch
    # original
    original_seis_sub = seismic_raw[top_left_y:top_left_y + height, top_left_x:top_left_x + width]
    original_fault_sub = fault_raw[top_left_y:top_left_y + height, top_left_x:top_left_x + width]
    original_fault_alpha=raw_alpha[top_left_y:top_left_y + height, top_left_x:top_left_x + width]
    ax[1][0].imshow(original_seis_sub,  cmap=plt.cm.gray, aspect="auto", vmin=0, vmax=1)
    ax[1][0].imshow(original_fault_sub, cmap="jet", aspect="auto", vmin=0.15, vmax=0.85, alpha=original_fault_alpha)
    ax[1][0].spines['bottom'].set_color(edgecolor)
    ax[1][0].spines['top'].set_color(edgecolor)
    ax[1][0].spines['left'].set_color(edgecolor)
    ax[1][0].spines['right'].set_color(edgecolor)
    ax[1][0].text(text_y * width / shape_y, text_x * height / shape_x, "(d)", font1)
    ax[1][0].set_ylabel("Samples", font)
    ax[1][0].set_xlabel("Traces", font)

    # bicubic
    bic_seis_sub = seismic_bic[top_left_y * 2:top_left_y * 2+ height * 2, top_left_x * 2:top_left_x * 2+ width * 2]
    bic_fault_sub = fault_bic[top_left_y * 2:top_left_y * 2+ height * 2, top_left_x * 2:top_left_x * 2+ width * 2]
    bic_fault_alpha = bic_alpha[top_left_y * 2:top_left_y * 2+ height * 2, top_left_x * 2:top_left_x * 2+ width * 2]
    ax[1][1].imshow(bic_seis_sub, cmap=plt.cm.gray, aspect="auto", vmin=0, vmax=1)
    ax[1][1].imshow(bic_fault_sub, cmap="jet", aspect="auto", vmin=0.15, vmax=0.85, alpha=bic_fault_alpha)
    ax[1][1].spines['bottom'].set_color(edgecolor)
    ax[1][1].spines['top'].set_color(edgecolor)
    ax[1][1].spines['left'].set_color(edgecolor)
    ax[1][1].spines['right'].set_color(edgecolor)
    ax[1][1].text(text_y * width / shape_y * 2, text_x * height / shape_x * 2, "(e)", font1)
    ax[1][1].set_xlabel("Traces", font)

    # GAN
    sr_seis_sub = seismic_sr[top_left_y * 2:top_left_y * 2+ height * 2, top_left_x * 2:top_left_x * 2+ width * 2]
    sr_fault_sub = fault_sr[top_left_y * 2:top_left_y * 2+ height * 2, top_left_x * 2:top_left_x * 2+ width * 2]
    sr_fault_alpha = sr_alpha[top_left_y * 2:top_left_y * 2+ height * 2, top_left_x * 2:top_left_x * 2+ width * 2]
    ax[1][2].imshow(sr_seis_sub, cmap=plt.cm.gray, aspect="auto", vmin=0, vmax=1)
    ax[1][2].imshow(sr_fault_sub, cmap="jet", aspect="auto", vmin=0.15, vmax=0.85, alpha=sr_fault_alpha)
    ax[1][2].spines['bottom'].set_color(edgecolor)
    ax[1][2].spines['top'].set_color(edgecolor)
    ax[1][2].spines['left'].set_color(edgecolor)
    ax[1][2].spines['right'].set_color(edgecolor)
    ax[1][2].text(text_y * width / shape_y * 2, text_x * height / shape_x * 2, "(f)", font1)
    ax[1][2].set_xlabel("Traces", font)
    plt.savefig("./fault_seg.png",dpi=300)

def show_colorbar():
    fig=plt.figure(figsize=(12,1))
    gs = GridSpec(4,60)
    # define colorbar
    font1 = {'family': 'Times New Roman',
             'color': 'black',
             'weight': 'normal',
             'size': 16,
             }
    dx1 = fig.add_subplot(gs[1:2, 2:28])
    norm1 = mpl.colors.Normalize(vmin=0,vmax=1)
    dbar1 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm1, cmap=plt.cm.gray),
                         # ticks=[-7.5, -5, -2.5, 0, 2.5, 5, 7.5],
                         cax=dx1,
                         orientation="horizontal")
    dbar1.set_label("Normalized Amplitude", loc='center', **font1)

    dx2 = fig.add_subplot(gs[1:2, 32:58])
    norm2 = mpl.colors.Normalize(vmin=0.15, vmax=0.85)
    dbar2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm2,cmap="jet"),
                         ticks=[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
                         cax=dx2,
                         orientation="horizontal")
    dbar2.set_label("Fault probability", loc='center', **font1)
    plt.savefig("./colorbar.png",dpi=300,bbox_inches="tight")


if __name__ == "__main__":
    # show_result()
    show_colorbar()
