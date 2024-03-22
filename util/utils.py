import torch.nn as nn
import torch
import random
import numpy as np
import logging
from .consts import (
    # C_LIB_PATH,
    color_list,
    hatch_list,
    linestyle_list,
    markertype_list,
)
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as font_manager

import seaborn as sns
from textwrap import wrap

import struct

def plot_cdf_sub(
    raw_data,
    file_name,
    linelabels,
    x_label,
    y_label="CDF",
    log_switch=False,
    rotate_xaxis=False,
    ylim_low=0,
    xlim=None,
    xlim_bottom=None,
    fontsize=15,
    legend_font=15,
    loc=2,
    title=None,
    enable_abs=False,
    group_size=1,
    enable_save=False,
):
    _fontsize = fontsize
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    if log_switch:
        ax.set_xscale("log")

    plt.ylabel(y_label, fontsize=_fontsize)
    plt.xlabel(x_label, fontsize=_fontsize)
    linelabels = ["\n".join(wrap(l, 30)) for l in linelabels]
    for i in range(len(raw_data)):
        data = raw_data[i]
        data = data[~np.isnan(data)]
        if len(data) == 0:
            continue
        if enable_abs:
            data = abs(data)
        # data=random.sample(data,min(1e6,len(data)))
        data_size = len(data)
        # data=list(filter(lambda score: 0<=score < std_val, data))
        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)

        # Plot the cdf
        if i < len(linelabels):
            plt.plot(
                bin_edges[0:-1],
                cdf,
                linestyle=linestyle_list[(i // group_size) % len(linestyle_list)],
                color=color_list[(i % group_size) % len(color_list)],
                label=linelabels[i],
                linewidth=3,
            )
        else:
            plt.plot(
                bin_edges[0:-1],
                cdf,
                linestyle=linestyle_list[(i // group_size) % len(linestyle_list)],
                color=color_list[(i % group_size) % len(color_list)],
                linewidth=3,
            )

    legend_properties = {"size": legend_font}
    plt.legend(
        prop=legend_properties,
        frameon=False,
        loc=loc,
    )

    plt.ylim((ylim_low, 1))
    if xlim_bottom:
        plt.xlim(left=xlim_bottom)
    if xlim:
        plt.xlim(right=xlim)
    # plt.tight_layout()
    plt.tight_layout(pad=0.5, w_pad=0.04, h_pad=0.01)
    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)
    # plt.grid(True)
    if rotate_xaxis:
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    if title:
        plt.title(title, fontsize=_fontsize - 5)
    if enable_save:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0)


def plot_cdf(
    raw_data,
    file_name,
    linelabels,
    x_label,
    y_label="CDF",
    log_switch=False,
    rotate_xaxis=False,
    ylim_low=0,
    xlim=None,
    xlim_bottom=None,
    fontsize=15,
    legend_font=15,
    loc=2,
    title=None,
    enable_abs=False,
    group_size=1,
    enable_save=False,
    fig_idx=0,
):
    _fontsize = fontsize
    fig = plt.figure(fig_idx, figsize=(6, 4))  # 2.5 inch for 1/3 double column width
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    if log_switch:
        ax.set_xscale("log")

    plt.ylabel(y_label, fontsize=_fontsize)
    plt.xlabel(x_label, fontsize=_fontsize)
    linelabels = ["\n".join(wrap(l, 30)) for l in linelabels]
    for i in range(len(raw_data)):
        data = raw_data[i]
        data = data[~np.isnan(data)]
        if len(data) == 0:
            continue
        if enable_abs:
            data = abs(data)
        # data=random.sample(data,min(1e6,len(data)))
        data_size = len(data)
        # data=list(filter(lambda score: 0<=score < std_val, data))
        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)

        # Plot the cdf
        if i < len(linelabels):
            plt.plot(
                bin_edges[0:-1],
                cdf,
                linestyle=linestyle_list[(i // group_size) % len(linestyle_list)],
                color=color_list[(i % group_size) % len(color_list)],
                label=linelabels[i],
                linewidth=3,
            )
        else:
            plt.plot(
                bin_edges[0:-1],
                cdf,
                linestyle=linestyle_list[(i // group_size) % len(linestyle_list)],
                color=color_list[(i % group_size) % len(color_list)],
                linewidth=3,
            )

    legend_properties = {"size": legend_font}
    plt.legend(
        prop=legend_properties,
        frameon=False,
        loc=loc,
    )

    plt.ylim((ylim_low, 1))
    if xlim_bottom:
        plt.xlim(left=xlim_bottom)
    if xlim:
        plt.xlim(right=xlim)
    # plt.tight_layout()
    # plt.tight_layout(pad=0.5, w_pad=0.04, h_pad=0.01)
    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)
    # plt.grid(True)
    if rotate_xaxis:
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    if title:
        plt.title(title, fontsize=_fontsize - 5)
    if enable_save:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0)


def plot_map(
    bandwidth_arr,
    file_name="./figs/metric.pdf",
    xticklabels=None,
    yticklabels=None,
    enable_abs=False,
    title_str=None,
    enable_bar_limit=True,
    enable_color_bar=True,
    vmin=0,
    vmax=1,
    fontsize=40,
    legend_name="P Error",
    labelpad=-230,
    fontsize_x=45,
    rotation_x=30,
    fontsize_y=45,
    rotation_y=0,
    fontsize_title=50,
    enable_customize_title=False,
    fontsize_colorbar=50,
):
    if enable_bar_limit:
        if enable_abs:
            bandwidth_arr = abs(bandwidth_arr)
            vmin = vmin
            vmax = vmax
        else:
            vmin = -1
            vmax = 1
    else:
        vmin = None
        vmax = None
    # fig = plt.figure(figsize=(30, 30))  # 2.5 inch for 1/3 double column width
    # ax = fig.add_subplot(111)
    # num_site = len(site_list)
    # bandwidth_arr = np.zeros((num_site, num_site))
    # Loop through all files in the directory
    # Heatmap
    # p99 = copy.deepcopy(bandwidth_arr)
    # np.fill_diagonal(p99, 0)
    # print(np.nanmean(p99))
    # mask = np.triu(np.ones_like(bandwidth_arr, dtype=bool), 1)
    colormap = sns.color_palette("ch:s=.25,rot=-.25")
    ax = sns.heatmap(
        bandwidth_arr,
        # mask=mask,
        # square=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": fontsize},
        # cbar_kws={"label": "Network Bandwidth (Mibit/s)"},
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        # cmap=colormap,
        linewidth=0.5,
        vmin=vmin,
        vmax=vmax,
    )
    # for t in ax.texts:
    #     t.set_text(t.get_text() + " %")
    # ax = sns.heatmap(bandwidth_list, xticklabels=city_list, yticklabels=city_list)
    if enable_color_bar:
        cbar = ax.collections[0].colorbar
        cbar.set_label(legend_name, labelpad=labelpad)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.yaxis.label.set_size(fontsize_colorbar)
    # cbar.ax.yaxis.set_major_formatter(
    #     ticker.FuncFormatter(lambda x, pos: "{:,.0f}%".format(x * 100))
    # )
    plt.xticks(fontsize=fontsize_x, rotation=rotation_x)
    plt.yticks(fontsize=fontsize_y, rotation=rotation_y)

    # 3D
    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # num_bars = 10
    # x_pos = np.array([[i] * num_bars for i in range(num_bars)]).flatten() - 0.25
    # y_pos = np.array(list(range(num_bars)) * num_bars) - 0.25
    # # x_pos = np.arange(num_bars)
    # # y_pos = np.arange(num_bars)
    # z_pos = np.zeros(len(x_pos))
    # x_size = np.ones(len(x_pos)) * 0.5
    # y_size = np.ones(len(x_pos)) * 0.5
    # z_size = np.array(bandwidth_list).flatten()

    # cmap = cm.get_cmap('Blues') # Get desired colormap - you can change this!
    # # colormap = sns.color_palette("Reds", 28)
    # max_height = np.nanmax(z_size)   # get range of colorbars so we can normalize
    # min_height = np.nanmin(z_size)
    # rgba = [cmap((k-min_height)/max_height) for k in z_size]

    # ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size, color=rgba)

    # ticksx = np.arange(10)
    # ticksz = ax.get_zticks() / 1000
    # ax.set_zticklabels(ticksz.astype(int))
    # ax.set_zlabel('Network Bandwidth (Mibit/s)', fontsize=12)
    # ax.grid(False)
    # # ax.set_zticks(ticksz / 1000)

    # plt.xticks(ticksx, city_list, fontsize=12)
    # plt.yticks(ticksx, city_list, fontsize=12)

    if enable_customize_title:
        if enable_abs:
            val_min = np.nanmin(bandwidth_arr)
            val_median = np.nanmedian(bandwidth_arr)
            val_mean = np.nanmean(bandwidth_arr)
            val_max = np.nanmax(bandwidth_arr)

            plt.title(
                title_str
                + "\nmin-{:0.3f},median-{:0.3f}/mean-{:0.3f},max-{:0.3f}".format(
                    val_min, val_median, val_mean, val_max
                ),
                fontsize=fontsize_title,
            )
        else:
            val_min = np.nanmin(bandwidth_arr)
            val_max = np.nanmax(bandwidth_arr)
            plt.title(
                title_str + "\nmin-{:0.3f},max-{:0.3f}".format(val_min, val_max),
                fontsize=fontsize_title,
            )
    else:
        plt.title(title_str, fontsize=fontsize_title)
    # plt.tight_layout()
    # plt.savefig(file_name, bbox_inches="tight")


def plot_line(
    y_data,
    x_data,
    legend_list=None,
    x_label=None,
    y_label="CDF",
    name="ss",
    line_width=2,
    log_switch_y=False,
    log_switch_x=False,
    legend_frame=False,
    legend_font=15,
    loc=2,
    fontsize=15,
    ncol=1,
    xlim=None,
    ylim=None,
    ylim_bottom=None,
    xlim_bottom=None,
    xticklabel=None,
    h_list=None,
    v_list=None,
    markersize=10,
    match_unit=100,
    text_fontsize=25,
    text_x=0,
    text_y=0,
    text_y_limit=1,
):
    _fontsize = fontsize
    fig = plt.figure(figsize=(5.2, 4))  # 2.5 inch for 1/3 double column width
    ax = fig.add_subplot(111)

    plt.ylabel(y_label, fontsize=_fontsize)
    plt.xlabel(x_label, fontsize=_fontsize)

    for i, data in enumerate(y_data):
        if x_data == None:
            tmp_xs = [i for i in range(1, len(data) + 1)]
        elif len(y_data) == len(x_data):
            tmp_xs = x_data[i]
        else:
            tmp_xs = x_data[: min(len(x_data), len(data))]
        if len(legend_list) > i:
            plt.plot(
                tmp_xs,
                data,
                linestyle_list[i // match_unit % (len(linestyle_list) - 1)],
                color=color_list[i % match_unit % len(color_list)],
                x_label=legend_list[i % len(legend_list)],
                linewidth=line_width,
                marker=markertype_list[i % len(markertype_list)],
                markersize=markersize,
            )
        else:
            plt.plot(
                tmp_xs,
                data,
                linestyle_list[i // match_unit % (len(linestyle_list) - 1)],
                color=color_list[i % match_unit % len(color_list)],
                linewidth=line_width,
                marker=markertype_list[i % len(markertype_list)],
                markersize=markersize,
            )
    if h_list:
        for i, data in enumerate(h_list):
            plt.axhline(
                data,
                linestyle=linestyle_list[-1],
                color=color_list[i % len(color_list)],
                linewidth=line_width,
                x_label=legend_list[(i + len(y_data)) % len(legend_list)],
            )
    if v_list:
        plt.axvline(
            v_list[0],
            ymin=0,
            ymax=(v_list[1] + 1) / text_y_limit,
            linestyle=linestyle_list[-1],
            color="black",
            linewidth=line_width,
            x_label=None,
        )
        plt.text(text_x, text_y, v_list[2], fontsize=text_fontsize)
    legend_properties = {"size": legend_font}

    plt.legend(
        facecolor="gray",
        framealpha=0.5,
        loc=loc,
        ncol=ncol,
        prop=legend_properties,
        frameon=legend_frame,
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    if log_switch_y:
        ax.set_yscale("log")
    if log_switch_x:
        ax.set_xscale("log")
    if xticklabel:
        plt.xticks(
            # np.flip(np.arange(y_data.shape[1] - 1, 0, -4)),
            np.arange(0, 18, 4),
            xticklabel,
            fontsize=_fontsize,
        )
    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)
    if xlim_bottom:
        plt.xlim(left=xlim_bottom)
    if xlim:
        plt.xlim(right=xlim)
    if ylim_bottom:
        plt.ylim(bottom=ylim_bottom)
    if ylim:
        plt.ylim(top=ylim)

    plt.savefig(name, bbox_inches="tight")


# def make_array(ctype, arr):
#     return (ctype * len(arr))(*arr)


# class INPUTStruct(Structure):
#     _fields_ = [
#         ("size_tmp", c_uint),
#         ("event_times", POINTER(c_double)),
#         ("enq_indices", POINTER(c_uint)),
#         ("deq_indices", POINTER(c_uint)),
#         ("weight_scatter_indices", POINTER(c_uint)),
#         ("num_active_flows", POINTER(c_uint)),
#         ("active_flows", POINTER(c_uint)),
#     ]


# class FCTStruct(Structure):
#     _fields_ = [
#         ("estimated_fcts", POINTER(c_double)),
#         ("t_flows", POINTER(c_double)),
#         ("num_flows", POINTER(c_uint)),
#         ("num_flows_enq", POINTER(c_uint)),
#     ]

# C_LIB = CDLL(C_LIB_PATH)

# C_LIB.get_input = C_LIB.get_input
# C_LIB.get_input.argtypes = [
#     c_uint,
#     POINTER(c_double),
#     POINTER(c_double),
# ]
# C_LIB.get_input.restype = INPUTStruct

# C_LIB.get_fct = C_LIB.get_fct
# C_LIB.get_fct.argtypes = [
#     c_uint,
#     POINTER(c_double),
#     POINTER(c_double),
#     POINTER(c_double),
# ]
# C_LIB.get_fct.restype = FCTStruct

# C_LIB.get_fct_flowsim = C_LIB.get_fct_flowsim
# C_LIB.get_fct_flowsim.argtypes = [
#     c_uint,
#     POINTER(c_double),
#     POINTER(c_double),
# ]
# C_LIB.get_fct_flowsim.restype = FCTStruct

# C_LIB.get_fct_by_cwnd_perflow = C_LIB.get_fct_by_cwnd_perflow
# C_LIB.get_fct_by_cwnd_perflow.argtypes = [
#     c_uint,
#     POINTER(c_double),
#     POINTER(c_double),
#     c_double,
#     c_double,
# ]
# C_LIB.get_fct_by_cwnd_perflow.restype = FCTStruct

# C_LIB.get_fct_by_cwnd_sum = C_LIB.get_fct_by_cwnd_sum
# C_LIB.get_fct_by_cwnd_sum.argtypes = [
#     c_uint,
#     POINTER(c_double),
#     POINTER(c_double),
#     c_double,
#     c_double,
# ]
# C_LIB.get_fct_by_cwnd_sum.restype = FCTStruct

# C_LIB.free_fctstruct = C_LIB.free_fctstruct
# C_LIB.free_fctstruct.argtypes = [FCTStruct]
# C_LIB.free_fctstruct.restype = None

# C_LIB.free_inputstruct = C_LIB.free_inputstruct
# C_LIB.free_inputstruct.argtypes = [INPUTStruct]
# C_LIB.free_inputstruct.restype = None


def parse_output_get_input(res, n_flows):
    size_tmp = res.size_tmp
    event_times = np.fromiter(res.event_times, dtype=np.float64, count=2 * n_flows)
    enq_indices = np.fromiter(res.enq_indices, dtype=np.uint, count=n_flows).astype(
        np.int64
    )
    deq_indices = np.fromiter(res.deq_indices, dtype=np.uint, count=n_flows).astype(
        np.int64
    )
    num_active_flows = np.fromiter(
        res.num_active_flows, dtype=np.uint, count=(2 * n_flows - 1)
    ).astype(np.int64)
    weight_scatter_indices = np.fromiter(
        res.weight_scatter_indices, dtype=np.uint, count=size_tmp
    ).astype(np.int64)
    active_flows = np.fromiter(res.active_flows, dtype=np.uint, count=size_tmp).astype(
        np.int64
    )
    return (
        event_times,
        enq_indices,
        deq_indices,
        num_active_flows,
        weight_scatter_indices,
        active_flows,
    )


def create_logger(log_name):
    logging.basicConfig(
        # format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        # format="%(asctime)s|%(levelname)s| %(processName)s [%(filename)s:%(lineno)d] %(message)s",
        format="%(asctime)s|%(filename)s:%(lineno)d|%(message)s",
        # datefmt="%Y-%m-%d:%H:%M:%S",
        datefmt="%m-%d:%H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_name, mode="a"), logging.StreamHandler()],
    )
    for handler in logging.root.handlers:
        handler.addFilter(fileFilter())


class fileFilter(logging.Filter):
    def filter(self, record):
        # return (not record.getMessage().startswith("Added")) and (
        #     not record.getMessage().startswith("Rank ")
        # )
        return True


def fix_seed(seed):
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)


def decode_dict(d, encoding_used="utf-8"):
    return {
        k.decode(encoding_used): (
            v.decode(encoding_used) if isinstance(v, bytes) else v
        )
        for k, v in d.items()
    }
def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)