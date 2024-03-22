import numpy as np

UNIT_G = 1000000000
UNIT_M = 1000000
UNIT_K = 1000

LINK_RATE = 10 * UNIT_G
SRC2BTL = 3 * UNIT_K  # 3us
BTL2DST = 1 * UNIT_K  # 1us
DELAY_PROPAGATION = SRC2BTL + BTL2DST
DELAY_PROPAGATION_BASE = 1000  # 1us

MTU = 1000
BDP = 10 * MTU

BDP_DICT = {
    3: 5 * MTU,
    5: 10 * MTU,
    7: 15 * MTU,
}
# LINK_TO_DELAY_DICT={
#     3:np.array([0,2*DELAY_PROPAGATION_BASE,0]),
#     5:np.array([0,2*DELAY_PROPAGATION_BASE,2*DELAY_PROPAGATION_BASE,2*DELAY_PROPAGATION_BASE,0]),
#     7:np.array([0,2*DELAY_PROPAGATION_BASE,2*DELAY_PROPAGATION_BASE,2*DELAY_PROPAGATION_BASE,2*DELAY_PROPAGATION_BASE,2*DELAY_PROPAGATION_BASE,0]),
# }
LINK_TO_DELAY_DICT={
    3:np.array([0,0,0]),
    5:np.array([0,0,1*DELAY_PROPAGATION_BASE,0,0]),
    7:np.array([0,0,1*DELAY_PROPAGATION_BASE,2*DELAY_PROPAGATION_BASE,1*DELAY_PROPAGATION_BASE,0,0]),
}
HEADER_SIZE = 48
BYTE_TO_BIT = 8
BDP_IN_BYTE = BDP + np.ceil(BDP / MTU) * HEADER_SIZE
BDP_IN_BIT = (BDP + np.ceil(BDP / MTU) * HEADER_SIZE) * BYTE_TO_BIT
MTU_IN_BYTE = MTU + np.ceil(MTU / MTU) * HEADER_SIZE
MTU_IN_BIT = (MTU + np.ceil(MTU / MTU) * HEADER_SIZE) * BYTE_TO_BIT
EPS = 1e-12
# SIZE_BUCKET_LIST = (80 * (1.5 ** np.arange(19))).astype(np.int32)
# SIZE_BUCKET_LIST = np.array([MTU, BDP, 2 * BDP, 5 * BDP, 10 * BDP, 20 * BDP, 50 * BDP])
# P99_DIVIDENT=[np.arange(0,1),np.arange(1,2),np.arange(2,4),np.arange(4,8)]

# P99_DIVIDENT=[np.arange(0,1),np.arange(1,2),np.arange(2,3),np.arange(3,4)]
# SIZE_BUCKET_LIST = np.array([MTU,BDP,5 * BDP])

# SIZE_BUCKET_LIST = np.array(
#     [MTU // 4, MTU // 2, MTU, 2 * MTU, 5 * MTU, BDP, 2 * BDP, 3 * BDP, 5 * BDP]
# )
# SIZE_BUCKET_LIST_LABEL = ["(0, 0.25MTU)","(0.25MTU, 0.5MTU)","(0.5MTU, MTU)","(MTU, 2MTU)","(2MTU, 5MTU)","(5MTU, BDP)","(BDP, 2BDP)","(2BDP, 3BDP)","(3BDP, 5BDP)","(5BDP, INF)"]
SIZE_BUCKET_LIST = np.array(
    [
        MTU // 4,
        MTU // 2,
        MTU * 3 // 4,
        MTU,
        BDP // 5,
        BDP // 2,
        BDP * 3 // 4,
        BDP,
        5 * BDP,
    ]
)
P99_DIVIDENT = [np.arange(0, 4), np.arange(4, 8), np.arange(8, 9), np.arange(9, 10)]
# P99_DIVIDENT = [np.arange(0, 2), np.arange(2, 4), np.arange(4, 6), np.arange(6, 8)]

def get_size_bucket_list(mtu, bdp):
    return np.array(
        [
            mtu // 4,
            mtu // 2,
            mtu * 3 // 4,
            mtu,
            bdp // 5,
            bdp // 2,
            bdp * 3 // 4,
            bdp,
            5 * bdp,
        ]
    )

def get_size_bucket_list_output(mtu, bdp):
    return np.array([mtu, bdp, 5 * bdp])

def get_base_delay(sizes, n_links_passed, lr):
    pkt_head = np.clip(sizes, a_min=0, a_max=MTU)
    return (
        DELAY_PROPAGATION_BASE * 2 * n_links_passed
        + (pkt_head) * BYTE_TO_BIT / lr * n_links_passed
    )

def get_base_delay_pmn(sizes, n_links_passed, lr_bottleneck,flow_idx_target,flow_idx_nontarget_internal):
    pkt_head = np.clip(sizes, a_min=0, a_max=MTU)
    delay_propagation = DELAY_PROPAGATION_BASE * n_links_passed
    pkt_size=(pkt_head + HEADER_SIZE) * BYTE_TO_BIT
    delay_transmission = np.multiply(pkt_size / lr_bottleneck,flow_idx_target) + pkt_size / (lr_bottleneck*4)*(n_links_passed-2)-np.multiply(pkt_size / lr_bottleneck,flow_idx_nontarget_internal)

    return delay_propagation + delay_transmission

SIZE_BUCKET_LIST_LABEL = [
    "(0, 0.25MTU)",
    "(0.25MTU, 0.5MTU)",
    "(0.5MTU, 0.75MTU)",
    "(0.75MTU, MTU)",
    "(MTU, 0.2BDP)",
    "(0.2BD, 0.5BDP)",
    "(0.5BDP, 0.75BDP)",
    "(0.75BDP, BDP)",
    "(BDP, 5BDP)",
    "(5BDP, INF)",
]

SIZE_BUCKET_LIST_LABEL_OUTPUT = ["(0, MTU)", "(MTU, BDP)", "(BDP, 5BDP)", "(5BDP, INF)"]

# P99_PERCENTILE_LIST = np.array(
#     [30, 50, 60, 70, 75, 80, 82, 84, 86,88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
# )
# P99_PERCENTILE_LIST = np.array(
#     [10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
# )
# P99_PERCENTILE_LIST = np.array(
#     [1, 25, 40, 55, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 100]
# )

P99_PERCENTILE_LIST = np.arange(1, 101, 1)
# P99_PERCENTILE_LIST = np.arange(1, 102, 1)
# P99_PERCENTILE_LIST[-1]=100

# P99_PERCENTILE_LIST = np.array(
#     [0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 98.2, 98.4, 98.6, 98.8, 99, 99.2,99.4, 99.6, 99.8, 100, 100]
# )

# P99_PERCENTILE_LIST = np.array(
#     [1, 10, 20, 30, 40, 50, 55, 60, 65, 70, 
#      75, 80, 85, 90, 92, 94, 96, 98, 98.2, 98.4, 
#      98.6, 98.8, 99, 99.2, 99.4, 99.6, 99.8, 100, 100]
# )
PERCENTILE_METHOD='nearest'

color_list = [
    "cornflowerblue",
    "orange",
    "deeppink",
    "blueviolet",
    "seagreen",
    "black",
]
# color_list = [
#     "C1",
#     "C2",
#     "C3",
#     "C4",
#     "C5",
#     "C6",
#     "C7",
# ]
hatch_list = ["o", "x", "/", ".", "*", "-", "\\"]
linestyle_list = ["solid", "dashed", "dashdot", "dotted"]
markertype_list = ["o", "^", "x", "x", "|"]

# FLOWSIZE_BUCKET_DICT = {
#     0: "0<size<=MTU",
#     1: "MTU<size<=BDP",
#     2: "BDP<size<=10BDP",
#     3: "10BDP<size",
# }
# FLOWSIZE_BUCKET_DICT = {
#     0: "(0,MTU]",
#     1: "(MTU,BDP]",
#     2: "(BDP,5BDP]",
#     3: "(5BDP,inf)",
# }
# SIZE_BIN_LIST = [MTU, BDP, 5 * BDP]

SIZEDIST_LIST_EMPIRICAL = [
    "GoogleRPC2008",
    "AliStorage2019",
    "FbHdp_distribution",
    "WebSearch_distribution",
]
UTIL_LIST = [
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    # 0.85,
    # 0.9,
]
IAS_LIST = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]