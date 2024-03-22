import argparse
import datetime, time


def create_config():
    parser = argparse.ArgumentParser(description="Run flow based network simulation")
    # basic config
    parser.add_argument(
        "--train_config",
        type=str,
        default="./config/train_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--test_config",
        type=str,
        default="./config/test_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--tune_config",
        type=str,
        default="./config/tune_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--note", type=str, default="debug")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--test_on_train", action="store_true")
    parser.add_argument("--test_on_empirical", action="store_true")
    parser.add_argument("--test_on_manual", action="store_true")
    parser.add_argument("--train_on_incast_only", action="store_true")
    parser.add_argument("--train_on_tc_only", action="store_true")
    parser.add_argument("--train_on_tc", action="store_true")
    parser.add_argument("--train_on_incast", action="store_true")
    # parser.add_argument(
    #     "--dataset_name",
    #     type=str,
    #     default=None,
    #     choices=["seg", "fairsharing", "sizebucket"],
    # )
    # parser.add_argument(
    #     "--model_name",
    #     type=str,
    #     default=None,
    #     choices=["queue", "fairsharing", "sizebucket"],
    # )
    parser.add_argument(
        "--version_id",
        type=int,
        default=0,
    )
    # dataset config
    args = parser.parse_args()
    args.timestamp = (
        datetime.datetime.fromtimestamp(time.time()).strftime("%m%d_%H%M%S")
        # + "_"
        # + str(random.randint(1, 60000))
    )
    return args


def create_config_dataset():
    parser = argparse.ArgumentParser(description="Run flow based network simulation")
    # dataset config
    parser.add_argument(
        "--dir_input",
        help="input directory",
        # default="/data2/Arash/simulation/linksim/genpar_size_lognormal_inter/1Mflows",
        # default="/data2/Arash/simulation/linksim/genpar_size_lognormal_inter",
    )
    parser.add_argument(
        "--dir_output",
        help="output directory",
        # default="/data1/lichenni/projects/flow_simulation/output",
        default="/data2/lichenni/parsimon",
    )
    parser.add_argument(
        "--cdf_file",
        help="cdf file",
        default="GoogleRPC2008.txt",
        choices=[
            "AliStorage2019.txt",
            "FbHdp_distribution.txt",
            "GoogleRPC2008.txt",
            "WebSearch_distribution.txt",
        ],
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="pareto_lognorm",
        # choices=["l1"],
        help="tag for dataset config",
    )
    parser.add_argument("--shard", type=int, default=0, help="random seed")
    parser.add_argument("--iasigma", type=float, default=2.0, help="burstiness")
    parser.add_argument("--dctcp_k_factor", type=float, default=3.0, help="burstiness")
    parser.add_argument(
        "--prop_delay_factor", type=float, default=1.0, help="burstiness"
    )
    parser.add_argument("--enable_auxi", action="store_true")
    parser.add_argument(
        "--avg_util", type=float, default=0.5, help="average utilization"
    )
    parser.add_argument(
        "--sizesigma",
        type=int,
        default=20000,
        # choices=[20000, 30000, 40000],
        help="size range",
    )
    parser.add_argument(
        "--lr", type=int, default=10, choices=[10], help="link rate in Gbps"
    )
    parser.add_argument("--n_flows", type=int, default=10000)
    parser.add_argument("--n_tcs", type=int, default=2)
    parser.add_argument(
        "--cwnd_perflow", type=float, default=0.5, help="average utilization"
    )
    parser.add_argument(
        "--cwnd_sum", type=float, default=0.5, help="average utilization"
    )
    parser.add_argument("--is_empirical", action="store_true")
    parser.add_argument("--is_incast", action="store_true")
    args = parser.parse_args()
    return args


def create_parser():
    parser = argparse.ArgumentParser(description="Run flow based network simulation")
    # basic config
    parser.add_argument(
        "--train_config",
        type=str,
        default="./config/train_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--seed", type=int, help="random_seed", default=0)
    parser.add_argument("--log_size", action="store_true")
    parser.add_argument("--norm_size", action="store_true")
    parser.add_argument("--sigmoid_size", action="store_true")
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--device", type=str, help="Compute device", default="cuda:1")
    parser.add_argument("--timestamp", type=str, default=None)
    # dataset config
    parser.add_argument(
        "--model",
        type=str,
        default="sldn",
        # choices=["l1"],
        help="tag for dataset config",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="pareto_lognorm",
        # choices=["l1"],
        help="tag for dataset config",
    )
    parser.add_argument(
        "--w_tag",
        type=str,
        default="pareto_lognorm",
        # choices=["l1"],
        help="tag for dataset config",
    )
    parser.add_argument(
        "--s_tag",
        type=str,
        default="pareto_lognorm",
        # choices=["l1"],
        help="tag for dataset config",
    )
    parser.add_argument(
        "--dir_parsimon",
        help="input directory",
        # default="/afs/csail.mit.edu/u/l/lichenni/data",
        default="/data2/lichenni/data",
    )
    parser.add_argument(
        "--dir_log",
        help="input directory",
        default="/data1/lichenni/projects/flow_simulation/log",
    )
    parser.add_argument(
        "--dir_par_log",
        help="input directory",
        default="/data1/lichenni/projects/linksim-data-gen/log",
    )
    parser.add_argument(
        "--dir_input",
        help="input directory",
        # default="/data2/Arash/simulation/linksim/genpar_size_lognormal_inter/1Mflows",
        # default="/data2/Arash/simulation/linksim/genpar_size_lognormal_inter",
    )
    parser.add_argument(
        "--dir_output",
        help="output directory",
        # default="/data1/lichenni/projects/flow_simulation/output",
        default="/data2/lichenni/output",
    )
    parser.add_argument(
        "--dir_lighting",
        help="output directory",
        # default="/data1/lichenni/projects/flow_simulation/output",
        default="/data2/lichenni/lighting",
    )
    parser.add_argument(
        "--cdf_file",
        help="cdf file",
        default="GoogleRPC2008.txt",
        choices=[
            "AliStorage2019.txt",
            "FbHdp_distribution.txt",
            "GoogleRPC2008.txt",
            "WebSearch_distribution.txt",
        ],
    )
    parser.add_argument("--shard", type=int, default=0, help="random seed")
    parser.add_argument("--iasigma", type=float, default=2.0, help="burstiness")
    parser.add_argument(
        "--avg_util", type=float, default=0.5, help="average utilization"
    )
    parser.add_argument(
        "--w_avg_util", type=float, default=0.5, help="average utilization"
    )
    parser.add_argument(
        "--s_avg_util", type=float, default=0.5, help="average utilization"
    )
    parser.add_argument(
        "--loss_mode",
        type=int,
        default=1,
        choices=[1],
        help="loss variants",
    )
    parser.add_argument(
        "--sizesigma",
        type=int,
        default=20000,
        # choices=[20000, 30000, 40000],
        help="size range?",
    )
    parser.add_argument(
        "--lr", type=int, default=10, choices=[10, 100], help="link rate in Gbps"
    )
    parser.add_argument("--gpu", nargs="+", type=int, default=[3])
    # ML training config
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="l1",
        choices=["mse", "l1"],
        # choices=["l1"],
        help="loss function type",
    )
    parser.add_argument("--tag_list_len", type=int, default=3)
    parser.add_argument("--avg_util_list_len", type=int, default=3)
    parser.add_argument("--iasigma_list", type=int, default=3)
    parser.add_argument("--sizesigma_list", type=int, default=3)
    parser.add_argument("--version_id", type=int, default=0)
    parser.add_argument("--enable_generalization", action="store_true")
    parser.add_argument("--enable_val", action="store_true")
    parser.add_argument("--clamp", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--n_epochs_warmup", type=int, default=0)
    parser.add_argument("--n_epochs_every_log", type=int, default=10)
    parser.add_argument("--n_epochs_every_output", type=int, default=100)
    parser.add_argument("--n_flows", type=int, default=10)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--loss_window", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--w_shard", type=int, default=0, help="weight shard")
    parser.add_argument(
        "--w_iasigma", type=float, default=2.0, help="weight inter-arrival sigma"
    )
    parser.add_argument(
        "--w_sizesigma", type=int, default=20000, help="weight size-sigma"
    )
    parser.add_argument("--s_shard", type=int, default=0, help="simulation shard")
    parser.add_argument(
        "--s_iasigma", type=float, default=2.0, help="simulation inter-arrival sigma"
    )
    parser.add_argument(
        "--s_sizesigma", type=int, default=20000, help="simulation size-sigma"
    )
    parser.add_argument(
        "--w_lr", type=int, default=10, help="weights link rate in Gbps"
    )
    parser.add_argument(
        "--s_lr", type=int, default=10, help="simulation link rate in Gbps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="ML model learning rate"
    )
    args = parser.parse_args()
    # args.timestamp = (
    #     datetime.datetime.fromtimestamp(time.time()).strftime("%m%d_%H%M%S")
    #     # + "_"
    #     # + str(random.randint(1, 60000))
    # )
    # args.shard = args.seed
    # args.w_shard=args.shard
    # args.w_iasigma=args.iasigma
    # args.w_sizesigma=args.sizesigma
    # args.s_shard=args.shard
    # args.s_iasigma=args.iasigma
    # args.s_sizesigma=args.sizesigma
    return args


if __name__ == "__main__":
    args = create_config()
    print(args)
