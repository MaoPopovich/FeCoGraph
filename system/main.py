#!/usr/bin/env python
import copy
import torch
import torch.nn as nn
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
from datetime import datetime

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverlocal import Local
from flcore.servers.serverditto import Ditto
from flcore.servers.serverprox import FedProx

from flcore.models.grace import GRACE
from flcore.models.gcn import GCN_Net

from utils.general_utils import average_data, setup_seed, init_logger, logger, parse_param_json
from utils.mem_utils import MemReporter


def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model



    for i in range(args.prev, args.times):
        logger.info(f"============= Running time: {i}th =============")
        logger.info("Creating server and clients ...")
        start = time.time()

        # Initialize model class
        if model_str == "gcn": # convex
            if args.dataset in ['nf2018v2', 'nfbotv2', 'nftonv2']:
                args.model = GCN_Net(in_channels=39, out_channels=args.num_classes).to(args.device)
        
        elif model_str == "grace":
            # parse hyper parameters w.r.t contrastive learning
            params = parse_param_json('params.json')

            if args.dataset in ['nf2018v2', 'nfbotv2', 'nftonv2']:
                args.model = GRACE(in_channels=39, out_channels=args.num_classes, num_hidden = 32, num_proj_hidden = 32, tau=0.2).to(args.device)
        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)
        elif args.algorithm == "FedProx":
            server = FedProx(args, i)
        else:
            raise NotImplementedError
        server.train()

        time_list.append(time.time()-start)

    logger.info("Average time cost:{:.2f}s.".format(round(np.average(time_list), 2)))
    

    # Global average
    average_data(args)

    logger.info("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--seed', type=int, default=0,help="random seed")
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="1")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=1)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", action="store_true", default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-par',"--partition", type=str, default="louvain")
    parser.add_argument('--dir_alpha', type=float, default=0.5)
    parser.add_argument('-out', "--outdir", type=str, default='exp', help="the dir used to save log")
    parser.add_argument('-exp', "--expname", type=str, default='', help="detailed exp name to distinguish different sub-exp")
    parser.add_argument('-tag', "--expname_tag", type=str, default='', help="detailed exp tag to distinguish different sub-exp with the same expname")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=2,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.1,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # SCAFFOLD
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # contrastive learning
    parser.add_argument('--supcon_enabled', action='store_true', default=False,
                        help="Whether to enable graph contrastive learning")
    parser.add_argument('--semi_enabled', action='store_true', default=False,
                        help="Whether to enable G3NN semi-supervised learning")
    parser.add_argument('--param', type=str, default='params.json')
    parser.add_argument('-train', '--train_ratio', type=float, default=0.3,
                        help="the ratio of train/test split.")
    parser.add_argument('--drop_feature_rate', type=float, default=0.1,)
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.1,)
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.1,)
    # intrusion detection
    parser.add_argument('--task_type', type=str, default='multi')
    

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("=" * 50)
    print("drop edge feature rate: {}".format(args.drop_feature_rate))
    print("drop edge rate 1: {}".format(args.drop_edge_rate_1))
    print("drop edge rate 2: {}".format(args.drop_edge_rate_2))

    init_logger(args)
    # logger.info("I am back")
    setup_seed(args.seed)

    run(args)
