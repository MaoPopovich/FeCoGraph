import h5py
import numpy as np
import os
import json
import torch
import pickle
import random
import logging
from datetime import datetime

logger = logging.getLogger()

def average_data(args):
    test_acc = get_all_results_for_one_algo(args)
    times = args.times
    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    logger.info("std for best accurancy:" + str(np.std(max_accurancy)))
    logger.info("mean for best accurancy:" + str(np.mean(max_accurancy)))


def get_all_results_for_one_algo(args):
    times = args.times
    test_acc = []
    algorithms_list = [args.algorithm] * times
    for i in range(times):
        file_name = args.dataset + "_" + algorithms_list[i] + "_" + args.goal + "_lr" + str(args.local_learning_rate) + "_rs" + str(args.global_rounds) + "_ls" + str(args.local_epochs)
        test_acc.append(np.array(read_data_then_delete(args, file_name, delete=False)))

    return test_acc


def read_data_then_delete(args, file_name, delete=False):
    file_path = args.outdir + '/' + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc

def read_client_data(dataset, partition, idx, train_ratio, alpha=0.5):
    if partition == 'hetero':
        data_dir = os.path.join('../dataset', dataset, 'subgraph', partition+str(train_ratio), str(alpha))
    else:
        data_dir = os.path.join('../dataset', dataset, 'subgraph', partition+str(train_ratio))
    data_file = data_dir + '/partition_' + str(idx) + '.pt'
    with open(data_file, 'rb') as f:
        data = torch.load(f)
    return data['client_data']

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def init_logger(args, log_file_level=logging.NOTSET):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger_format = logging.Formatter(
        "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # set console handler
    # console = logging.StreamHandler()
    # console.setFormatter(logger_format)
    # logger.addHandler(console)

    # ================ create outdir to save log, exp_config, models, etc,.
    if args.outdir == "":
        args.outdir = os.path.join(os.getcwd(), "exp")
    if args.expname == "":
        args.expname = f"{args.algorithm}_{args.model}_on" \
                      f"_{args.dataset}_lr{args.local_learning_rate}_lste" \
                      f"p{args.local_epochs}_train{args.train_ratio}"
    if args.expname_tag:
        args.expname = f"{args.expname}/{args.expname_tag}"
    args.outdir = os.path.join(args.outdir, args.expname)
    print(args.outdir)

    # if exist, make directory with given name and time
    if os.path.isdir(args.outdir) and os.path.exists(args.outdir):
        outdir = os.path.join(args.outdir, "sub_exp" +
                              datetime.now().strftime('_%Y%m%d%H%M%S')
                              )  # e.g., sub_exp_20220411030524
        while os.path.exists(outdir):
            time.sleep(1)
            outdir = os.path.join(
                args.outdir,
                "sub_exp" + datetime.now().strftime('_%Y%m%d%H%M%S'))
        args.outdir = outdir
    # if not, make directory with given name
    os.makedirs(args.outdir)
    
    # set file handler to specify directory
    fh = logging.FileHandler(os.path.join(args.outdir,'log.txt'))
    fh.setFormatter(logger_format)
    logger.addHandler(fh)

    return logger

def parse_param_json(param_file):
    path = os.path.join('../system/flcore/params', param_file)
    params = json.loads(open(path).read())
    return params
