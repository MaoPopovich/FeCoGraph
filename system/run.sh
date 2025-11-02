#!/bin/bash

# louvain 和 metis 两种划分方式会产生许多孤立点，在对比图增强阶段报错。因此，只使用 shard 和 hetero 两种划分方式。
#========================================= cicids2018 ================================================
# python main.py -data cicids2018 -tag metis -par metis -m grace -algo FedAvg -lr 0.001 -ls 5 -gr 100 -nc 10 -nb 7
# python main.py -data cicids2018 -tag metis -par metis -m grace -algo Local -lr 0.001 -ls 5 -gr 100 -nc 10 -nb 7
# python main.py -data cicids2018 -tag metis -par metis -m grace -algo Ditto -lr 0.001 -ls 5 -gr 100 -nc 10 -nb 7 -pls 5 -mu 0.1
 
# python main.py -data cicids2018 -tag hetero -par hetero -m grace -algo Local -lr 0.001 -ls 5 -gr 100 -nc 10 -nb 7
# python main.py -data cicids2018 -tag hetero -par hetero -m grace -algo FedAvg -lr 0.001 -ls 5 -gr 100 -nc 10 -nb 7
# python main.py -data cicids2018 -tag hetero -par hetero -m grace -algo Ditto -lr 0.001 -ls 5 -gr 100 -nc 10 -nb 7 -pls 5 -mu 0.1

# python main.py -out supcon -data cicids2018 -tag metis -par metis -m grace -algo FedAvg -lr 0.001 -ls 5 -gr 100 -nc 10 -nb 7 --supcon_enabled
# python main.py -out supcon -data cicids2018 -tag metis -par metis -m grace -algo Local -lr 0.001 -ls 5 -gr 100 -nc 10 -nb 7  --supcon_enabled
# python main.py -out supcon -data cicids2018 -tag metis -par metis -m grace -algo Ditto -lr 0.001 -ls 5 -gr 100 -nc 10 -nb 7 -pls 5 -mu 0.1  --supcon_enabled
 
python main.py -out supcon -data nf2018v2 -tag hetero -par hetero -m grace -algo Local -lr 0.001 -ls 5 -gr 100 -nc 10 -nb 7  --supcon_enabled
python main.py -out supcon -data nf2018v2 -tag hetero -par hetero -m grace -algo FedAvg -lr 0.001 -ls 5 -gr 100 -nc 10 -nb 7  --supcon_enabled
python main.py -out supcon -data nf2018v2 -tag hetero -par hetero -m grace -algo Ditto -lr 0.005 -ls 5 -gr 100 -nc 10 -nb 7 -pls 5 -mu 0.1  --supcon_enabled
python main.py -out supcon -data nf2018v2 -tag hetero -par hetero -m grace -algo FedProx -lr 0.001 -ls 5 -gr 100 -nc 10 -nb 7 -mu 0.1  --supcon_enabled

python main.py -out supcon -data nf2018v2 -tag shard -par shard -m grace -algo Local -lr 0.001 -ls 5 -gr 100 -nc 7 -nb 7  --supcon_enabled
python main.py -out supcon -data nf2018v2 -tag shard -par shard -m grace -algo FedAvg -lr 0.001 -ls 5 -gr 100 -nc 7 -nb 7  --supcon_enabled
python main.py -out supcon -data nf2018v2 -tag shard -par shard -m grace -algo Ditto -lr 0.005 -ls 5 -gr 100 -nc 7 -nb 7 -pls 5 -mu 0.1  --supcon_enabled
python main.py -out supcon -data nf2018v2 -tag shard -par shard -m grace -algo FedProx -lr 0.001 -ls 5 -gr 100 -nc 7 -nb 7 -mu 0.1  --supcon_enabled

