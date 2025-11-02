# FeCoGraph
The Official Implementation of "FeCoGraph: Label-Aware Federated Graph Contrastive Learning for Few-Shot Network Intrusion Detection" (TIFS 2025)

# Label-aware Federated Graph Contrastive Learning for Few-shot NIDS (FeCoGraph)
+ This repository presents a unified federated graph learning framework for NIDS.
+ Part of these codes are motivated by [PFL-Non-IID](https://github.com/TsingZ0/PFL-Non-IID.git)
+ To run our ConFGL implementation, please select "grace" model and "Ditto" FL algorithm in  `run.sh` script.
+ You can set `supcon_enabled` and relevant hyper-parameters to adjust the impact of label-aware graph contrastive learning.
+ place your customized dataset (e.g., network graph for intrusion detection) in  `dataset` directory.

# Citation
+ please cite our work if you appreciate it, thank you a lot!!!
```
@article{mao2025fecograph,
  title={FeCoGraph: Label-Aware Federated Graph Contrastive Learning for Few-Shot Network Intrusion Detection},
  author={Mao, Qinghua and Lin, Xi and Xu, Wenchao and Qi, Yuxin and Su, Xiu and Li, Gaolei and Li, Jianhua},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025},
  publisher={IEEE}
}
```

