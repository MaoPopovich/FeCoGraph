import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *
from utils.augmentation import generate_views
from utils.general_utils import parse_param_json
import logging

logger = logging.getLogger(__name__)

class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_data(batch_size=1)
        # self.model.to(self.device)
        self.model.train()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for data in trainloader:
                if type(data) == type([]):
                    data = data[0].to(self.device)
                else:
                    data = data.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                if self.task_type == 'binary':
                    y_true = data.Label
                else:
                    y_true = data.Attack 

                if self.supcon_enabled:
                    # parse ssl hyper-parameters from json file
                    params = parse_param_json(self.param)

                    x_1, x_2, edge_index_1, edge_index_2 = generate_views(params, data, self.device)
                    z = self.model((data.x, data.edge_index))
                    z1 = self.model((data.x, data.edge_index))
                    z2 = self.model((x_2, edge_index_2))
                    output = self.model.generate_logits(z1)
                    

                    supcon_loss = self.model.supcon_loss(z1[data.train_mask], z2[data.train_mask], y_true[data.train_mask], self.batch_size) 
                    ce_loss = self.loss(output[data.train_mask], y_true[data.train_mask])
                    loss = (1 - params['scale_ratio']) * supcon_loss + params['scale_ratio'] * ce_loss
                else:
                    output = self.model(data)
                    loss = self.loss(output[data.train_mask], y_true[data.train_mask])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")