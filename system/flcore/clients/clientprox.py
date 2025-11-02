import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientbase import Client
from utils.augmentation import generate_views
from utils.general_utils import parse_param_json

class clientProx(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu

        self.global_params = copy.deepcopy(list(self.model.parameters()))

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = PerturbedGradientDescent(
            self.model.parameters(), lr=self.learning_rate, mu=self.mu)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_data(batch_size=1)
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

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
                self.optimizer.step(self.global_params, self.device)

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()

    def train_metrics(self):
        trainloader = self.load_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for data in trainloader:
                if type(data) == type([]):
                    data = data[0].to(self.device)
                else:
                    data = data.to(self.device)

                if self.task_type == 'binary':
                    y_true = data.Label
                else:
                    y_true = data.Attack

                if self.supcon_enabled:
                    z = self.model(data)
                    output = self.model.generate_logits(z)
                else:
                    output = self.model(data)

                loss = self.loss(output[data.train_mask], y_true[data.train_mask])

                gm = torch.cat([p.data.view(-1) for p in self.global_params], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                loss += 0.5 * self.mu * torch.norm(gm-pm, p=2)

                train_num += data.train_mask.sum()
                losses += loss.item() * data.train_mask.sum()

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
