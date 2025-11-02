import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientbase import Client
import torch.nn.functional as F
from utils.augmentation import generate_views
from utils.general_utils import parse_param_json


class clientDitto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu
        self.plocal_steps = args.plocal_steps
        self.lamda = args.lamda
        self.drop_feature_rate = args.drop_feature_rate
        self.drop_edge_rate_1 = args.drop_edge_rate_1
        self.drop_edge_rate_2 = args.drop_edge_rate_2

        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = PerturbedGradientDescent(
            self.model_per.parameters(), lr=self.learning_rate, mu=self.mu)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_data()
        
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

                    x_1, x_2, edge_index_1, edge_index_2 = generate_views(params, data, self.device,self.drop_feature_rate,self.drop_edge_rate_1,self.drop_edge_rate_2)
                    z = self.model((data.x, data.edge_index))
                    z1 = self.model((x_1, edge_index_1))
                    z2 = self.model((x_2, edge_index_2))
                    output = self.model.generate_logits(z)
                    

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
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        
    def ptrain(self):
        trainloader = self.load_data()

        start_time = time.time()

        # self.model.to(self.device)
        self.model_per.train()

        max_local_steps = self.plocal_steps
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

                    x_1, x_2, edge_index_1, edge_index_2 = generate_views(params, data, self.device,self.drop_feature_rate,self.drop_edge_rate_1,self.drop_edge_rate_2)
                    z = self.model_per((data.x, data.edge_index))
                    z1 = self.model_per((x_1, edge_index_1))
                    z2 = self.model_per((x_2, edge_index_2))
                    output = self.model.generate_logits(z)

                    supcon_loss = self.model.supcon_loss(z1[data.train_mask], z2[data.train_mask], y_true[data.train_mask], self.batch_size) 
                    ce_loss = self.loss(output[data.train_mask], y_true[data.train_mask])
                    # loss = (1 - params['scale_ratio']) * supcon_loss + params['scale_ratio'] * ce_loss
                    loss = ce_loss
                else:
                    output = self.model_per(data)
                    loss = self.loss(output[data.train_mask], y_true[data.train_mask])
                self.optimizer_per.zero_grad()
                loss.backward()
                self.optimizer_per.step(self.model.parameters(), self.device)

        # self.model.cpu()

        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics_personalized(self):
        testloader = self.load_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model_per.eval()

        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for data in testloader:
                if type(data) == type([]):
                    data = data[0].to(self.device)
                else:
                    data = data.to(self.device)
                if self.supcon_enabled:
                    z = self.model_per(data)
                    output = self.model_per.generate_logits(z)
                else:
                    output = self.model_per(data)
                
                if self.task_type == 'binary':
                    y_true = data.Label
                else:
                    y_true = data.Attack

                test_acc += torch.sum(torch.argmax(output, dim=-1)[data.test_mask] == y_true[data.test_mask]).item()
                test_num += data.test_mask.sum()
        
        return test_acc, test_num

    def train_metrics_personalized(self):
        trainloader = self.load_data()
        self.model_per.eval()

        train_num = 0
        train_acc = 0
        losses = 0

        with torch.no_grad():
            for data in trainloader:
                if type(data) == type([]):
                    data = data[0].to(self.device)
                else:
                    data = data.to(self.device)
                if self.supcon_enabled:
                    z = self.model_per(data)
                    output = self.model_per.generate_logits(z)
                else:
                    output = self.model_per(data)
                
                if self.task_type == 'binary':
                    y_true = data.Label
                else:
                    y_true = data.Attack

                loss = self.loss(output[data.train_mask], y_true[data.train_mask])

                gm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model_per.parameters()], dim=0)
                loss += 0.5 * self.mu * torch.norm(gm-pm, p=2)
                
                train_acc += torch.sum(torch.argmax(output, dim=1)[data.train_mask] == y_true[data.train_mask]).item()
                train_num += data.train_mask.sum()
                losses += loss.item() * data.train_mask.sum()

        return train_acc, losses, train_num