import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.general_utils import read_client_data, parse_param_json


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.partition = args.partition
        self.dir_alpha = args.dir_alpha
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

        # supcon_enabled decides whether to perform contrastive learing and param includes ssl hyper-parameters
        self.param = args.param
        self.supcon_enabled = args.supcon_enabled
        self.task_type = args.task_type
        self.semi_enabled = args.semi_enabled
        self.train_ratio = args.train_ratio
        

    def load_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        data = read_client_data(self.dataset, self.partition, self.id, self.train_ratio, self.dir_alpha)
        return DataLoader([data])
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloader = self.load_data(batch_size=1)
        
        self.model.eval()

        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for data in testloader:
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
                test_acc += torch.sum(output.argmax(dim=-1)[data.test_mask] == y_true[data.test_mask]).item()
                test_num += data.test_mask.sum()
        
        return test_acc, test_num

    def train_metrics(self):
        trainloader = self.load_data(batch_size=1)
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
                train_num += data.train_mask.sum()
                losses += loss.item() * data.train_mask.sum()

        return losses, train_num


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
