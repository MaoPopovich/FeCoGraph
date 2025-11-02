import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl.function as fn
import tqdm
import gc
from torch_geometric.nn.conv import MessagePassing,GCNConv
from torch_geometric.utils import scatter
from torch_geometric.data import Data
import gc
from typing import Optional

class Encoder(nn.Module):
    def __init__(self, in_channels: int,
                 num_hidden: int):
        super(Encoder, self).__init__()
        # define 2-layer GCN
        self.conv1 = GCNConv(in_channels, 2*num_hidden)
        self.conv2 = GCNConv(2*num_hidden, num_hidden)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x 


class GRACE(nn.Module):
    def __init__(self, in_channels: int=1433,
                 out_channels: int=7, 
                 num_hidden: int=256, 
                 num_proj_hidden: int=256,  
                 tau: float=1.0):
        super(GRACE, self).__init__()
        # define base encoder
        self.base = Encoder(in_channels, num_hidden)

        # define projection layers
        self.proj = nn.Sequential(nn.Linear(num_hidden, num_proj_hidden), nn.ELU(), nn.Linear(num_proj_hidden, num_hidden))

        # define classifier
        self.cls = nn.Linear(num_hidden, out_channels)

        self.tau = tau

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')
        
        # graph convolution operation
        z = self.base(x, edge_index)
        return z 

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        return torch.cat(losses)

    # We only define contrastive loss inside GRACE 
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
          l1 = self.semi_loss(h1, h2)
          l2 = self.semi_loss(h2, h1)
        else:
          l1 = self.batched_semi_loss(h1, h2, batch_size)
          l2 = self.batched_semi_loss(h2, h1, batch_size)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    # We only implement classifier inside GRACE, the calculation of CrossEntropy is implemented in Client.train()
    def generate_logits(self, z: torch.Tensor):
        return self.cls(z)
    
    def batched_supcon_loss(self, features, labels, batch_size, device, mask=None):
        """
        Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            we need to split features and labels according to batch size
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            # if labels.shape[0] != batch_size:
            #     raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # for simplicity, we use 'one' contrast_mode
        anchor_feature = features[:, 0]
        anchor_count = 1

        # normalize and compute logits
        anchor_feature = F.normalize(anchor_feature)
        contrast_feature = F.normalize(contrast_feature)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.tau)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.tau) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).squeeze(0)
        
        return loss

    def supcon_loss(self, z1, z2, labels, batch_size):
        device = z1.device
        h1 = self.proj(z1)
        h2 = self.proj(z2)
        h_cat = torch.cat([h1.unsqueeze(1), h2.unsqueeze(1)], dim=1)

        n_samples = h_cat.shape[0]
        n_batches = (n_samples - 1) // batch_size + 1
        losses = []

        for i in range(n_batches):
            feat = h_cat[i * batch_size:(i+1) * batch_size]
            lab = labels[i * batch_size: (i+1) * batch_size]

            losses.append(self.batched_supcon_loss(feat, lab, batch_size, device))
      
        return torch.cat(losses).mean()


class SupConLoss(nn.Module):
  """
  Adapted original SupConLoss to graph augmentation
  """
  def __init__(self, tau=0.07, device='cpu'):
    super(SupConLoss, self).__init__()
    self.tau = tau
    self.device = device

  def forward(self,features, labels=None, batch_size=64):
      n_samples = features.shape[0]
      n_batches = (n_samples - 1) // batch_size + 1
      losses = []

      for i in range(n_batches-1):
        feat = features[i * batch_size:(i+1) * batch_size]
        lab = labels[i * batch_size: (i+1) * batch_size]

        losses.append(self.batched_supcon_loss(feat, lab, batch_size, self.device))
      
      return torch.cat(losses).mean()

  
  def batched_supcon_loss(self, features, labels, batch_size, device, mask=None):
    """
    Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        we need to split features and labels according to batch size
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        # if labels.shape[0] != batch_size:
        #     raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(self.device)
    else:
        mask = mask.float().to(self.device)
    
    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    # for simplicity, we use 'one' contrast_mode
    anchor_feature = features[:, 0]
    anchor_count = 1

    # normalize and compute logits
    anchor_feature = F.normalize(anchor_feature)
    contrast_feature = F.normalize(contrast_feature)
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        self.tau)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (self.tau) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).squeeze(0)
    
    return loss
    


class SupConLoss_(nn.Module):
  """
  SupConLoss from SC_Net with only one augmentation views 
  """
  def __init__(self, tau=0.07, device='cpu'):
    super(SupConLoss, self).__init__()
    self.tau = tau
    self.device = device
  
  def forward(self, H, labels, batch_size=64):
    """
    Args:
        H: hidden vector of shape [n_samples, feature_dim, ...].
        labels: ground truth of shape [n_samples].
    Returns:
        A loss scalar.
    """
    num_samples = H.shape[0]
    Z = F.normalize(H)
    num_batches = (num_samples - 1) // batch_size + 1
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    a = torch.ones(num_samples).to(self.device) - torch.eye(num_samples).to(self.device)
    mask = mask * a
    mask_sum = mask.sum(dim=1)
    f = lambda x: torch.exp(x / self.tau)
    indices = torch.arange(0, num_samples).to(self.device)
    losses = []

    for i in range(num_batches):
      truc_mask = indices[i * batch_size: (i+1) * batch_size]
      refl_sim = f(torch.mm(Z[truc_mask], Z.t()))
      refl_sim_sum = refl_sim.sum(1)
      positive_matrix = refl_sim * mask
      ans = positive_matrix / refl_sim_sum # frac = exp(z_i*z_j / tau) / (\sum_k=1^2N 1(k≠i) exp(z_i*z_k / tau))
      ans = torch.where(mask<1e-8, 1 - mask, ans)  # fill with zero
      ans = torch.log().sum(1) / (mask_sum + 1e-8)  # SupCon_i = 1/|P(i)|* \sum_i=1^|P(i)| log(frac) 
      ans = - ans / num_samples # -1/2N * SupCon_i
      losses.append(ans)
    
    return torch.cat(losses).sum()



class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

class MulticlassEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        total = y_true.size(0)
        correct = (y_true == y_pred).to(torch.float32).sum()
        return (correct / total).item()

    def eval(self, res):
        return {'acc': self._eval(**res)}


