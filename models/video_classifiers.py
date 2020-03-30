import torch
import torch.nn as nn
import torch.nn.functional as F

from models.NeXtVLAD import NeXtVLAD


class NeXtVLADModel(nn.Module):
    def __init__(self, num_classes, num_clusters=64, dim=1024, lamb=2, hidden_size=1024,
                 groups=8, max_frames=300, drop_rate=0.5, gating_reduction=8):
        super(NeXtVLADModel, self).__init__()
        self.drop_rate = drop_rate
        self.group_size = int((lamb * dim) // groups)
        self.fc0 = nn.Linear(num_clusters * self.group_size, hidden_size)
        self.bn0 = nn.BatchNorm1d(1)
        self.fc1 = nn.Linear(hidden_size, hidden_size // gating_reduction)
        self.bn1 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(hidden_size // gating_reduction, hidden_size)
        self.logistic = nn.Linear(hidden_size, num_classes)

        self.video_nextvlad = NeXtVLAD(1024, max_frames=max_frames, lamb=lamb,
                                       num_clusters=num_clusters, groups=groups)

    def forward(self, x, mask=None):
        # B x M x N -> B x (K * (λN/G))
        vlad = self.video_nextvlad(x, mask=mask)

        # B x (K * (λN/G))
        if self.drop_rate > 0.:
            vlad = F.dropout(vlad, p=self.drop_rate)

        # B x (K * (λN/G))  -> B x H0
        activation = self.fc0(vlad)
        activation = self.bn0(activation.unsqueeze(1)).squeeze()
        activation = F.relu(activation)
        # B x H0 -> B x Gr
        gates = self.fc1(activation)
        gates = self.bn1(gates.unsqueeze(1)).squeeze()
        # B x Gr -> B x H0
        gates = self.fc2(gates)
        gates = torch.sigmoid(gates)
        # B x H0 -> B x H0
        activation = torch.mul(activation, gates)
        # B x H0 -> B x k
        out = self.logistic(activation)
        out = torch.sigmoid(out)

        return out

