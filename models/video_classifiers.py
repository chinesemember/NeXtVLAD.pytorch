import torch
import torch.nn as nn
import torch.nn.functional as F

from .NeXtVLAD import NeXtVLAD


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


class ConvNeXtVLADModel(nn.Module):
    """
    A full Conv + neXtVLAD video classifier pipeline
    """

    def __init__(self, nextvlad_model, eigenvecs, eigenvals, center, device, opt):
        super(ConvNeXtVLADModel, self).__init__()
        import pretrainedmodels
        self.ftype = opt['type']
        self.conv = pretrainedmodels.__dict__[opt['type']](num_classes=1000, pretrained='imagenet')
        self.device = device
        self.eigenvecs = torch.from_numpy(eigenvecs).type(torch.FloatTensor)
        self.eigenvals = torch.from_numpy(eigenvals).type(torch.FloatTensor)
        self.center = torch.from_numpy(center).type(torch.FloatTensor)
        self.video_classifier = nextvlad_model

    def _process_batch(self, batch):
        output_features = self.conv.features(batch)
        # output_features = output_features.data.cpu()

        conv_size = output_features.shape[-1]

        if self.ftype == 'nasnetalarge' or self.ftype == 'pnasnet5large':
            relu = nn.ReLU()
            rf = relu(output_features)
            avg_pool = nn.AvgPool2d(conv_size, stride=1, padding=0)
            out_feats = avg_pool(rf)
        else:
            avg_pool = nn.AvgPool2d(conv_size, stride=1, padding=0)
            # B x H0 x 1 x 1
            out_feats = avg_pool(output_features)
        # B x H0
        out_feats = out_feats.view(out_feats.size(0), -1).cpu()

        # PCA (no whiten):
        # B x H0 (-) B x H0
        out_feats = out_feats - self.center
        # B x H0 -> B x 1 x (H0/2)
        out_feats = out_feats.unsqueeze(1).matmul(torch.t(self.eigenvecs))
        # verification:
        # (np) out_feats[0].detach().cpu().numpy().reshape(1, 2048).dot(self.eigenvecs.detach().cpu().numpy().T)
        #   ==
        # (torch) out_feats.unsqueeze(1).matmul(torch.t(self.eigenvecs))[0]

        # B x (H0/2)
        return out_feats.squeeze(1)

    def conv_forward(self, frame_batch):
        return self._process_batch(frame_batch)

    def nextvlad_model_forward(self, vid_feats, mask):
        return self.video_classifier.forward(vid_feats, mask)
