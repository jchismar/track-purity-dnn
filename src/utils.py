import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Subset
from torch.optim import LBFGS

def get_feature_names():
    return [
        "trk_pt", 
        "trk_inner_px", "trk_inner_py", "trk_inner_pz", "trk_inner_pt",
        "trk_outer_px", "trk_outer_py", "trk_outer_pz", "trk_outer_pt",
        "trk_ptErr",
        "trk_dxyClosestPV", "trk_dzClosestPV", "trk_dxy", "trk_dz", "trk_dxyErr", "trk_dzErr",
        "trk_nChi2", 
        "trk_eta", "trk_phi", "trk_etaErr", "trk_phiErr",
        "trk_ndof",
        "trk_nInnerLost", "trk_nOuterLost", "trk_nInnerInactive", "trk_nOuterInactive", "trk_nLostLay",
        "trk_nPixel", "trk_nStrip", 
        # 'trk_px', 'trk_py', 'trk_pz', 'trk_pt', 
        # 'trk_inner_px', 'trk_inner_py', 'trk_inner_pz', 'trk_inner_pt', 
        # 'trk_outer_px', 'trk_outer_py', 'trk_outer_pz', 'trk_outer_pt',
        # 'trk_eta', 'trk_lambda', 'trk_cotTheta', 'trk_phi', 
        # 'trk_dxy', 'trk_dz', 'trk_dxyPV', 'trk_dzPV', 'trk_dxyClosestPV', 'trk_dzClosestPV',
        # 'trk_ptErr', 'trk_etaErr', 'trk_lambdaErr', 'trk_phiErr', 'trk_dxyErr', 'trk_dzErr',
        # 'trk_refpoint_x', 'trk_refpoint_y', 'trk_refpoint_z',
        # 'trk_nChi2', 'trk_nChi2_1Dmod', 'trk_ndof', 'trk_q', 
        # 'trk_nValid', 'trk_nLost', 'trk_nInactive', 'trk_nPixel', 'trk_nStrip', 
        # 'trk_nOuterLost', 'trk_nInnerLost', 'trk_nOuterInactive', 'trk_nInnerInactive',
        # 'trk_nPixelLay', 'trk_nStripLay', 'trk_n3DLay', 'trk_nLostLay', 'trk_nCluster'
    ]

class MinMaxScaler:
    def __init__(self, data, feature_min=None, feature_max=None):
        if feature_min is not None and feature_max is not None:
            self.min = feature_min.float()
            self.max = feature_max.float()
        else:
            self.min = data.min(dim=0).values.float()
            self.max = data.max(dim=0).values.float()

    def transform(self, data):
        return (data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data):
        return data * (self.max - self.min + 1e-8) + self.min
    
    def __call__(self, sample):
        data, label = sample
        return self.transform(data), label
        
class FocalLoss(nn.Module):
    def __init__(self, pos_weight, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = pos_weight / (1 + pos_weight)
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        probas = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probas, 1 - probas)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_factor = alpha_t * (1 - pt) ** self.gamma
        loss = focal_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def stratified_split(dataset, train_fraction, val_fraction, test_fraction, random_seed=42):
    np.random.seed(random_seed)
    
    if hasattr(dataset, '_labels') and dataset._labels is not None:
        labels = dataset._labels.numpy()
    else:
        labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
    
    labels_tensor = torch.from_numpy(labels)
    fake_indices = torch.where(labels_tensor == 0)[0].numpy()
    real_indices = torch.where(labels_tensor == 1)[0].numpy()
    
    np.random.shuffle(fake_indices)
    np.random.shuffle(real_indices)
    
    n_fake = len(fake_indices)
    n_real = len(real_indices)
    
    n_train_fake = int(train_fraction * n_fake)
    n_val_fake = int(val_fraction * n_fake)
    
    n_train_real = int(train_fraction * n_real)
    n_val_real = int(val_fraction * n_real)
    
    train_indices = np.concatenate([
        fake_indices[:n_train_fake],
        real_indices[:n_train_real]
    ])
    val_indices = np.concatenate([
        fake_indices[n_train_fake:n_train_fake + n_val_fake],
        real_indices[n_train_real:n_train_real + n_val_real]
    ])
    test_indices = np.concatenate([
        fake_indices[n_train_fake + n_val_fake:],
        real_indices[n_train_real + n_val_real:]
    ])
    
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    return (Subset(dataset, train_indices), 
            Subset(dataset, val_indices), 
            Subset(dataset, test_indices))

def save_jit_inference_model(model, feature_min, feature_max, save_path, input_dim):
    class ModelWrapper(nn.Module):
        def __init__(self, model, feature_min=None, feature_max=None):
            super(ModelWrapper, self).__init__()
            self.model = model
            if feature_min is not None:
                self.register_buffer('feature_min', feature_min.float())
            else:
                self.register_buffer('feature_min', torch.zeros(input_dim))
            
            if feature_max is not None:
                self.register_buffer('feature_max', feature_max.float())
            else:
                self.register_buffer('feature_max', torch.ones(input_dim))

        def normalize(self, x):
            return (x - self.feature_min) / (self.feature_max - self.feature_min + 1e-8)
        
        def set_normalization_params(self, feature_min, feature_max):
            self.feature_min = feature_min.float().to(self.feature_min.device)
            self.feature_max = feature_max.float().to(self.feature_max.device)
        
        def forward(self, x):
            x = self.normalize(x)
            return torch.sigmoid(self.model(x))
        
    model.eval()
    sample_input = torch.randn(1, input_dim)
    traced_model = torch.jit.trace(ModelWrapper(model, feature_min, feature_max), sample_input)
    torch.jit.save(traced_model, save_path)