import numpy as np
import torch
from torch.utils.data import Dataset
from mmap_ninja import RaggedMmap
from pathlib import Path
import os
import glob
import torch.nn as nn

import torch
from fm4npp.utils import *
from .voxelizer import *

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

torch.manual_seed(42)

def rescale_serialize_Rlast(centers, scaler = 1e4, order='z'):
    """
    Reorder centroids based on a designated order.
    Rlast indicates that R will be the last global order.
    arr: (N x 3) -> should be integer location
    """
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if len(centers.shape) > 2:
        centers = centers.squeeze(0)
    arr = centers[..., 1:]
    
    arr = swap_dim(arr)
    toserial = (arr * scaler).long() # Making the floating points to integer.
    ordered = encode(toserial, batch=None, depth=16, order=order)
    sorter = torch.argsort(ordered)
    out = arr[sorter]
    out = swap_dim(out)
    out = torch.cat([centers[..., 0:1], out], dim=-1)
    
    return out.unsqueeze(0), sorter


def knn_later_indices_batch(A, k):
    """
    A: Tensor of shape (B, N, 3), where B = batch size, N = number of points per batch, D=3 coordinates.
       Assumed to be sorted by the last dimension if needed, but sorting is not mandatory for the logic here.
    k: Number of neighbors to find for each point, using only indices j > i.
    
    Returns:
        Tensor of shape (B, N, 3*k):
          - For each batch b, row i, we gather up to k neighbors from rows j>i.
          - If fewer than k neighbors exist, the remainder is padded with -100.
    """
    B, N, D = A.shape
    assert D == 3, "A must have shape (B, N, 3)"

    # 1) Compute pairwise distances for each batch
    #    - shape: (B, N, N)
    #      * A_expanded: (B, N, 1, 3)
    #      * A_tiled:    (B, 1, N, 3)
    #      => difference => norm => (B, N, N)
    A_expanded = A.unsqueeze(2)  # (B, N, 1, 3)
    A_tiled = A.unsqueeze(1)     # (B, 1, N, 3)
    pairwise_distances = torch.norm(A_expanded - A_tiled, dim=-1)  # (B, N, N)

    # 2) Only allow neighbors with strictly larger index j>i
    #    => we set i>=j to infinity so they won't be selected
    #    Build a mask for the upper triangle above the diagonal (i < j).
    #    mask_2d shape: (N, N), then broadcast to (B, N, N).
    mask_2d = torch.triu(torch.ones(N, N, device=A.device), diagonal=1).bool()  # 1 for j>i
    mask_3d = mask_2d.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
    pairwise_distances[~mask_3d] = float('inf')       # i>=j => inf

    # 3) Use top-k to find the nearest neighbors among valid (finite) ones
    #    - topk(...) along dimension=2
    #    - largest=False => we want the smallest distances
    #    * topk_vals: (B, N, k_limited)
    #    * topk_idx : (B, N, k_limited)
    #    where k_limited = min(k, N-1)
    k_limited = min(k, N-1)
    topk_vals, topk_idx = torch.topk(
        pairwise_distances, 
        k=k_limited,
        dim=2,  # neighbor dimension
        largest=False
    )  # shapes: (B, N, k_limited), (B, N, k_limited)

    # 4) If the user-specified k > k_limited, pad with inf/-1 to get final shape (B, N, k)
    if k_limited < k:
        pad_size = k - k_limited
        inf_pad = torch.full((B, N, pad_size), float('inf'), device=A.device)
        minus1_pad = torch.full((B, N, pad_size), -1, device=A.device, dtype=torch.long)

        topk_vals = torch.cat([topk_vals, inf_pad], dim=2)    # (B, N, k)
        topk_idx  = torch.cat([topk_idx,  minus1_pad], dim=2) # (B, N, k)

    # 5) Convert any inf distances to invalid => set index = -1
    inf_mask = torch.isinf(topk_vals)  # (B, N, k)
    topk_idx[inf_mask] = -1

    # 6) We now gather the actual coordinates for these neighbor indices
    #    - Create an output array full of -100 for padding
    knn_neighbors = torch.full((B, N, k, D), -100, device=A.device, dtype=A.dtype)  # (B, N, k, 3)

    # 6a) Build a "safe" version of the indices, replacing -1 with 0 to avoid index errors
    safe_idx = topk_idx.clone()
    safe_idx[safe_idx < 0] = 0

    # 6b) We'll do advanced indexing to fill valid neighbor slots
    valid_mask = (topk_idx >= 0)  # (B, N, k) => True where neighbor is valid

    # To do advanced indexing, we need the broadcasted batch/row/col indices:
    b_idx = torch.arange(B, device=A.device).view(B, 1, 1).expand(B, N, k)    # (B, N, k)
    n_idx = torch.arange(N, device=A.device).view(1, N, 1).expand(B, N, k)    # (B, N, k)
    # The "safe_idx" dimension is the neighbor index for each (b, n)
    # so we'll gather from dimension=1 in A => A[b, safe_idx, :]
    # We'll do advanced indexing on "neighbors[b, n, j, :]" = A[b, safe_idx[b, n, j], :]

    # Where valid, copy the data
    knn_neighbors[valid_mask] = A[b_idx[valid_mask], safe_idx[valid_mask], :]

    # 7) Finally, reshape to (B, N, 3*k)
    knn_neighbors = knn_neighbors.view(B, N, 3*k)
    return knn_neighbors

def swap_dim(arr, dims = [1,2]):
    c = arr.clone()
    c[..., 1] = arr[..., 2]
    c[..., 2] = arr[..., 1]
    return c

def strip_masked(g, maskval = -100):
    """input: 1 x N x group_size x 4"""
    assert g.size(0) == 1, 'only for batch_size of 1'
    masker = g[..., 1:].mean(-1).mean(-1) != -100
    return g[masker].unsqueeze(0)

def rescale_serialize_Rlast(centers, scaler = 1e4, order='z'):
    """
    Reorder centroids based on a designated order.
    Rlast indicates that R will be the last global order.
    arr: (N x 3) -> should be integer location
    """
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if len(centers.shape) > 2:
        centers = centers.squeeze(0)
    arr = centers[..., 1:]
    
    arr = swap_dim(arr)
    toserial = (arr * scaler).long() # Making the floating points to integer.
    ordered = encode(toserial, batch=None, depth=16, order=order)
    sorter = torch.argsort(ordered)
    out = arr[sorter]
    out = swap_dim(out)
    out = torch.cat([centers[..., 0:1], out], dim=-1)
    
    return out.unsqueeze(0), sorter

def serialize_neighbors(neighs, order='z'):
    """
    Reorder points by for-loop. Not efficient [LOOP], but aim for precision at this point.
    neighs: 1 x number of groups x group size x 4
    """
    if len(neighs.shape) > 3:
        neighs = neighs.squeeze(0)
        
    out = []
    ng, gs, c = neighs.shape
    pout, sorter = rescale_serialize_Rlast(neighs.reshape(-1, c), scaler = 1e4, order=order)
    proxy = torch.arange(ng).unsqueeze(-1).repeat(1, gs).reshape(-1)
    proxy_sorted = proxy[sorter]
    
    for i in range(ng):
        psorted = pout[:, proxy_sorted == i, :]
        out.append(psorted)
    sorted_neighs = torch.cat(out, dim=0)
    return sorted_neighs.unsqueeze(0)
    # return pout.reshape(1, ng, gs, c)

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size 
    
    def forward(self, exyz):
        '''
            input:1 N 4
            ---------------------------
            output: 1 G M 4
            center : 1 G 4
        '''
        xyz = exyz[..., 1:]
        batch_size, num_points, _ = xyz.shape
        center, cidx = sample_farthest_points(xyz, K=self.num_group) # B G 3
        
             
        idx = knn_points(center, xyz, K = self.group_size)[1] # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = exyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 4).contiguous()
        return neighborhood, exyz[:, cidx.squeeze(0)]
    
def rescale_polar_radius(arr, dim=-1, reduction_rate = 0.5):
    toadjust = arr.clone()
    toadjust[..., -1] *= reduction_rate
    return toadjust

def minmax_normalize(arr, max_, min_):
    return (arr - min_) / (max_ - min_)
    
def apply_norm(features):
    """Dim 2 and 3 are the same to preserve absolute distance"""
    fnorm = features.clone()
    for i in range(4):
        fnorm[..., i] = minmax_normalize(fnorm[..., i], features[..., i].max(), features[..., i].min())
    return fnorm
    
def group_points(arr, group_size, pad_val = -100):
    """
    Given a sequence of N x 4, group them by (N//group_size+1) x group_size x 4
    """
    if len(arr.shape) > 2:
        arr = arr.squeeze(0)
        
    n, c = arr.size()
    remainder = n % group_size
    gs_ = n // group_size
    if remainder != 0:
        pad = torch.ones(group_size - remainder, c) * pad_val
        arr = torch.cat([arr, pad], dim=0)
        gs_ += 1
    return arr.reshape(gs_, group_size, c)

def set_simpler(inputs, target, nleave = 3, npoint_lower_thr = 5):
        """
        Collisions tend to show several tens / hundreds of particles. Let's leave a few easier cases and train on them
        parameters
        ----------
            inputs: feature data - (1 x n x 4)
            target: data cluster identifier - (1 x n)
            leave: (maximum) number of trajectories to use per collision

        return
        ------
        reduced_inputs: reduced feature data - (1 x min(max_traj, nleave) x 4)
        reduced_target: reduced target - (1 x min(max_traj, nleave))
        """

        list_traj = torch.unique(target)
        n_traj = len(list_traj)
        traj2count = {}
        for traj in list_traj:
            # filter out those below count 5
            if (target == traj).sum() > npoint_lower_thr:
                traj2count[traj] = (target == traj).sum()

        trajs = list(traj2count.keys())
        counts = list(traj2count.values())

        chosen_indice = np.argsort(np.array(counts))[::-1][:nleave]
        chosen_trajs = [trajs[idx] for idx in chosen_indice]

        reduced_inputs = []
        reduced_target = []
        for traj in chosen_trajs:
            cidx = (target == traj).squeeze()
            reduced_inputs.append(inputs[:, cidx])
            reduced_target.append(target[:, cidx])

        reduced_inputs = torch.from_numpy(np.concatenate(reduced_inputs, axis = 1))
        reduced_target = torch.from_numpy(np.concatenate(reduced_target, axis = 1))

        post_n_traj = len(torch.unique(reduced_target))

        # print(n_traj, post_n_traj)

        return reduced_inputs, reduced_target
    
class TPCBatchDataset(Dataset):
    def __init__(self, 
                 data_root, 
                 version = 'pp_100k',
                 train = True,
                 split = 'pretrain',
                 nleave = 1e6,
                 npoint_lower_thr = 5,                  
                 group_size = 32, 
                 normalize_by_center = False, 
                 normalize = True,
                 order = 'REP', 
                 num_pred_points = 10, 
                 klen = 5,
                 len_chunk = 512,
                 chunk_training = False,
                 limit_data = False,
                 limit_size = 8000, 
                 voxelize = True,
                 space_filling_order = None,
                 space_filling_curve = 'z',
                 bin_dir = ''):
        
        split = split
        self.memmap_feature = RaggedMmap(os.path.join(data_root, 'features_{}'.format(split)))
        self.memmap_seg_target = RaggedMmap(os.path.join(data_root, 'seg_target_{}'.format(split)))
        self.memmap_reg_target = RaggedMmap(os.path.join(data_root, 'reg_target_{}'.format(split)))
        

        self.reco_cols = ['E', 'x', 'y', 'z']
        self.particle_reg_cols = ['px', 'py', 'pz', 'vtx_x', 'vtx_y', 'vtx_z', 'energy']
        self.particle_seg_col = 'track_id'
        
        # filtering out some trajectories
        self.nleave = nleave
        self.order = order
        self.npoint_lower_thr = npoint_lower_thr
        self.num_pred_points = num_pred_points
        
        # voxelization ablation
        self.voxelize = voxelize
        self.space_filling_order = space_filling_order
        self.space_filling_curve = space_filling_curve
        
        # for normalization
        self.eta_lim = {'min':-2, 'max':2}
        self.phi_lim = {'min':-torch.pi, 'max':torch.pi}
        self.r_lim = {'min': 31.371997833251953, 'max': 75.38493347167969}
        self.E_mean, self.E_std = 253.0982, 268.7093
        # (E)ta / (P)hi / (R)adius
        self.orderdict = {
            'EPR': {'dim_sweep_order':[2,1,0], 'revert_order':[2,1,0]},
            'RPE': {'dim_sweep_order':[0,1,2], 'revert_order':[0,1,2]},
            'REP': {'dim_sweep_order':[1,0,2], 'revert_order':[1,0,2]},
            'PER': {'dim_sweep_order':[2,0,1], 'revert_order':[1,2,0]},
                 }

        dim_sweep_order = self.orderdict[self.order]['dim_sweep_order']
        revert_order = self.orderdict[self.order]['revert_order']
        
        self.low_thr = 50
        self.normalize = normalize
        
        # Tokenizer
        self.group_size = group_size
        self.normalize_by_center = normalize_by_center
        self.voxelizer = Voxelizer(bin_dir = bin_dir, bin_version = 'v3', n_bins = (8, 8, 6), dim_sweep_order=dim_sweep_order, revert_order=revert_order)
        self.dim_sweep_order = dim_sweep_order
        self.revert_order = revert_order
        self.limit_data = limit_data
        self.limit_size = limit_size
        self.len_chunk = len_chunk
        
        self.train = train
        self.chunk_training = chunk_training
        self.filter_data(high_thr = 3200)
        import math
        self.data_scaler = 1 # [TOGGLE][TEMPORARY] SCALER
        
    def znormalize(self, arr, mean_, std_):
        """z-normalize"""
        return (arr - mean_) / std_
    
    def z_unnormalize(self, arr, mean_, std_):        
        return arr*std_ + mean_
    
    def minmax_normalize(self, arr, max_, min_):
        """Normalize between -1 and 1"""
        return (arr - min_) / (max_ - min_)
    
    def minmax_unnormalize(self, arr, max_, min_):
        return arr * (max_ - min_) + min_       
    
    def apply_norm(self, features):
        fnorm = features.clone()
        fnorm[..., 0] = self.znormalize(fnorm[..., 0], self.E_mean, self.E_std)
        fnorm[..., 1] = self.minmax_normalize(fnorm[..., 1], self.eta_lim['max'], self.eta_lim['min'])
        fnorm[..., 2] = self.minmax_normalize(fnorm[..., 2], self.phi_lim['max'], self.phi_lim['min'])
        fnorm[..., 3] = self.minmax_normalize(fnorm[..., 3], self.r_lim['max'], self.r_lim['min']) 
        return fnorm
    
    def apply_unnorm(self, features):
        fnorm = features.clone()
        fnorm[..., 0] = self.z_unnormalize(fnorm[..., 0], self.E_mean, self.E_std)
        fnorm[..., 1] = self.minmax_unnormalize(fnorm[..., 1], self.eta_lim['max'], self.eta_lim['min'])
        fnorm[..., 2] = self.minmax_unnormalize(fnorm[..., 2], self.phi_lim['max'], self.phi_lim['min'])
        fnorm[..., 3] = self.minmax_unnormalize(fnorm[..., 3], self.r_lim['max'], self.r_lim['min']) 
        return fnorm
    
    def filter_data(self, low_thr = -1, high_thr = 10e10):
        self.idxlist = []
        self.seqlens = []
        self.tooshort = []
        self.toolong = []
        self.longest = 0
        self.shortest = 1e10
        for i in range(len(self.memmap_feature)):
            len_ = self.memmap_feature[i].shape[0]
            if len_ < low_thr:
                self.tooshort.append(i)
            elif len_ > high_thr:
                self.toolong.append(i)
            else:
                self.idxlist.append(i)
                self.seqlens.append(len_)
                
                if self.longest < len_:
                    self.longest = len_
                if self.shortest > len_:
                    self.shortest = len_
           

            if self.limit_data and len(self.idxlist) == self.limit_size: 
                break

        # self.idxlist = create_sampled_lists_with_seq(self.idxlist, self.seqlens)
        
        print('[INFO] Filtering by N points. From {}, removed short {} long {}, remaining {}'.format(len(self.memmap_feature),
                                                                                                     len(self.tooshort),
                                                                                                     len(self.toolong),
                                                                                                     len(self.idxlist)))
        print('[INFO] Shortest: {}, Longest: {}'.format(self.shortest, self.longest))

        
        
        if not self.train and self.chunk_training:
            self.idxlist_chunking = []
            for k, idx in enumerate(self.idxlist):
                seqlen = self.seqlens[k]
                start_indices = get_chunk_start_indices(self.len_chunk, seqlen)
                for sidx in start_indices:
                    if seqlen - sidx > self.low_thr: # minimum multiplicity at 50 points.
                        self.idxlist_chunking.append((idx, sidx))
                    
            print('[INFO] Chunking the validation set. Original {} -> Chunk all {}'.format(len(self.idxlist), len(self.idxlist_chunking)))
        
    def cut_chunk(self, sequence, maxlen):
        """
        Apply chunk-based training. 
        If seq_len > maxlen, cut a sub-chunk from a random location.
        If the seq_len <= maxlen, return as it is.
        """
        N, D = sequence.shape
        start_idx = 0
        
        if maxlen > N:
            return sequence, start_idx
        
        else:
            # Select a random starting position
            start_idx = torch.randint(0, N - self.low_thr + 1, (1,)).item()
            
            # Slice out the chunk
            chunk = sequence[start_idx : start_idx + maxlen]
            return chunk, start_idx
        
        
    def __len__(self):
        if not self.train and self.chunk_training:
            return len(self.idxlist_chunking)   
        else:
            return len(self.idxlist)    
    
    def __getitem__(self, index):
        
        if not self.train and self.chunk_training:
            real_idx, start_idx = self.idxlist_chunking[index]
        else:
            real_idx = self.idxlist[index]
            
        features = torch.from_numpy(np.copy(self.memmap_feature[real_idx])).unsqueeze(0)
        target = torch.from_numpy(np.copy(self.memmap_seg_target[real_idx])).unsqueeze(0)

        # print(features.shape, target.shape)
        if not self.train and self.chunk_training:
            features = features[:, start_idx : start_idx+self.len_chunk]
            target = target[:, start_idx : start_idx+self.len_chunk]
            # print(features.shape, target.shape)
            
        # features, target = set_simpler(features.unsqueeze(0), target.unsqueeze(0), nleave = self.nleave, npoint_lower_thr = self.npoint_lower_thr)
        
        ## To polar representation
        polar_coord = cartesian_to_polar_batched(features[..., 1:])
        E = features[..., 0:1]
        polar_features = torch.cat([E, polar_coord], dim=-1)
        
        ## Normalize the polar representation
        if self.normalize:
            norm_features = self.apply_norm(polar_features)
        else:
            norm_features = polar_features
        
        # Sort by R
        ind = norm_features[...,-1].argsort(dim=1)
        norm_features = norm_features[:, ind.squeeze()]
        knearest_points = knn_later_indices_batch(norm_features[..., 1:], k=self.num_pred_points)
        norm_target = target[:, ind.squeeze()]
        
        if self.space_filling_order:
            _, zsorter = rescale_serialize_Rlast(norm_features, scaler = 1e4, order=self.space_filling_curve)
            serialized_points = norm_features[:, zsorter.squeeze()].squeeze(0)
            knearest_points = knearest_points[:, zsorter.squeeze()].squeeze(0)
            serialized_target = norm_target[:, zsorter.squeeze()].squeeze(0)
        
        else:
        
            if self.voxelize:
                quantized = self.voxelizer.tokenize(norm_features, start_idx = 1)
                grouped = self.voxelizer.grouping(quantized)
                gsort, sorter = grouped.sort(dim=-1, stable=True)
                serialized_points = norm_features[:, sorter.squeeze()].squeeze(0)
                knearest_points = knearest_points[:, sorter.squeeze()].squeeze(0)
                serialized_target = norm_target[:, sorter.squeeze()].squeeze(0)
            else:
                serialized_points = norm_features.squeeze(0)
                knearest_points = knearest_points.squeeze(0)
                serialized_target = norm_target.squeeze(0)
        
        return serialized_points * self.data_scaler, serialized_target, knearest_points * self.data_scaler


class MyCollator(object):
    def __init__(self):
        pass
        
    def __call__(self, batch):
        """
        Batchify data considering original point level input and center-level input
        pair1: features / target at original (after minor filtering)
        pair2: centers / neighs after centering and knn
        mask: masking the variable number of centers.
        """

        # Getting the longest point
        point_longest = 0
        for g, t, k in batch:
            if point_longest < g.size(0):
                point_longest = g.size(0)
        
        grouped,targets,knearest= [], [], []
        
        pad_val = -100
        glengths = []
        for g, t, k in batch:
            grouped.append(torch.nn.functional.pad(g, (0, 0, 0, point_longest - g.size(0)), value = pad_val))    
            targets.append(torch.nn.functional.pad(t, (0, point_longest - g.size(0)), value = pad_val))
            knearest.append(torch.nn.functional.pad(k, (0, 0, 0, point_longest - g.size(0)), value = pad_val))
       
        grouped = torch.stack(grouped)
        targets = torch.stack(targets)
        knearest = torch.stack(knearest)
            
        return (grouped, targets, knearest)



def get_data_loader(params, distributed):

    train_dataset = TPCBatchDataset(data_root = params.data_root, 
                                    version = params.data_version, 
                                    split = 'pretrain', 
                                    group_size = params.group_size, 
                                    normalize = True, 
                                    limit_data = params.limit_data, 
                                    limit_size = params.limit_size, 
                                    nleave = params.nleave, 
                                    order = params.order, 
                                    num_pred_points = params.klen, 
                                    len_chunk = params.len_chunk,
                                    chunk_training = params.chunk_training,
                                    voxelize = params.voxelize,
                                    bin_dir = params.stat_dir,
                                    space_filling_order = params.space_filling_order,
                                    space_filling_curve = params.space_filling_curve,
                                    train = True)
    
    test_dataset = TPCBatchDataset(data_root = params.data_root, 
                                   version = params.data_version, 
                                   split = 'test', 
                                   num_pred_points = params.klen,
                                   group_size = params.group_size, 
                                   normalize = True, 
                                   nleave = params.nleave, 
                                   chunk_training = params.chunk_training,
                                   bin_dir = params.stat_dir,
                                   voxelize = params.voxelize,
                                   order = params.order,
                                   space_filling_order = params.space_filling_order,
                                   space_filling_curve = params.space_filling_curve,
                                   train = False)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None

    my_collate_fn = MyCollator()
    
    train_dataloader = DataLoader(train_dataset,
                            batch_size=int(params.local_batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=(train_sampler is None),
                            sampler=train_sampler,
                            drop_last=True,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2,
                            collate_fn = my_collate_fn)
    
    test_dataloader = DataLoader(test_dataset,
                            batch_size=int(params.local_valid_batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=False,
                            sampler=test_sampler,
                            drop_last=True,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2,
                            collate_fn = my_collate_fn)
    
    return train_dataloader, train_sampler, test_dataloader, test_sampler

def get_val_loader(params, distributed):

    test_dataset = TPCBatchDataset(data_root = params.data_root, 
                                   version = params.data_version, 
                                   split = 'test', 
                                   num_pred_points = params.klen,
                                   group_size = params.group_size, 
                                   normalize = True, 
                                   **self.orderdict[params.order], 
                                   nleave = params.nleave, 
                                   chunk_training = params.chunk_training,
                                   train = False,
                                   order = params.order,)

   
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None

    my_collate_fn = MyCollator()
    
        
    test_dataloader = DataLoader(test_dataset,
                            batch_size=int(params.local_valid_batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=False,
                            sampler=test_sampler,
                            drop_last=True,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2,
                            collate_fn = my_collate_fn)
    
    return test_dataloader