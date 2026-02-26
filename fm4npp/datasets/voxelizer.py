import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import plotly
import plotly.io as pio
import plotly.graph_objs as go
import matplotlib.pyplot as plt

import pickle
import matplotlib as mpl
# import seaborn as sns
import itertools

def equalObs(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))

def r_normalize(a, max_=75.38493347167969, min_=31.371997833251953):
    return (a - min_) / (max_ - min_)

def vertices(xmin=0, ymin=0, zmin=0, xmax=1, ymax=1, zmax=1):
        return {
            "x": [xmin, xmin, xmax, xmax, xmin, xmin, xmax, xmax],
            "y": [ymin, ymax, ymax, ymin, ymin, ymax, ymax, ymin],
            "z": [zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax],
            "i": [7, 0, 0, 0, 4, 4, 6, 1, 4, 0, 3, 6],
            "j": [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            "k": [0, 7, 2, 3, 6, 7, 1, 6, 5, 5, 7, 2],
        }

class Voxelizer:
    def __init__(self, bin_dir = './stats', bin_version = 'v1', n_bins = (12,12,9), dataset=None, limit = 10000, dim_sweep_order = [0,1,2], revert_order = [0,1,2]):
        self.data_proxy = None
        self.n_bins = n_bins
        bin_addr = os.path.join(bin_dir, 'bin_edges_{}_nbins_{}_{}_{}.pkl'.format(bin_version, *n_bins))
        if os.path.exists(bin_addr):
            self.final_bins = self.pickle_load(bin_addr)
        else:
            assert dataset is not None, 'dataset must be provided for computing bins'
            self.get_data(dataset, limit = limit)
            self.final_bins = self.compute_bins(n_bins = n_bins)
            self.pickle_save(bin_addr)
            
        self.all_bins = self.get_unique_voxel_IDs
        self.dim_sweep_order = dim_sweep_order
        self.revert_order = revert_order
        self.init_group_stats(dim_sweep_order=dim_sweep_order, revert_order=revert_order)
    
    def get_unique_voxel_IDs(self):
        return ['{}|{}|{}'.format(*a) for a in list(itertools.product(list(range(1, self.n_bins[0])), 
                                                                    list(range(1, self.n_bins[1])), 
                                                                    list(range(1, self.n_bins[2]))))]
    
    def draw_2d_edges(self, dimension = 'Eta', idx = 0):
        assert self.data_proxy is not None, 'must run self.compute_bins to draw this.'
        fig, ax = plt.subplots(1,1, figsize=(8, 4))
        sns.histplot(self.data_proxy[:, idx], ax = ax, kde = True)
        for a in self.final_bins[dimension]:
            ax.axvline(x = a, color = 'r', linestyle = '--') 
        ax.set_title('Distribution of {}'.format(dimension))
        plt.show()
        
    def draw_mesh(self, opacity=0.1):
        n_lines = self.n_bins[0] * self.n_bins[1] * self.n_bins[2]
        cmap = mpl.colormaps['plasma']
        color_cycle = [mpl.colors.rgb2hex(a) for a in cmap(np.linspace(0, 1.0, n_lines))]

        toplot = []
        x, y, z = self.final_bins.values()
        idx = 0
        for i in range(len(x)-1):
            for j in range(len(y)-1):
                for k in range(len(z)-1):
                    toplot.append(go.Mesh3d(
                    **vertices(xmin=x[i], ymin=y[j], zmin=z[k], xmax=x[i+1], ymax=y[j+1], zmax=z[k+1]),

                    opacity=opacity,
                    color=color_cycle[idx],
                    flatshading = True
                ))
                    idx += 1

        fig = go.Figure(data=toplot)

        self.drawit(fig)
        return 0
    
    def draw_3d_serialized_points(self, tensor, start_idx = 1):
        if len(tensor.shape) == 3:
            tensor = tensor.squeeze(0)
            
        color_cycle = pio.templates['plotly'].layout.colorway

        fig = go.Figure()
        
        ranks = list(range(tensor.size(0)))
        trace_tpc = go.Scatter3d(
            x=tensor[:, start_idx].numpy(), # x-coords
            y=tensor[:, start_idx+1].numpy(), # y-coords
            z=tensor[:, start_idx+2].numpy(), # z-coords, 0 is Energy
            mode='markers',
            legendgroup='0',
            marker=dict(size=4, opacity=1, color=ranks, colorscale='Viridis'),
            name='colored by radius'
        )
        fig.add_trace(trace_tpc)
        self.drawit(fig)
    def drawit(self, fig):
        camera = dict(
            up=dict(x=1, y=0, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
        fig.update_layout(
            title="voxelization mesh", 
            scene=dict(
                xaxis=dict(title="Eta"),
                yaxis=dict(title="Phi"),
                zaxis=dict(title="Radius")
            ),
            showlegend=True,
            width=1200, 
            height=800,
            scene_camera=camera,
        )

        fig.show()
        return 0
    
    def init_group_stats(self, dim_sweep_order, revert_order):
        """Define several linking dictionaries and reference array for grouping operation"""
        self.bin2idx = {}
        self.idx2bin = {}
        self.bin2edges = {}
        self.idx2edges = {}
        refarray = []
        dims = ['Eta', 'Phi', 'Radius']
        self.bins_in_order = []

        for i, a in enumerate(list(itertools.product(
            list(range(1, self.n_bins[dim_sweep_order[0]]+1)),
            list(range(1, self.n_bins[dim_sweep_order[1]]+1)), 
            list(range(1, self.n_bins[dim_sweep_order[2]]+1)), 
            ))):
            
            eidx, pidx, ridx = a[revert_order[0]], a[revert_order[1]], a[revert_order[2]]

            refarray.append(torch.Tensor([eidx, pidx, ridx]))
            bin_ = '{}|{}|{}'.format(eidx, pidx, ridx)
            self.bins_in_order.append(bin_)
            self.bin2idx[bin_] = i
            self.idx2bin[i] = [bin_]
            edges = []
            for dim_, idx in zip(dims, [eidx, pidx, ridx]):
                edges += [self.final_bins[dim_][iidx] for iidx in [idx-1, idx]]
            self.bin2edges[bin_] = edges
            self.idx2edges[i] = edges   
        self.refarray = torch.stack(refarray, dim=0).unsqueeze(0)
        return 0
        
    def tokenize(self, tensor, start_idx = 0):
        """Given B x N x C -> compute the bin for all dimensions B x N x C"""
        if len(tensor.shape) == 3:
            tensor = tensor.squeeze(0)
        three_voxels = []
        for name, idx in [('Eta', start_idx), ('Phi', start_idx + 1), ('Radius', start_idx + 2)]:
            boundaries = torch.Tensor(self.final_bins[name])
            three_voxels.append(torch.bucketize(tensor[:, idx].contiguous(), boundaries))
        voxelized = torch.stack(three_voxels, dim = -1)
        return voxelized
    
    def grouping(self, voxelized):
        """Given B x N x C -> compute the group cell for all dimensions B x N"""
        grouped = torch.cdist(voxelized.float(), self.refarray, p=2.0).argmin(-1)
        return grouped.squeeze(0) 
    
        
    def get_data(self, train_dataset, limit = 100000):
        all_dat = []
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        for i, (features, targets, centers, neighs, batch_mask) in tqdm(enumerate(train_loader)):
            all_dat.append(features[0, :, 1:])
            if i > limit:
                print('Limit is {}, early stopping...'.format(limit))
                break
        self.data_proxy = torch.cat(all_dat, dim =0)
        return 0
    
    def pickle_load(self, addr):
        import pickle
        with open(addr, 'rb') as f:
            final_bins = pickle.load(f)
        print('self.final_bins loaded from {}'.format(addr))
        return final_bins
        
    def pickle_save(self, addr):
        import pickle
        with open(addr, 'wb') as f:
            pickle.dump(self.final_bins, f)
        print('self.final_bins saved to {}'.format(addr))
        
    def compute_bins(self, n_bins = (12,12,9)):
        final_bins = {}
        # ETA/Phi
        for name, idx in [('Eta', 0), ('Phi', 1)]:
            edges = list(equalObs(self.data_proxy[:, idx], n_bins[idx]))
            edges[0] = 0.0
            edges[-1] = 1.0
            final_bins[name] = edges
            
        # Radius
        final_bins['Radius'] = self.get_Radius_bins(idx = -1)
        return final_bins

    def get_Radius_bins(self, idx = -1, layer_split_thresholds = [40, 57]):
        # R - it considers layers of detectors
        uniq = torch.unique(self.data_proxy[:, idx]) 
        Ly1, Ly2 = layer_split_thresholds 
        R_sections_boundary = [uniq.min(), 
                             uniq[uniq < r_normalize(Ly1)].max()  + (uniq[uniq >= r_normalize(Ly1)].min() -  uniq[uniq < r_normalize(Ly1)].max()) / 2 ,                  
                             uniq[uniq < r_normalize(Ly2)].max()  + (uniq[uniq >= r_normalize(Ly2)].min() - uniq[uniq < r_normalize(Ly2)].max()) / 2,                  
                            uniq.max()]

        layer_bins = [self.n_bins[-1]//3] * 3
        rad_data = self.data_proxy[:, idx]
        R_boundaries = {}
        for i, bin_ in enumerate(layer_bins):
            sec_start, sec_end = R_sections_boundary[i], R_sections_boundary[i+1]

            sec_boundaries = equalObs(rad_data[np.logical_and(sec_start < rad_data,  rad_data< sec_end)], bin_)
            R_boundaries[i] = sec_boundaries

        for i in range(3):
            start, end = R_sections_boundary[i].item(), R_sections_boundary[i+1].item()
            R_boundaries[i][0], R_boundaries[i][-1] = start, end

        return list(R_boundaries[0]) + list(R_boundaries[1][1:]) + list(R_boundaries[2][1:])

    