import os

import math
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import random
import math

import torch
import torch.nn as nn

from .z_order import xyz2key as z_order_encode_
from .z_order import key2xyz as z_order_decode_
from .hilbert import encode as hilbert_encode_
from .hilbert import decode as hilbert_decode_

from ruamel.yaml import YAML

class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)        
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and k not in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = type(value)(self.__class__(x)
                     if isinstance(x, dict) else x for x in value)
        elif isinstance(value, dict) and not isinstance(value, EasyDict):
            value = EasyDict(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, *args):
        if hasattr(self, k):
            delattr(self, k)
        return super(EasyDict, self).pop(k, *args)

    
class YParams():
  """ Yaml file parser """
  def __init__(self, yaml_filename, config_name, print_params=False):
    self._yaml_filename = yaml_filename
    self._config_name = config_name
    self.params = {}

    if print_params:
      print("------------------ Configuration ------------------")

    with open(yaml_filename) as _file:

      for key, val in YAML().load(_file)[config_name].items():
        if print_params: print(key, val)
        if val =='None': val = None

        self.params[key] = val
        self.__setattr__(key, val)

    if print_params:
      print("---------------------------------------------------")

  def __getitem__(self, key):
    return self.params[key]

  def __setitem__(self, key, val):
    self.params[key] = val
    self.__setattr__(key, val)

  def __contains__(self, key):
    return (key in self.params)

  def update_params(self, config):
    for key, val in config.items():
      self.params[key] = val
      self.__setattr__(key, val)

  def log(self):
    print("------------------ Configuration ------------------")
    print("Configuration file: "+str(self._yaml_filename))
    print("Configuration name: "+str(self._config_name))
    for key, val in self.params.items():
        print(str(key) + ' ' + str(val))
    print("---------------------------------------------------")



def register_fine_grained_forward_hooks(model):
    """
    Registers forward hooks on all modules that have parameters.
    When a module produces an output containing NaNs, the hook prints the module's full name 
    (as provided by model.named_modules()) along with the output index.
    
    Returns:
        A list of hook handles.
    """
    # Build a dictionary mapping module objects to their full names.
    module_to_name = {module: name for name, module in model.named_modules()}
    
    hooks = []
    def forward_hook(module, input, output):
        # Normalize output to a list for uniformity.
        outputs = output if isinstance(output, (list, tuple)) else [output]
        for idx, o in enumerate(outputs):
            if o is not None and torch.isnan(o).any():
                # Retrieve the full name of the module from our dictionary.
                mod_name = module_to_name.get(module, "UnnamedModule")
                print(f"[Forward Hook] NaN detected in module '{mod_name}' at output index {idx}")
    # Only register hooks on modules that own parameters (non-recursively)
    for module in model.modules():
        # If a module has any parameters that require gradients, attach the hook.
        if any(p.requires_grad for p in module.parameters(recurse=False)):
            hook_handle = module.register_forward_hook(forward_hook)
            hooks.append(hook_handle)
    return hooks

def register_module_forward_nan_hooks(model):
    """
    Registers forward hooks on all modules in the model.
    When a module produces an output that contains any NaN values,
    it prints a message with the module's name.
    Returns a list of hook handles.
    """
    hooks = []
    def forward_hook(module, input, output):
        # Check if the output is a Tensor (or list/tuple) and print the module class name.
        outputs = output if isinstance(output, (list, tuple)) else [output]
        for idx, o in enumerate(outputs):
            if o is not None and torch.isnan(o).any():
                print(f"[Forward Hook] NaN detected in output index {idx} of module {module.__class__.__name__}")
    for module in model.modules():
        hooks.append(module.register_forward_hook(forward_hook))
    return hooks


def register_param_backward_nan_hooks(model):
    """
    Registers a backward hook on each parameter that requires gradients.
    When the gradient contains any NaN values, the hook prints the parameter's name.
    Returns a list of hook handles so you can remove them later if needed.
    """
    hooks = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Bind the current name with a default argument.
            hook = param.register_hook(lambda grad, n=name: 
                                         print(f"[Backward Hook] NaN detected in gradient for parameter {n}")
                                         if torch.isnan(grad).any() else None)
            hooks.append(hook)
    return hooks


def register_forward_nan_hooks(model):
    """
    Registers forward hooks on all modules in the model.
    When a module produces an output that contains any NaN values,
    it prints a message with the module's class name.
    Returns a list of hook handles (so you can remove them later if desired).
    """
    hooks = []
    def forward_hook(module, input, output):
        # Check if output is a Tensor or a list/tuple of Tensors
        outputs = output if isinstance(output, (list, tuple)) else [output]
        for idx, o in enumerate(outputs):
            if o is not None and torch.isnan(o).any():
                print(f"[Forward Hook] NaN detected in module {module.__class__.__name__} at output index {idx}")
    for module in model.modules():
        hooks.append(module.register_forward_hook(forward_hook))
    return hooks

def register_backward_nan_hooks(model):
    """
    Registers backward hooks on all modules in the model.
    When a module's gradient output contains any NaN values,
    it prints a message with the module's class name.
    Returns a list of hook handles.
    """
    hooks = []
    def backward_hook(module, grad_input, grad_output):
        for idx, grad in enumerate(grad_output):
            if grad is not None and torch.isnan(grad).any():
                print(f"[Backward Hook] NaN detected in module {module.__class__.__name__} at grad_output index {idx}")
    for module in model.modules():
        hooks.append(module.register_backward_hook(backward_hook))
    return hooks

def check_model_parameters(model):
    """
    Checks all parameters in the model and returns a comma-separated string
    where for each parameter of interest (e.g., A_log, dt_bias, lin_B, lin_C, D)
    we output a string of the form:
       <param_name> min=<value>, max=<value>, mean=<value>, NaN=<n>, Inf=<n>
    """
    info_list = []
    def param_stats(tensor):
        flat = tensor.view(-1)
        return f"min={flat.min().item():.4f}, max={flat.max().item():.4f}, mean={flat.mean().item():.4f}, NaN={torch.isnan(flat).sum().item()}, Inf={torch.isinf(flat).sum().item()}"
    
    for name, param in model.named_parameters():
        # Choose parameters you suspect; modify the list as needed.
        if any(key in name for key in ["A_log", "dt_bias", "lin_B", "lin_C", "D"]):
            stats = param_stats(param.data)
            info_list.append(f"{name}: {stats}")
    return ", ".join(info_list)


def get_ssm_debug_info_str(model, prefix=""):
    """
    Returns a comma-separated string with statistics for selected SSM parameters.
    
    For each parameter whose name contains one of the keys ("A_log", "dt_bias", "lin_B", "lin_C", "D"),
    the function produces output in the format:
       <param_name> min=<value>, <param_name> max=<value>, <param_name> mean=<value>, <param_name> NaN=<value>, <param_name> Inf=<value>
    
    Args:
        model : nn.Module
            The model (or submodule) to inspect.
        prefix: str
            An optional prefix to prepend to each parameter name.
    
    Returns:
        A single comma-separated string with all the statistics.
    """
    with torch.no_grad():
        def param_stats(tensor):
            flat = tensor.view(-1)
            return {
                "min": flat.min().item(),
                "max": flat.max().item(),
                "mean": flat.mean().item(),
                "NaN": torch.isnan(flat).sum().item(),
                "Inf": torch.isinf(flat).sum().item(),
            }
        
        stats_list = []
        for name, param in model.named_parameters():
            if any(key in name for key in ["A_log", "dt_bias", "lin_B", "lin_C", "D"]):
                if param.numel() > 0:
                    stats = param_stats(param.data)
                    stats_list.append(
                        f"{prefix}{name} min={stats['min']:.4f}, "
                        f"{prefix}{name} max={stats['max']:.4f}, "
                        f"{prefix}{name} mean={stats['mean']:.4f}, "
                        f"{prefix}{name} NaN={stats['NaN']}, "
                        f"{prefix}{name} Inf={stats['Inf']}"
                    )
        return ", ".join(stats_list)


def compute_relative_spherical_angles(points):
    """
    Computes relative spherical angles (θ, φ, ψ) for a 3D trajectory.
    Inputs:
      - points: (B, N, 3) Tensor representing a batch of 3D trajectories
    Outputs:
      - angles: (B, N, 3) Tensor with θ (azimuth), φ (elevation), ψ (roll)
    """
    # Compute displacement vectors (B, N-1, 3)
    displacement = points[:, 1:, :] - points[:, :-1, :]

    # Compute θ (Azimuth) - Rotation around Z-axis
    θ = torch.atan2(displacement[:, :, 1], displacement[:, :, 0])  # (B, N-1)

    # Compute φ (Elevation) - Angle from XY-plane
    φ = torch.atan2(displacement[:, :, 2], torch.norm(displacement[:, :, :2], dim=-1))  # (B, N-1)

    # Compute ψ (Roll) - Rotation around displacement vector
    d_prev = displacement[:, :-1, :]  # (B, N-2, 3)
    d_next = displacement[:, 1:, :]   # (B, N-2, 3)
    dot_product = torch.sum(d_prev * d_next, dim=-1)  # (B, N-2)
    norms = torch.norm(d_prev, dim=-1) * torch.norm(d_next, dim=-1)  # (B, N-2)
    
    # Ensure numerical stability by clamping
    cos_ψ = torch.clamp(dot_product / (norms + 1e-8), -1.0, 1.0)  # Avoid NaNs
    ψ = torch.acos(cos_ψ)  # (B, N-2)

    # Pad first step with zeros
    θ = torch.cat([torch.zeros(θ.shape[0], 1, device=θ.device), θ], dim=1)
    φ = torch.cat([torch.zeros(φ.shape[0], 1, device=φ.device), φ], dim=1)
    ψ = torch.cat([torch.zeros(ψ.shape[0], 2, device=ψ.device), ψ], dim=1)  # Two padding for roll

    return torch.stack([θ, φ, ψ], dim=-1)  # (B, N, 3)


def compute_pairwise_distances(x):
    """ Compute relative positional encoding based on pairwise Euclidean distance """
    B, N, D = x.shape
    x_i = x.unsqueeze(2)  # (B, N, 1, D)
    x_j = x.unsqueeze(1)  # (B, 1, N, D)
    
    pairwise_dist = torch.norm(x_i - x_j, dim=-1)  # (B, N, N)
    return pairwise_dist

def apply_bin_weights_torch(bin_edges: torch.Tensor,
                            bin_weights: torch.Tensor,
                            x: torch.Tensor) -> torch.Tensor:
    """
    Given:
      bin_edges   : 1D Tensor of shape (k+1,), monotonically increasing.
      bin_weights : 1D Tensor of shape (k,), where bin_weights[i] is the weight for
                    the bin [bin_edges[i], bin_edges[i+1]).
      x           : 1D Tensor of shape (n,), values to be binned.

    Returns:
      weights_out : 1D Tensor of shape (n,), the weight for each x[i] based on its bin.

    Example:
      bin_edges   = tensor([0., 10., 20., 30.])
      bin_weights = tensor([1.0, 2.0, 3.0])
      x           = tensor([ 3., 15., 28., 11.,  5.])

      For each value of x:
        3.0  -> bin 0 => weight=1.0
        15.0 -> bin 1 => weight=2.0
        28.0 -> bin 2 => weight=3.0
        11.0 -> bin 1 => weight=2.0
        5.0  -> bin 0 => weight=1.0

      Output => [1.0, 2.0, 3.0, 2.0, 1.0]
    """
    # 1) Use torch.bucketize to find the bin index for each x[i].
    #    bucketize returns integers in [0, len(bin_edges)].
    #    If x[i] < bin_edges[0], index=0
    #    If x[i] >= bin_edges[-1], index=len(bin_edges)
    indices = torch.bucketize(x, bin_edges, right=False)

    # 2) Subtract 1 so that a value falling in bin_edges[i..i+1] => bin index = i
    #    This is analogous to np.digitize(...)-1 in NumPy.
    indices = indices - 1

    # 3) Clamp indices to [0, len(bin_weights)-1] in case x is outside the bin ranges
    indices = indices.clamp(min=0, max=bin_weights.shape[0] - 1)

    # 4) Gather the final weights for each element in x
    weights_out = bin_weights[indices]
    return weights_out
    
def pickle_load(addr):
    import pickle
    with open(addr, 'rb') as f:
        final_bins = pickle.load(f)
    print('Loaded stat from {}'.format(addr))
    return final_bins
    
def z_normalize_min1(tensor, mask_value=-100):
    # Create a boolean mask for valid values.
    valid_mask = (tensor != mask_value)
    
    # Compute the per-row mean using only valid values.
    mean = torch.sum(tensor * valid_mask, dim=1, keepdim=True) / torch.sum(valid_mask, dim=1, keepdim=True)
    
    # Compute the per-row standard deviation using only valid values.
    std = torch.sqrt(torch.sum(((tensor - mean) * valid_mask) ** 2, dim=1, keepdim=True) / torch.sum(valid_mask, dim=1, keepdim=True))
    
    # Compute z-scores.
    z = (tensor - mean) / std
    
    # Compute per-row minimum of z (only over valid values). 
    # We set masked positions to +infinity so they do not affect the min computation.
    z_min_candidates = torch.where(valid_mask, z, torch.tensor(float('inf'), device=tensor.device))
    z_min = z_min_candidates.min(dim=1, keepdim=True).values
    
    # Shift the z-scores so that the minimum valid value becomes 1.
    result = z - z_min + 1
    
    # Optionally, keep the masked values as mask_value.
    result = torch.where(valid_mask, result, torch.tensor(mask_value, dtype=tensor.dtype, device=tensor.device))
    return result
    
def get_chunk_start_indices(x: int, seqlen: int) -> torch.Tensor:
    """
    Generate a tensor of starting indices to divide a sequence into chunks of size x.
    The last chunk may be shorter if seqlen is not an exact multiple of x.

    Args:
        x (int): Chunk size.
        seqlen (int): Total sequence length.

    Returns:
        torch.Tensor: A 1D tensor containing starting indices for each chunk.
    """
    return torch.arange(0, seqlen, step=x)
    
def assign_bins(indices, seq_lengths, bins):
    """
    Assign each index to one of four bins based on its sequence length.
    
    Args:
        indices (list): List of indices.
        seq_lengths (list): List of sequence lengths corresponding to each index.
        bins (list): List of bin edges. For example: [50, 726, 1249, 1939, 3200]
                     (Assumed to define half-open intervals: [bins[i], bins[i+1]))
    
    Returns:
        dict: A dictionary with keys 1, 2, 3, 4 where each value is a list of indices 
              belonging to that bin.
    """
    bin_dict = {1: [], 2: [], 3: [], 4: []}
    for idx, length in zip(indices, seq_lengths):
        for i in range(len(bins) - 1):
            if bins[i] <= length < bins[i+1]:
                bin_dict[i+1].append(idx)
                break
    return bin_dict

def sample_partition(items, weights):
    """
    Partition a list of items into len(weights) parts according to the relative weights.
    
    Args:
        items (list): List of items to partition.
        weights (list of float): The weights corresponding to each partition.
                                 The weights need not sum to 1.
    
    Returns:
        list of lists: A list of sublists where each sublist corresponds to the items 
                       for that weight.
    """
    n = len(items)
    total = sum(weights)
    counts = [int(round(n * (w / total))) for w in weights]
    
    # Adjust counts so that the total equals n
    diff = n - sum(counts)
    for i in range(abs(diff)):
        counts[i % len(counts)] += 1 if diff > 0 else -1

    parts = []
    start = 0
    for count in counts:
        parts.append(items[start:start+count])
        start += count
    return parts

def create_sampled_lists(bin_dict):
    """
    Partition the indices from each bin into four final lists with the target compositions:
    
      List 1: 100% from bin1.
      List 2: 50% from bin1, 50% from bin2.
      List 3: 33% from bin1, 33% from bin2, 34% from bin3.
      List 4: 25% from bin1, 25% from bin2, 25% from bin3, 25% from bin4.
    
    Args:
        bin_dict (dict): Dictionary of bins (keys 1,2,3,4) with lists of indices.
    
    Returns:
        list: A list of four sublists (final lists) containing indices.
    """
    final_lists = {1: [], 2: [], 3: [], 4: []}
    
    # Bin 1 contributes to lists 1, 2, 3, 4.
    if bin_dict[1]:
        random.shuffle(bin_dict[1])
        parts = sample_partition(bin_dict[1], [1, 0.5, 0.33, 0.25])
        final_lists[1].extend(parts[0])
        final_lists[2].extend(parts[1])
        final_lists[3].extend(parts[2])
        final_lists[4].extend(parts[3])
    
    # Bin 2 contributes to lists 2, 3, 4.
    if bin_dict[2]:
        random.shuffle(bin_dict[2])
        parts = sample_partition(bin_dict[2], [0.5, 0.33, 0.25])
        final_lists[2].extend(parts[0])
        final_lists[3].extend(parts[1])
        final_lists[4].extend(parts[2])
    
    # Bin 3 contributes to lists 3 and 4.
    if bin_dict[3]:
        random.shuffle(bin_dict[3])
        parts = sample_partition(bin_dict[3], [0.34, 0.25])
        final_lists[3].extend(parts[0])
        final_lists[4].extend(parts[1])
    
    # Bin 4 contributes only to list 4.
    if bin_dict[4]:
        random.shuffle(bin_dict[4])
        final_lists[4].extend(bin_dict[4])
    
    return [final_lists[1], final_lists[2], final_lists[3], final_lists[4]]

def create_sampled_lists_with_seq(indices, seq_lengths):
    """
    Creates final lists of indices and corresponding sequence lengths.
    
    Args:
        indices (list): List of indices.
        seq_lengths (list): List of sequence lengths corresponding to each index.
        bins (list): List of bin edges.
    
    Returns:
        tuple: (final_lists, final_seq_lists)
          - final_lists: list of four lists of indices.
          - final_seq_lists: list of four lists of sequence lengths corresponding to those indices.
    """
    bins = [0, 726, 1249, 1939, 3201]
    
    # Create a mapping from index to sequence length.
    index_to_length = {idx: length for idx, length in zip(indices, seq_lengths)}
    
    # Step 1: Assign indices to bins.
    bin_dict = assign_bins(indices, seq_lengths, bins)
    
    # Step 2: Create final lists of indices.
    final_lists = create_sampled_lists(bin_dict)
    
    # Build parallel lists of sequence lengths using the mapping.
    final_seq_lists = [[index_to_length[i] for i in lst] for lst in final_lists]
     
    out = []
    for v in final_lists:
        out += v
    return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ConstantFieldPropagatorTorch:
    def __init__(self, magnetic_field_z):
        """
        Initialize the propagator.

        Args:
            charge (float): Particle charge in units of elementary charge (e.g., +1, -1).
            momentum (torch.Tensor): Initial momentum vector [px, py, pz] in GeV/c.
            magnetic_field_z (float): Constant magnetic field along the z-axis in Tesla.
        """
        self.Bz = torch.tensor([magnetic_field_z])  # Tesla

        # Constants
        self.c = 2.99792458e8  # Speed of light (m/s)
        self.GeV_to_kg_m_per_s = 5.344e-19  # GeV/c to kg·m/s
        self.m_to_cm = 100
        self.e_charge = 1.602e-19
        self.r_scale = self.GeV_to_kg_m_per_s / self.e_charge * self.m_to_cm
        self.z_hat = torch.tensor([0.0, 0.0, 1.0])
        self.B_hat = torch.sign(self.Bz) * self.z_hat

    def safe_normalize(self, vectors):
        norms = torch.norm(vectors, dim=1, keepdim=True)  # Calculate norms
        norms = torch.where(norms == 0, torch.tensor(1e-12, device=vectors.device, dtype=vectors.dtype), norms)  # Prevent div by 0
        return norms, vectors / norms


    def propagate(self, freepars):
        """
        Propagate the (charged)particle for a given path length to perigee.

        Args:
            freepars (torch.Tensor): Input tensor of shape (N, 7), where each row is (x, y, z, px, py, pz, q).

        Returns:
            ppars (torch.Tensor): Output tensor of shape (N, 7), where each row is (x, y, z, px, py, pz, q). The track state after propagate to perigee
        """
        position = freepars[..., :3]  # Shape: (N, 3)
        momentum = freepars[..., 3:6]  # Shape: (N, 3)
        pz = momentum[...,2:3]
        charge = freepars[..., 6:]  # Shape: (N, 1)
        charge_sign = torch.sign(charge) # Shape: (N, 1)

        B_hat = self.B_hat.to(freepars.device).unsqueeze(0).expand_as(momentum)

        # Compute radius vector of curvature R = (p x B_hat) / (q * Bz)
        R = torch.cross(momentum, B_hat, dim=1) / (charge * self.Bz) *self.r_scale # Shape: (N, 3)
        #print(R)
        R_length,_ = self.safe_normalize(R)
        #print(R_length)
        #center of the circle in x-y plane
        Center = position + R
        #project to x-y plane
        z_projection_shift = position[..., 2:3]
        Center[..., 2] = 0
        #perigee lie along the the line between beam axis and circle center
        _, C_hat = self.safe_normalize(Center)

        #angular momentum direction for pT
        omega_hat = -charge_sign * B_hat

        pt = momentum.clone()
        pt[...,2] = 0
        pt_length, pt_hat = self.safe_normalize(pt)
        #propagated pT
        ptp_hat = torch.cross(C_hat, omega_hat, dim=1) #(N,3)
        ptp = pt_length * ptp_hat #(N,3)

        cos_theta = torch.sum(pt_hat * ptp_hat, dim=1, keepdim=True)  # Shape: (N, 1)

        # Clamp the values to avoid numerical errors in acos
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.acos(cos_theta)  # Shape: (N, 1)
        #print(theta)

        theta_sign = torch.sign(torch.sum(torch.cross(pt_hat, ptp_hat, dim=1) * omega_hat, dim=1, keepdim=True))  # Shape: (N, 1)
        #print(theta_sign)

        z_propagation_shift = (pz / pt_length) * (theta * theta_sign) * R_length  # Ensure all terms are (N, 1)
        #propogated vertex, without Z propagation shift
        v_p = Center - C_hat * R_length
        v_p = torch.cat([v_p[..., :2], v_p[..., 2:] + z_projection_shift + z_propagation_shift], dim=1)
        p_p = ptp
        p_p = torch.cat([ptp[..., :2], ptp[..., 2:] + pz], dim=1)  # Shape: (N, 3)

        ppars = torch.cat([v_p, p_p, charge], dim=1)  # Shape: (N, 7)

        return ppars
    
def calculate_basis_vectors(momentum):
    """
    Given the particle momentum direction, calculate the basis vectors for a perigee surface at (0, 0, 0),
    ensuring they are unit vectors with a NaN guard.

    Args:
        momentum (torch.Tensor): Input tensor of shape (N, 3), where each row is the normalized momentum (px, py, pz)/p.

    Returns:
        tuple: Three tensors of shape (N, 3), representing the unit basis vectors.
    """
    # Calculate unit components
    upx = momentum[..., 0:1]  # Shape: (N, 1)
    upy = momentum[..., 1:2]  # Shape: (N, 1)

    # Create the basis vectors (unnormalized)
    basis_D = torch.cat([-upy, upx, torch.zeros_like(upx)], dim=1)  # Shape: (N, 3)
    basis_Z = torch.cat([torch.zeros_like(upx), torch.zeros_like(upx), torch.ones_like(upx)], dim=1)  # Shape: (N, 3)
    basis_shift = torch.cat([upx, upy, torch.zeros_like(upx)], dim=1)  # Shape: (N, 3)

    # Normalize the basis vectors to ensure they are unit vectors
    def safe_normalize(vectors):
        norms = torch.norm(vectors, dim=1, keepdim=True)  # Calculate norms
        norms = torch.where(norms == 0, torch.tensor(1e-12, device=vectors.device, dtype=vectors.dtype), norms)  # Prevent div by 0
        return vectors / norms

    basis_D = safe_normalize(basis_D)
    basis_Z = safe_normalize(basis_Z)
    basis_shift = safe_normalize(basis_shift)

    return basis_D, basis_Z, basis_shift

def free_to_bound(input_data):
    """
    Given the particle momentum direction, calculate projections for a perigee surface at (0,0,0),
    spherical angles (phi, theta), and q/p.

    Args:
        input_data (torch.Tensor): Input tensor of shape (N, 7), where each row is (x, y, z, px, py, pz, q).

    Returns:
        torch.Tensor: A tensor containing bound parameters (D, Z, shift, phi, theta, q/p), shape (N, 6).
    """
    # Split position and momentum
    position = input_data[..., :3]  # Shape: (N, 3)
    momentum = input_data[..., 3:6]  # Shape: (N, 3)
    charge = input_data[..., 6:]  # Shape: (N, 1)

    # Calculate the magnitude of momentum
    p = torch.norm(momentum, dim=1, keepdim=True)  # Shape: (N, 1)
    p = torch.where(p == 0, torch.tensor(1e-12, device=momentum.device), p)  # Avoid division by zero

    qp = charge / p  # Calculate q/p

    # Calculate unit components
    u_momentum = momentum / p

    # Get basis vectors
    basis_D, basis_Z, basis_shift = calculate_basis_vectors(u_momentum)

    # Extract unit momentum components
    upx = u_momentum[..., 0:1]  # Shape: (N, 1)
    upy = u_momentum[..., 1:2]  # Shape: (N, 1)
    upz = u_momentum[..., 2:3]  # Shape: (N, 1)

    # Calculate spherical angles phi and theta
    phi = torch.atan2(upy, upx)  # Azimuthal angle, Shape: (N, 1)
    theta = torch.acos(upz)      # Polar angle, Shape: (N, 1)

    # Project position onto the basis
    D_projection = torch.sum(position * basis_D, dim=1, keepdim=True)  # Shape: (N, 1)
    Z_projection = torch.sum(position * basis_Z, dim=1, keepdim=True)  # Shape: (N, 1)
    shift_projection = torch.sum(position * basis_shift, dim=1, keepdim=True)  # Shape: (N, 1)

    #position_back = (D_projection * basis_D) + (Z_projection * basis_Z) + (shift_projection * basis_shift)  # Shape: (N, 3)
    #print(position_back)

    # Concatenate all bound parameters into a single tensor
    bound_parameters = torch.cat([D_projection, Z_projection, shift_projection, phi, theta, qp], dim=1)  # Shape: (N, 6)

    return bound_parameters

def bound_to_free(bound_parameters):
    """
    Convert bound parameters (D, Z, shift, phi, theta, q/p) to free parameters (x, y, z, px, py, pz, q).

    Args:
        bound_parameters (torch.Tensor): Input tensor of shape (N, 6), where each row is
                                          (D, Z, shift, phi, theta, q/p).

    Returns:
        torch.Tensor: A tensor of shape (N, 7), where each row is (x, y, z, px, py, pz, q).
    """
    # Split the bound parameters
    D = bound_parameters[..., 0:1]  # Shape: (N, 1)
    Z = bound_parameters[..., 1:2]  # Shape: (N, 1)
    shift = bound_parameters[..., 2:3]  # Shape: (N, 1)
    phi = bound_parameters[..., 3:4]  # Shape: (N, 1)
    theta = bound_parameters[..., 4:5]  # Shape: (N, 1)
    qp = bound_parameters[..., 5:6]  # Shape: (N, 1)

    # Recover momentum (px, py, pz) from q/p, phi, and theta
    p = 1.0 / torch.abs(qp)  # Magnitude of momentum
    q = torch.sign(qp)  # Charge (sign of q/p)

    px = p * torch.cos(phi) * torch.sin(theta)  # Shape: (N, 1)
    py = p * torch.sin(phi) * torch.sin(theta)  # Shape: (N, 1)
    pz = p * torch.cos(theta)                   # Shape: (N, 1)
    momentum = torch.cat([px, py, pz], dim=1)   # Shape: (N, 3)

    u_momentum = momentum / p

    # Get basis vectors
    basis_D, basis_Z, basis_shift = calculate_basis_vectors(u_momentum)

    # Recover position (x, y, z) using projections
    position = (D * basis_D) + (Z * basis_Z)
      #+ (shift * basis_shift)  # Shape: (N, 3)

    # Concatenate position, momentum, and charge to form free parameters
    free_parameters = torch.cat([position, momentum, q], dim=1)  # Shape: (N, 7)

    return free_parameters

def designate_layer(pn, p, layer = 'point_encoder', n_layer = 3, init_std = 0.025, mup_width_multiplier = 1.0):
    # Adjust hidden weight initialization variance by 1 / mup_width_multiplier
    if layer in pn and (pn.endswith('in_proj_weight') or ('mlp.0.weight' in pn)):
        torch.nn.init.normal_(p, mean=0.0, std=init_std)

    if layer in pn and (pn.endswith('out_proj.weight') or ('mlp.2.weight' in pn)):
        torch.nn.init.normal_(p, mean=0.0, std=init_std / math.sqrt(2 * n_layer * mup_width_multiplier))

def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.025 / math.sqrt(2))
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.025 / math.sqrt(2))
    
def cartesian_to_eta_phi_r(x, y, z):
    #r (transverse distance)
    r = np.sqrt(x**2 + y**2)
    #phi (azimuthal angle)
    phi = np.arctan2(y, x)
    #eta (Pseudorapidity)
    eta = np.arctanh(z / np.sqrt(x**2 + y**2 + z**2))
    return np.array([eta, phi, r])

def cartesian_to_polar_batched(arr):
    """
    Batched version of cartesian to polar coordinate.
    arr: (B x 3), where axis=1 corresponds to (x,y,z)
    """
    #r (transverse distance)
    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]
    r = torch.sqrt(x**2 + y**2)
    #phi (azimuthal angle)
    phi = torch.arctan2(y, x)
    #eta (Pseudorapidity)
    eta = torch.arctanh(z / np.sqrt(x**2 + y**2 + z**2))
    return torch.stack([eta, phi, r], axis=-1)

def unit_test_on_batched(arr):
    ## Non-batch version
    nonbatch = []
    for x, y, z in arr:
        nonbatch.append(cartesian_to_eta_phi_r(x, y, z))
    nonbatch = np.stack(nonbatch)
    
    ## batch version
    batched = cartesian_to_polar_batched(arr)
    
    assert np.allclose(nonbatch, batched)
    print('unit test of batched version of cart to polar coord successful!')
    return 0

def parse_mean_E(E, target, stat='mean'):
    """
    Get an array of length target, but with a stat computed with E
    E: (lenseq)
    target: (lenseq)
    stat: ['mean' or 'std']
    """
    out = torch.zeros(E.shape)
    for u in torch.unique(target):
        out[target == u] = E[target == u].mean() if stat == 'mean' else E[target == u].std()
    return out


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def z_order_decode(code: torch.Tensor, depth):
    print(code, len(z_order_decode_(code, depth=depth)))
    x, y, z = z_order_decode_(code, depth=depth)
    grid_coord = torch.stack([x, y, z], dim=-1)  # (N,  3)
    return grid_coord


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def hilbert_decode(code: torch.Tensor, depth: int = 16):
    return hilbert_decode_(code, num_dims=3, num_bits=depth)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def makedir(addr):
    if not os.path.exists(addr):
        os.makedirs(addr)
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def save_checkpoint(addr, model, optimizer, epoch):
    
    dir_ = '/'.join(addr.split('/')[:-1])
    makedir(dir_)
    
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    optim_dict = optimizer.state_dict()
    
    lr = get_lr(optimizer)
    
    torch.save({
        'model': model_state_dict,
        'optim': optim_dict,
        'epoch': epoch,
        'lr': lr,
    }, addr)
    print('Checkpoint is saved at %s' % addr)

    
def load_checkpoint(path, name, model):
    checkpoint = torch.load(os.path.join(path, name))
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    
    print('Checkpoint is loaded from {}, starting epoch: {}'.format(os.path.join(path, name), epoch+1))
    
    return epoch


class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
       arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """

    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None):
        super(PositionEmbeddingCoordsSine, self).__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb
    
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)
        
    
class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device, dtype=tensor.dtype)
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc
    
    
    
def masked_loss(arr, target, mask):
    """Deprecated"""
    loss = loss_func_c(arr, target)
    non_zero_elements = mask.sum()
    out = loss / non_zero_elements
    raise RuntimeError
    return out

def maskout(arr, mask):
    B, G, P, C = arr.size()
    return arr.reshape(-1, P, C)[mask[..., 1:].squeeze(1).reshape(-1).bool()]