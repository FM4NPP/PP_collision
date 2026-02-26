import torch

def get_trackinfo_noiselabel(reg, noise_pt_threshold=0.06):
    """
    Extract track information from the reg tensor.
    reg: B X N X 8 tensor containing track information (
    'px': reg[..., 0],
    'py': reg[..., 1],
    'pz': reg[..., 2],
    'vtx_x': reg[..., 3],
    'vtx_y': reg[..., 4],
    'vtx_z': reg[..., 5],
    'q': reg[..., 6],
    'e': reg[..., 7],)
    Returns: dictionary of
        "track_info" : B X N X 4 tensor with track information (q/(pT + 1), theta, sin_phi, cos_phi)
        "valid_tracks" : B X N tensor indicating valid tracks (1 for valid, 0 for invalid) we require track's production vertex is within 1cm
        "noise_labels" : B X N tensor with noise labels (1 for noise, 0 for valid points)
    """
    # Extract the relevant columns from reg
    px = reg[..., 0] 
    py = reg[..., 1]
    pz = reg[..., 2]
    vtx_x = reg[..., 3]
    vtx_y = reg[..., 4]
    q = reg[..., 6]
    pt = torch.sqrt(px**2 + py**2)  # Calculate transverse momentum
    transformed_pt = q / (pt + 1)  # Add 1 to avoid division by zero
    theta = torch.atan2(pt, pz)  # Calculate theta angle
    sin_phi = py / pt  # Calculate sine of phi
    cos_phi = px / pt  # Calculate cosine of phi
    vtx_r = torch.sqrt(vtx_x**2 + vtx_y**2)  # Calculate radial distance of vertex
    valid_tracks = (vtx_r < 1.0).int()  # Check if vertex is within 1 cm radius
    noise_labels = (pt < noise_pt_threshold).long()  # Identify noise points based on transverse momentum threshold
    # Create the track_info tensor
    
    track_info = torch.stack(
        [transformed_pt, theta, sin_phi, cos_phi], 
        dim=-1
    )
    return {
        "track_info": track_info,  # B X N X 4 tensor with track information
        "valid_tracks": valid_tracks,  # B X N tensor indicating valid tracks
        "noise_labels": noise_labels  # B X N tensor with noise labels
    }

def get_pidlabel(pid):
    """
    Extract track information from the reg tensor.
    pid: B X N tensor containing particle IDs
    Returns: dictionary of
        "pid_class" : B X N tensor with particle class information (pi, k, p, e), 0 if not belong to any of these classes
    """
    # Extract the relevant columns from reg
    # pion abs(pid) == 211
    # kaon abs(pid) == 321
    # proton abs(pid) == 2212
    # electron abs(pid) == 11
    pid_class = torch.zeros_like(pid, dtype=torch.long)  # Initialize with zeros
    pid_class[pid.abs() == 211] = 1
    pid_class[pid.abs() == 321] = 2
    pid_class[pid.abs() == 2212] = 3
    pid_class[pid.abs() == 11] = 4
    return {
        "pid_class": pid_class,  # B X N tensor with particle class information, 0 if not belong to any of these classes
    }

def get_weakdecaylabel(mid):
    """
    Extract weak decay labels from the mid tensor.
    mid: B X N tensor containing mother IDs
    Returns: dictionary of
        "weak_decay_class" : B X N tensor with weak decay labels (k_0)
    """
    # Define weak decay mother IDs
    weak_decay_class = torch.zeros_like(mid, dtype=torch.long)  # Initialize with zeros
    weak_decay_class[mid == 130] = 1  # K0
    weak_decay_class[mid == 310] = 1
    return {
        "weak_decay_class": weak_decay_class,  # B X N tensor with weak decay labels (1 for K0, 0 otherwise)
    }