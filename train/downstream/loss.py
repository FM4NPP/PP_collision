import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score



#calculate precision and recall
def compute_multiclass_metrics(outputs, targets, average='macro'):
    """
    Computes precision, recall, and accuracy for multi-class classification using argmax predictions.

    Args:
        outputs (dict): contains 'pred_logits' of shape (B, N, C)
        targets (dict): contains 'labels' of shape (B, N)
        average (str): 'macro', 'micro', or None for per-class scores

    Returns:
        precision (float or array)
        recall (float or array)
        accuracy (float)
    """
    pred_logits = outputs['pred_logits']  # (B, N, C)
    labels = targets['labels']            # (B, N)

    # Predict class using argmax over class dimension
    pred_classes = torch.argmax(pred_logits, dim=-1)  # (B, N)

    # Flatten to 1D
    pred_flat = pred_classes.view(-1).cpu().numpy()
    true_flat = labels.view(-1).cpu().numpy()

    # Compute metrics
    precision = precision_score(true_flat, pred_flat, average=average, zero_division=0)
    recall = recall_score(true_flat, pred_flat, average=average, zero_division=0)
    accuracy = accuracy_score(true_flat, pred_flat)

    return precision, recall, accuracy


def compute_masked_cross_entropy_loss(logits, targets, valid_mask):
    """
    Args:
        logits: (B, N, C) or (N, C) — raw class scores (before softmax)
        targets: (B, N) or (N,) — ground truth class labels
        valid_mask: (B, N) or (N,) — boolean or 0/1 tensor indicating valid points

    Returns:
        Scalar: averaged cross entropy loss over valid points
    """
    if logits.dim() == 3:
        B, N, C = logits.shape
        logits = logits.view(-1, C)              # (B*N, C)
        targets = targets.view(-1)               # (B*N,)
        valid_mask = valid_mask.view(-1) > 0     # (B*N,)
    elif logits.dim() == 2:
        # unbatched: (N, C)
        logits = logits
        targets = targets
        valid_mask = valid_mask > 0
    else:
        raise ValueError(f"Unsupported logits shape: {logits.shape}")

    if valid_mask.sum() == 0:
        return torch.tensor(0., device=logits.device)

    loss = F.cross_entropy(logits[valid_mask], targets[valid_mask], reduction='mean')
    return loss



def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, eps=1e-6):
    """
    Args:
        inputs: (B, N_pred, N_points) - Predicted point probabilities for each prototype
        targets: (B, N_gt, N_points) - Ground truth one-hot encoded masks
        mask: (B, N_points) - Valid points mask (1=valid, 0=padding)
    Returns:
        (B, N_pred, N_gt) - Dice loss matrix between all prediction-target pairs
    """

    inputs = inputs * mask.unsqueeze(1)  # (B, N_pred, N_points)
    targets = targets * mask.unsqueeze(1)          # (B, N_gt, N_points)

    # Compute pairwise intersection: (B, N_pred, N_gt)
    #intersection = torch.sum(inputs.unsqueeze(2) * targets.unsqueeze(1), dim=-1)
    intersection = torch.bmm(inputs, targets.transpose(1, 2))
    
    # Compute sum for each prediction and target
    pred_sum = torch.sum(inputs, dim=-1)         # (B, N_pred)
    target_sum = torch.sum(targets, dim=-1)      # (B, N_gt)

    # Dice coefficient calculation
    dice = (2 * intersection + eps) / (
        pred_sum.unsqueeze(-1) +    # (B, N_pred, 1)
        target_sum.unsqueeze(1) +   # (B, 1, N_gt)
        eps
    )
    return 1 - dice  # (B, N_pred, N_gt)

def matched_dice_loss(
    src_masks: torch.Tensor,  # (M, P) - Matched predictions
    tgt_masks: torch.Tensor,  # (M, P) - Matched targets
    valid: torch.Tensor,      # (M, P) - Valid points mask
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute DICE loss for pre-matched pairs.
    
    Args:
        src_masks: Predicted masks for matched pairs (M, P)
        tgt_masks: Ground truth masks for matched pairs (M, P)
        valid: Valid points indicator (M, P)
        eps: Epsilon for numerical stability
    
    Returns:
        Scalar DICE loss averaged over matches
    """
    # Apply valid mask to both predictions and targets
    src_masks = src_masks * valid
    tgt_masks = tgt_masks * valid

    # Calculate intersection and sums
    intersection = (src_masks * tgt_masks).sum(dim=1)  # (M,)
    pred_sum = src_masks.sum(dim=1)                    # (M,)
    target_sum = tgt_masks.sum(dim=1)                 # (M,)

    # Compute DICE coefficient per match
    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)  # (M,)
    
    # Convert to loss and average
    return (1 - dice).sum()  # Scalar

def batch_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, alpha=0.25, gamma=2):
    """
    Args:
        inputs: (B, N_pred, N_points) - Predicted point probs
        targets: (B, N_gt, N_points) - Ground truth binary masks
        mask: (B, N_points) - Valid points mask
    Returns:
        (B, N_pred, N_gt) - Focal loss matrix between all prediction-target pairs
    """
    # padding mask
    inputs = inputs * mask.unsqueeze(1)   # (B, N_pred, N_points)
    targets = targets * mask.unsqueeze(1) # (B, N_gt, N_points)

    prob = inputs # (B, N_pred, N_points)
    
    # Focal loss components
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy(
        inputs, torch.ones_like(inputs), reduction="none"
    )  # (B, N_pred, N_points)
    
    focal_neg = (prob ** gamma) * F.binary_cross_entropy(
        inputs, torch.zeros_like(inputs), reduction="none"
    )  # (B, N_pred, N_points)

    # Apply alpha weighting
    if alpha >= 0:
        focal_pos = alpha * focal_pos
        focal_neg = (1 - alpha) * focal_neg

    # Compute pairwise loss matrix
    pos_loss = torch.einsum("bnp,bgp->bng", focal_pos, targets)  # (B, N_pred, N_gt)
    neg_loss = torch.einsum("bnp,bgp->bng", focal_neg, 1 - targets)  # (B, N_pred, N_gt)
    
    total_loss = pos_loss + neg_loss
    
    # Normalize by valid points per batch element
    #print(total_loss.shape, mask.sum(dim=-1, keepdim=True).shape)
    return total_loss / mask.sum(dim=-1, keepdim=True).unsqueeze(-1)

def matched_focal_loss(
    src_masks: torch.Tensor,  # (M, P) - Matched predictions
    tgt_masks: torch.Tensor,  # (M, P) - Matched targets
    valid: torch.Tensor,      # (M, P) - Valid points
    alpha: float = 0.25,
    gamma: float = 2
) -> torch.Tensor:
    """
    Compute focal loss for already matched pairs.
    
    Args:
        src_masks: Predicted masks for matched pairs (M, P)
        tgt_masks: Ground truth masks for matched pairs (M, P)
        valid: Valid points indicator (M, P)
    
    Returns:
        Scalar focal loss
    """
    # Apply valid mask
    src_masks = src_masks * valid
    tgt_masks = tgt_masks * valid

    # Flatten to (M, P)
    src_masks = src_masks.flatten(1)
    tgt_masks = tgt_masks.flatten(1)

    # Compute probabilities
    prob = src_masks

    # Focal loss components
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy(
        prob, torch.ones_like(prob), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy(
        prob, torch.zeros_like(prob), reduction="none"
    )

    # Apply alpha weighting
    if alpha >= 0:
        focal_pos = alpha * focal_pos
        focal_neg = (1 - alpha) * focal_neg

    # Compute loss per element
    loss = focal_pos * tgt_masks + focal_neg * (1 - tgt_masks) # (M, P)
    
    # Normalize by valid points and sum over matches
    valid_points = valid.sum(dim=1).clamp(min=1)  # (M,)
    normalized_loss = (loss.sum(dim=1) / valid_points).sum()

    return normalized_loss


class PointHungarianMatcher(torch.nn.Module):
    def __init__(self, cost_class=1, cost_dice=1, cost_focal=20):
        super().__init__()
        self.cost_class = cost_class  # Classification cost (if applicable)
        self.cost_dice = cost_dice    # Dice loss weight
        self.cost_focal = cost_focal  # Focal loss weight

    @torch.no_grad()
    def forward_backup(self, outputs, targets, mask):
        """
        Args:
            outputs: Dict with "pred_probs" (B, N_pred, C) and "pred_masks" (B, N_pred, N_points)
            targets: Dict with "labels" (B, N_gt) and "masks" (B, N_gt, N_points)
            mask: (B, N_points) indicating valid points
        """
        bs, n_pred = outputs["pred_probs"].shape[:2]
        device = outputs["pred_probs"].device
        # Initialize cost matrix (B, N_pred, N_gt)
        cost_matrix = torch.zeros(bs, n_pred, len(targets["labels"][0]), device=device)

        # Classification cost (B, N_pred, N_gt)
        if self.cost_class != 0:
            class_probs = outputs["pred_probs"]  # (B, N_pred, C)
            gt_labels = targets["labels"]  # (B, N_gt)
            
            # Expand labels to match pred_probs dimensions
            gt_labels_expanded = gt_labels.unsqueeze(1).expand(-1, class_probs.shape[1], -1)  # (B, N_pred, N_gt)
            
            # Gather probabilities for target classes
            cost_class = -torch.gather(
                class_probs, 
                dim=2, 
                index=gt_labels_expanded
            )  # (B, N_pred, N_gt)
            
            cost_matrix += self.cost_class * cost_class

        # Mask costs (B, N_pred, N_gt)
        if self.cost_dice != 0:
            cost_dice = batch_dice_loss(
                outputs["pred_masks"],  # (B, N_pred, N_points)
                targets["masks"],       # (B, N_gt, N_points)
                mask                    # (B, N_points)
            )
            cost_matrix += self.cost_dice * cost_dice

        if self.cost_focal != 0:
            cost_focal = batch_focal_loss(
                outputs["pred_masks"],  # (B, N_pred, N_points)
                targets["masks"],       # (B, N_gt, N_points)
                mask                    # (B, N_points)
            )
            cost_matrix += self.cost_focal * cost_focal

        # Hungarian matching per batch element
        #print(cost_matrix.shape)
        indices = [linear_sum_assignment(c.cpu()) for c in cost_matrix]
        return [
            (torch.as_tensor(i, dtype=torch.int64, device=device),
             torch.as_tensor(j, dtype=torch.int64, device=device))
            for i, j in indices
        ]

    
    @torch.no_grad()
    def forward(self, outputs, targets, mask):
        """
        Args:
            outputs: Dict with "pred_probs" (B, N_pred, C) and "pred_masks" (B, N_pred, N_points)
            targets: List of Dict with "labels" (N_gt) and "masks" (N_gt, N_points)
            mask: (B, N_points) indicating valid points
        """
        bs = outputs["pred_probs"].shape[0]
        device = outputs["pred_probs"].device
        indices = []
        #print(outputs)
        #print(targets)
        #print(mask)

        for b in range(bs):
            # Extract per-batch data (single element from batch)
            pred_probs = outputs["pred_probs"][b]  # (N_pred, C)
            pred_masks = outputs["pred_masks"][b]  # (N_pred, N_points)
            gt_labels = targets[b]["labels"]       # (N_gt,)
            gt_masks = targets[b]["masks"]        # (N_gt, N_points)
            batch_mask = mask[b]                   # (N_points,)

            n_pred = pred_probs.shape[0]
            n_gt = gt_labels.shape[0]

            cost_matrix = torch.zeros(n_pred, n_gt, device=device)
            # Classification cost
            # Directly gather probabilities for GT classes
            cost_class = -pred_probs[:, gt_labels]  # (N_pred, N_gt)
            cost_matrix += self.cost_class * cost_class

            # Mask costs
            # Add batch dimension (B=1) to match loss function expectations
            pred = pred_masks.unsqueeze(0)    # (1, N_pred, N_points)
            gt = gt_masks.unsqueeze(0)        # (1, N_gt, N_points)
            valid_mask = batch_mask.unsqueeze(0)  # (1, N_points)
            if self.cost_dice != 0:
                dice_cost = batch_dice_loss(
                    pred, gt, valid_mask
                ).squeeze(0)  # Remove batch dim -> (N_pred, N_gt)
                cost_matrix += self.cost_dice * dice_cost
            if self.cost_focal != 0:
                focal_cost = batch_focal_loss(
                    pred, gt, valid_mask
                ).squeeze(0)  # Remove batch dim -> (N_pred, N_gt)
                cost_matrix += self.cost_focal * focal_cost
            
            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu())
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64, device=device),
                torch.as_tensor(col_ind, dtype=torch.int64, device=device)
            ))

        return indices


def focal_loss_mc(logits: torch.Tensor,
                  targets: torch.Tensor,
                  gamma: float = 2.0,
                  alpha: torch.Tensor | None = None,
                  reduction: str = "mean") -> torch.Tensor:
    """
    Multi-class (single-label) focal loss.

    Args
    ----
    logits     : (N, C)  raw scores from the network
    targets    : (N,)    int64 class indices 0 … C-1
    gamma      : focusing parameter (γ = 0 → CE)
    alpha      : None | (C,) tensor of per-class weights
    reduction  : 'mean' | 'sum' | 'none'

    Returns
    -------
    Tensor scalar if reduced else (N,) tensor of per-sample losses
    """
    log_probs = F.log_softmax(logits, dim=-1)        # (N, C)
    probs     = log_probs.exp()                      # (N, C)

    # Probabilities of the true class for each sample
    idx  = torch.arange(logits.size(0), device=logits.device)
    p_t  = probs[idx, targets]                       # (N,)
    log_p_t = log_probs[idx, targets]                # (N,)

    # Class weighting
    if alpha is not None:
        alpha_t = alpha.to(logits.device)[targets]   # (N,)
    else:
        alpha_t = 1.0

    # Focal term
    focal_weight = (1.0 - p_t) ** gamma

    loss = -alpha_t * focal_weight * log_p_t         # (N,)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss




    
def simple_point_loss(outputs, targets, mask, class_weights = None, option = "ce"):
    """
    Args:
        outputs: {
            'pred_logits': (B, N, C)  # raw logits (not softmax)
        }
        targets: {
            'labels': (B, N)  # ground truth class indices
        }
        mask: (B, N)  # boolean or 0/1 mask indicating valid points
        class_weights: (C,)  # class weights for CE loss
        option: str, either "ce" for cross-entropy or "focal" for focal loss
    Returns:
        dict: {
            'loss': classification loss (scalar)
        }
    """

    pred_logits = outputs['pred_logits']  # (B, N, C)
    labels = targets['labels']  # (B, N)

    # Flatten everything to (B*N, ...)
    B, N, C = pred_logits.shape
    pred_logits = pred_logits.reshape(-1, C)
    labels = labels.reshape(-1)
    mask = mask.reshape(-1).bool()

    # Apply the mask to select only valid points
    pred_logits = pred_logits[mask]  # (M, C)
    labels = labels[mask]            # (M,)

    # If nothing is valid, return zero loss (avoid division by zero)
    if pred_logits.shape[0] == 0:
        return {'loss': torch.tensor(0.0, device=pred_logits.device)}

    loss = torch.tensor(0.0, device=pred_logits.device)
    # Compute weighted CE loss
    if option == "focal":
        # Focal loss implementation
        loss = focal_loss_mc(pred_logits, labels, alpha = torch.tensor([0.25, 0.75], device=pred_logits.device))
    else:
        loss = F.cross_entropy(pred_logits, labels, weight=class_weights, reduction='mean')

    return {'loss': loss}
    




def compute_point_loss(outputs, targets, mask, matcher, no_object_class=0):
    """
    Args:
        outputs: {
            "pred_probs": (B, N_pred, C),  # Class probabilities (softmax output)
            "pred_masks": (B, N_pred, N_points)  # Mask probabilities (sigmoid output)
            'track_reg_result': track_reg_result, # (B, C, num_track_features)
            'pid_probs': pid_probs, # (B, C, num_pid_classes + 1)
            'noise_logits': noise_logits, # (B, N, 2)
        }
        targets: list of dicts, each with: {
            "labels": (N_gt),  # Class labels for GT instances
            "masks": (N_gt, N_points)  # Binary GT masks
            # Optional: "track_info": (N_gt, num_track_features)  # Track info for regression
            # Optional: "valid_tracks" : (N_gt)  # Valid track info indicator
            # Optional: "pid_labels": (N_gt)  # PIDs for classification
            # Optional: "noise_labels": (N,)  # Noise labels for classification
        }
        mask: (B, N_points)  # Valid points mask
        
        matcher: PointHungarianMatcher instance
    Returns:
        Dict with classification, dice, and focal losses
    """
    device = outputs["pred_probs"].device
    B, N_pred, C = outputs["pred_probs"].shape
    
    # 1. Perform Hungarian matching and get global indices
    indices = matcher(outputs, targets, mask)
    src_batch_idx, src_idx = _get_src_permutation_idx(indices)
    tgt_batch_idx, tgt_idx = _get_tgt_permutation_idx(indices)
    num_matched = len(src_batch_idx)
    # 2. Classification Losses ================================================
    #target_classes = torch.full((B, N_pred), no_object_class, 
                              #dtype=torch.int64, device=device)
    loss_matched_ce = torch.tensor(0., device=device)
    loss_unmatched_ce = torch.tensor(0., device=device)

    #optional perpoint noise classification loss
    #first pad the input to (B, N, 1)
    if "noise_probs" in outputs:
        noise_logits = outputs["noise_logits"] # (B, N, 2)
        noise_gt = torch.stack([t["noise_labels"] for t in targets], dim=0) # (B, N)
        loss_noise_ce = compute_masked_cross_entropy_loss(noise_logits, noise_gt, mask)
    else:
        loss_noise_ce = torch.tensor(0., device=device)


    # Matched classification loss
    if num_matched > 0:
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # (num_matched_total,)
        #target_classes[src_batch_idx, src_idx] = target_classes_o
        
        matched_probs = outputs["pred_probs"][src_batch_idx, src_idx] #(num_matched_total, C)
        loss_matched_ce = F.nll_loss(torch.log(matched_probs + 1e-6), 
                                   target_classes_o, reduction='sum')
    
        # optional: matched track info regression loss
        
        if all("track_info" in t for t in targets):
            target_track_info = torch.cat([t["track_info"][J] for t, (_, J) in zip(targets, indices)]) #(num_matched_total, num_track_features)
            pred_track_info = outputs["track_reg_result"][src_batch_idx, src_idx] #(num_matched_total, num_track_features)
            valid_track = torch.cat([t["valid_tracks"][J] for t, (_, J) in zip(targets, indices)]) #(num_matched_total,)
            # Apply valid track mask
            if valid_track.any():
                pred_track_info = pred_track_info[valid_track]
                target_track_info = target_track_info[valid_track]
            loss_track_reg = F.l1_loss(pred_track_info, target_track_info, reduction='sum') #(num_matched_total, num_track_features)
        else:
            loss_track_reg = torch.tensor(0., device=device)

        # Matched PID classification loss (Cross Entropy)
        if all("pid_labels" in t for t in targets):
            target_pid_labels = torch.cat([t["pid_labels"][J] for t, (_, J) in zip(targets, indices)]) #(num_matched_total,)
            pred_pid_probs = outputs["pid_probs"][src_batch_idx, src_idx]  #(num_matched_total, num_pid_classes + 1) 
            loss_pid_ce = F.nll_loss(torch.log(pred_pid_probs + 1e-6), 
                                     target_pid_labels, reduction='sum')
        else:
            loss_pid_ce = torch.tensor(0., device=device)
        


    # Unmatched classification loss
    src_mask = torch.zeros((B, N_pred), dtype=torch.bool, device=device)
    src_mask[src_batch_idx, src_idx] = True
    if (~src_mask).any():
        unmatched_probs = outputs["pred_probs"][~src_mask]
        loss_unmatched_ce = F.nll_loss(torch.log(unmatched_probs + 1e-6),
                                     torch.full((unmatched_probs.size(0),), 
                                              no_object_class, device=device),
                                     reduction='sum')

    # 3. Mask Losses =========================================================
    loss_dice = torch.tensor(0., device=device)
    loss_focal = torch.tensor(0., device=device)
    
    if num_matched > 0:
        # Gather matched pairs across all batches
        src_masks = outputs["pred_masks"][src_batch_idx, src_idx]  # (M, P)
        tgt_masks = torch.cat([t["masks"][i] for t, (_, i) in zip(targets, indices)]).to(device)
        #tgt_masks = torch.cat([t["masks"] for t in targets])[tgt_batch_idx, tgt_idx]
        valid = mask[src_batch_idx]  # (M, P)
        #print(src_masks.shape, tgt_masks.shape, valid.shape)

        # Compute losses
        loss_dice = matched_dice_loss(src_masks, tgt_masks, valid)
        loss_focal = matched_focal_loss(src_masks, tgt_masks, valid)



    # 4. Normalize classification losses
    num_total = B * N_pred  # Or use num_matched for matched_ce
    return {
        "loss_matched_ce": loss_matched_ce /B,
        "loss_unmatched_ce": loss_unmatched_ce / B,
        "loss_dice": loss_dice /B,
        "loss_focal": loss_focal /B,
        "loss_track_reg": loss_track_reg / B,
        "loss_pid_ce": loss_pid_ce / B,
        "loss_noise_ce": loss_noise_ce #this is normalized by all points in batch already so
    }

# Helper functions ============================================================
def _get_src_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def _get_tgt_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx








def assign_points_to_masks(outputs, no_object_class=0, option=1, threshold=0.0):
    """
    Assign points to masks using either:
    1. Maximum mask probability
    2. Maximum (mask prob × class prob) excluding no-object class
    
    Args:
        outputs: {
            "pred_probs": (B, N_pred, C),  # Class probabilities
            "pred_masks": (B, N_pred, N_points)  # Mask probabilities
        }
        no_object_class: Background class index
        option: 1 or 2 for strategy selection
        threshold: Minimum score for valid assignment
        
    Returns:
        assignments: (B, N_points) tensor of mask indices (-1 for no assignment)
        classes: (B, N_points) tensor of class labels
    """
    device = outputs["pred_probs"].device
    B, N_pred, _ = outputs["pred_probs"].shape
    N_points = outputs["pred_masks"].shape[-1]
    
    all_assignments = []
    all_classes = []

    for b in range(B):
        pred_probs = outputs["pred_probs"][b]  # (N_pred, C)
        pred_masks = outputs["pred_masks"][b]  # (N_pred, N_points)
        
        # Get maximum class probabilities and labels
        max_class_probs, max_classes = torch.max(pred_probs, dim=-1)  # (N_pred,)

        if option == 1:
            # Option 1: Assign based on mask probability only
            mask_scores = pred_masks
            max_scores, max_indices = torch.max(mask_scores, dim=0)  # (N_points,)
            
        elif option == 2:
            # Option 2: (mask prob × max class prob) excluding no-object
            valid_mask = max_classes != no_object_class
            adjusted_probs = max_class_probs.clone()
            adjusted_probs[~valid_mask] = -1  # Exclude no-object class
            
            # Calculate combined scores
            mask_scores = pred_masks * adjusted_probs.unsqueeze(-1)  # (N_pred, N_points)
            max_scores, max_indices = torch.max(mask_scores, dim=0)  # (N_points,)
            
        else:
            raise ValueError(f"Invalid option {option} - must be 1 or 2")

        # Create assignments and class labels
        assignments = torch.full((N_points,), -1, dtype=torch.long, device=device)
        class_labels = torch.full((N_points,), no_object_class, dtype=torch.long, device=device)
        
        # Apply threshold
        valid_points = max_scores > threshold
        valid_indices = max_indices[valid_points]
        
        assignments[valid_points] = valid_indices
        class_labels[valid_points] = max_classes[valid_indices]

        all_assignments.append(assignments)
        all_classes.append(class_labels)

    return {
        "assignments": torch.stack(all_assignments),  # (B, N_points)
        "classes": torch.stack(all_classes)           # (B, N_points)
    }


# this is serves as a test and example

def test_point_loss():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test configuration
    B = 2  # Batch size
    N_pred = 5  # Predictions per batch
    N_gt = 3  # Ground truths per batch
    N_points = 10  # Points per sample
    C = 2  # Number of classes (0: no-object, 1: object)
    device = torch.device("cpu")
    no_object_class = 0

    # Generate synthetic data -------------------------------------------------
    # Model outputs (probabilities)
    outputs = {
        "pred_probs": torch.softmax(torch.randn(B, N_pred, C), dim=-1),  # (B, N_pred, C) 
        "pred_masks": torch.sigmoid(torch.randn(B, N_pred, N_points))
    }

    # Ground truth targets as list of dicts
    targets = [
        {
            "labels": torch.randint(1, C, (N_gt,)),  # (N_gt,)
            "masks": torch.randint(0, 2, (N_gt, N_points)).float()  # (N_gt, P)
        }
        for _ in range(B)
    ]

    # Valid points mask tests ================================================
    print("\n=== Testing with partial valid mask ===")
    # Test 1: Random valid mask (80% valid points)
    mask = torch.bernoulli(torch.full((B, N_points), 0.8))
    _run_test_case(outputs, targets, mask, "Partial valid mask")

    # Test 2: All points valid
    print("\n=== Testing with full valid mask ===")
    full_mask = torch.ones(B, N_points)
    _run_test_case(outputs, targets, full_mask, "Full valid mask")

    # Test 3: No valid points (edge case)
    #print("\n=== Testing with empty valid mask ===")
    #empty_mask = torch.zeros(B, N_points)
    #_run_test_case(outputs, targets, empty_mask, "Empty valid mask")

def _run_test_case(outputs, targets, mask, case_name):
    device = outputs["pred_probs"].device
    matcher = PointHungarianMatcher(cost_class=1, cost_dice=1, cost_focal=20)
    
    print(f"\nRunning test case: {case_name}")
    
    # Convert to device
    outputs = {k: v.to(device) for k, v in outputs.items()}
    targets = [
        {k: v.to(device) for k, v in t.items()}
        for t in targets
    ]
    mask = mask.to(device)

    # Compute losses
    losses = compute_point_loss(
        outputs=outputs,
        targets=targets,
        mask=mask,
        matcher=matcher,
        no_object_class=0
    )

    # Basic validation
    assert losses["loss_matched_ce"].dim() == 0, "CE loss should be scalar"
    assert losses["loss_dice"].dim() == 0, "Dice loss should be scalar"
    assert losses["loss_focal"].dim() == 0, "Focal loss should be scalar"
    assert not torch.isnan(losses["loss_dice"]), "Dice loss NaN"
    assert not torch.isnan(losses["loss_focal"]), "Focal loss NaN"

    # Special handling for empty mask
    if mask.sum() == 0:
        assert losses["loss_dice"] == 0, "Dice should be 0 with no valid points"
        assert losses["loss_focal"] == 0, "Focal should be 0 with no valid points"
    else:
        assert losses["loss_dice"] >= 0, "Dice loss negative"
        assert losses["loss_focal"] >= 0, "Focal loss negative"

    print(f"{case_name} losses:", {k: v.item() for k, v in losses.items()})

    # Perfect prediction test
    if mask.sum() > 0:  # Skip if no valid points
        print(f"\nTesting perfect predictions for {case_name}")
        perfect_outputs = {
            "pred_probs": torch.zeros_like(outputs["pred_probs"]),
            "pred_masks": torch.zeros_like(outputs["pred_masks"])
        }

        # Set perfect matches for each batch
        for b in range(len(targets)):
            # Get valid mask for this batch
            valid = mask[b].bool()  # (N_points,)
            num_gt = len(targets[b]["labels"])

            # Match first N_gt predictions to GT
            # Copy only valid regions from GT
            gt_masks = targets[b]["masks"]  # (N_gt, N_points)

            for i in range(num_gt):
                # Perfect prediction for valid points
                perfect_outputs["pred_masks"][b, i, valid] = gt_masks[i, valid]

                # Introduce errors in invalid regions
                perfect_outputs["pred_masks"][b, i, ~valid] = 1 - gt_masks[i, ~valid]  # Flip values

            # Class probabilities
            perfect_outputs["pred_probs"][b, :num_gt, 1] = 1.0  # Correct class
            perfect_outputs["pred_probs"][b, num_gt:, 0] = 1.0  # No-object for others

        perfect_losses = compute_point_loss(
            outputs=perfect_outputs,
            targets=targets,
            mask=mask,
            matcher=matcher,
            no_object_class=0
        )
        print(f"{case_name} losses:", {k: v.item() for k, v in perfect_losses.items()})
        # Verify near-zero losses
        assert perfect_losses["loss_dice"] < 1e-4, f"Dice loss not perfect: {perfect_losses['loss_dice']}"
        assert perfect_losses["loss_focal"] < 1e-4, f"Focal loss not perfect: {perfect_losses['loss_focal']}"
        print(f"Perfect {case_name} losses:", {k: v.item() for k, v in perfect_losses.items()})

#test_point_loss()
