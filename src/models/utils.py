from typing import Tuple, List, Union
from matplotlib import scale
import torch

def scale_boxes(boxes, target_sizes):
    """
    Scale batch of bounding boxes to the target sizes.

    Args:
        boxes (`torch.Tensor` of shape `(batch_size, num_boxes, 4)`):
            Bounding boxes to scale. Each box is expected to be in (x1, y1, x2, y2) format.
        target_sizes (`list[tuple[int, int]]` or `torch.Tensor` of shape `(batch_size, 2)`):
            Target sizes to scale the boxes to. Each target size is expected to be in (height, width) format.

    Returns:
        `torch.Tensor` of shape `(batch_size, num_boxes, 4)`: Scaled bounding boxes.
    """

    if isinstance(target_sizes, (list, tuple)):
        image_height = torch.tensor([i[0] for i in target_sizes])
        image_width = torch.tensor([i[1] for i in target_sizes])
    elif isinstance(target_sizes, torch.Tensor):
        image_height, image_width = target_sizes.unbind(1)
    else:
        raise TypeError("`target_sizes` must be a list, tuple or torch.Tensor")

    scale_factor = torch.stack([image_width, image_height, image_width, image_height], dim=1)
    scale_factor = scale_factor.unsqueeze(1).to(boxes.device)
    boxes = boxes * scale_factor
    return boxes

def create_mask_from_bboxes(
    bboxes: Union[List, torch.Tensor], 
    img_resolution: int
) -> torch.Tensor:
    """
    Create a binary mask from bounding boxes.
    
    Args:
        bboxes: Input bounding boxes, either:
                - List of bounding boxes (will be converted to tensor)
                - Tensor of shape (B, K, 4) where each box is [xmin, ymin, xmax, ymax]
        img_resolution: Resolution of the square image
    
    Returns:
        Tensor of shape (B, img_resolution, img_resolution) where:
        - 1 indicates regions covered by bounding boxes
        - 0 indicates background
    """
    if isinstance(bboxes, list):
        bboxes = convert_bboxes_list_to_tensor(bboxes)
    
    if not isinstance(bboxes, torch.Tensor):
        raise TypeError("bboxes must be either list or torch.Tensor")
        
    B, K, _ = bboxes.shape
    device = bboxes.device
    
    assert bboxes.shape[2] == 4, "Bounding boxes must have shape (B, K, 4)"
    
    # Initialize mask with zeros
    mask = torch.zeros((B, img_resolution, img_resolution), 
                      dtype=torch.float32, 
                      device=device)
    
    # Vectorized implementation
    xmin = torch.clamp(bboxes[:, :, 0].long(), 0, img_resolution-1)
    ymin = torch.clamp(bboxes[:, :, 1].long(), 0, img_resolution-1)
    xmax = torch.clamp(bboxes[:, :, 2].long(), 0, img_resolution-1)
    ymax = torch.clamp(bboxes[:, :, 3].long(), 0, img_resolution-1)
    
    for b in range(B):
        for k in range(K):
            mask[b, ymin[b,k]:ymax[b,k]+1, xmin[b,k]:xmax[b,k]+1] = 1.0
            
    return mask


def extract_patch_ids_from_bboxes(
    images: torch.Tensor,  # Shape: [B, C, H, W]
    bboxes: Union[List[List[float]], torch.Tensor],  # Shape: [B, K, 4] (pixel coordinates)
    patch_size: Tuple[int, int],  # (patch_h, patch_w)
    iou_threshold: float = 0.5,  # Minimum IoU for patch inclusion
    max_condition_length: int = 1000,  # Maximum number of patches to consider
) -> torch.Tensor:
    """
    Extract patch IDs covered by bounding boxes (in pixel coordinates) for each image in batch.

    Args:
        images: Input image batch tensor [B, C, H, W]
        bboxes: Bounding box coordinates in pixels [B, K, 4] (xmin, ymin, xmax, ymax)
        patch_size: Size of patches to divide image into (patch_h, patch_w)
        iou_threshold: Minimum IoU for patch inclusion

    Returns:
        Tensor of shape [B, N] where N is max number of patch IDs across batch,
        padded with -1 where necessary
    """
    if isinstance(bboxes, list):
        # Convert list of bounding boxes to tensor
        bboxes = convert_bboxes_list_to_tensor(bboxes)
    
    if bboxes[0][0][0] >= 0.0 and bboxes[0][0][0] <= 1.0:
        target_sizes = torch.tensor(images.shape[-2:]).unsqueeze(0).repeat(bboxes.shape[0], 1)
        bboxes = scale_boxes(bboxes, target_sizes)
    
    
    B, C, H, W = images.shape
    K = bboxes.shape[1]
    patch_h, patch_w = patch_size

    # Calculate number of patches in each dimension
    num_patches_h = (H + patch_h - 1) // patch_h  # ceil division
    num_patches_w = (W + patch_w - 1) // patch_w
    total_patches = num_patches_h * num_patches_w

    # Generate patch grid coordinates [num_patches_h, num_patches_w, 4]
    # Each patch has [xmin, ymin, xmax, ymax] in pixel coordinates
    patch_coords = torch.zeros(num_patches_h, num_patches_w, 4, dtype=torch.float32, device=bboxes.device)

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            ymin = i * patch_h
            xmin = j * patch_w
            ymax = min(ymin + patch_h, H)
            xmax = min(xmin + patch_w, W)
            patch_coords[i, j] = torch.tensor([xmin, ymin, xmax, ymax])

    # Flatten patch coordinates [total_patches, 4]
    patch_coords = patch_coords.view(-1, 4)

    # Process each image in batch
    all_patch_ids = []
    for batch_idx in range(B):
        current_bboxes = bboxes[batch_idx]  # Shape: [K, 4]

        # Calculate IoU between all patches and all bboxes [total_patches, K]
        # Using vectorized operations for efficiency
        patch_x1 = patch_coords[:, 0].unsqueeze(1)  # [total_patches, 1]
        patch_y1 = patch_coords[:, 1].unsqueeze(1)
        patch_x2 = patch_coords[:, 2].unsqueeze(1)
        patch_y2 = patch_coords[:, 3].unsqueeze(1)

        bbox_x1 = current_bboxes[:, 0]  # [K]
        bbox_y1 = current_bboxes[:, 1]
        bbox_x2 = current_bboxes[:, 2]
        bbox_y2 = current_bboxes[:, 3]

        # Intersection coordinates
        inter_x1 = torch.max(patch_x1, bbox_x1)
        inter_y1 = torch.max(patch_y1, bbox_y1)
        inter_x2 = torch.min(patch_x2, bbox_x2)
        inter_y2 = torch.min(patch_y2, bbox_y2)

        # Intersection areas
        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection = inter_width * inter_height

        # Patch and bbox areas
        patch_areas = (patch_x2 - patch_x1) * (
            patch_y2 - patch_y1
        )  # [total_patches, 1]
        bbox_areas = (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)  # [K]

        # IoU calculation [total_patches, K]
        union = patch_areas + bbox_areas - intersection
        iou = intersection / (union + 1e-6)

        # Find patches with IoU > threshold for any bbox
        valid_patches = (iou > iou_threshold).any(dim=1)  # [total_patches]
        patch_ids = torch.nonzero(valid_patches, as_tuple=True)[0]

        all_patch_ids.append(patch_ids)

    # Pad all sequences to max length in batch
    padded_patch_ids = torch.full((B, max_condition_length), -1, dtype=torch.long)

    for i, ids in enumerate(all_patch_ids):
        if len(ids) > 0:
            padded_patch_ids[i, : len(ids)] = ids

    return padded_patch_ids

def gather_with_gaussian_padding(
    visual_features: torch.Tensor,  # [B, L, D] input visual tokens
    padded_patch_ids: torch.Tensor,  # [B, N] indices with -1 for padding
    mean: float = 0.0,  # Mean for Gaussian distribution
    std: float = 1.0,  # Standard deviation for Gaussian
) -> torch.Tensor:
    """
    Gathers visual tokens using patch indices, filling padded positions (-1) with Gaussian noise.

    Args:
        visual_features: Input token features of shape [batch, num_patches, dim]
        padded_patch_ids: Patch indices of shape [batch, max_selected], where -1 indicates padding
        mean: Mean value for Gaussian noise
        std: Standard deviation for Gaussian noise

    Returns:
        Tensor of shape [batch, max_selected, dim] containing:
        - Gathered tokens for valid indices
        - Gaussian random values for padded positions (-1)

    Note:
        - Uses vectorized operations for efficiency
        - Handles edge cases (all padding, out-of-bound indices)
    """
    # Get input dimensions and device
    B, L, D = visual_features.shape
    device = visual_features.device
    
    padded_patch_ids = padded_patch_ids.to(device)

    # 1. Generate base output tensor filled with Gaussian noise
    # This will be overwritten for valid positions
    random_tokens = (
        torch.randn(B, padded_patch_ids.shape[1], D, device=device) * std + mean
    )

    # 2. Create mask for valid (non-padded) positions
    valid_mask = (padded_patch_ids != -1).to(device)

    # 3. Handle invalid indices safely:
    # Replace -1 with 0 (won't be used due to mask)
    safe_indices = torch.where(valid_mask, padded_patch_ids, 0).to(device)

    # 4. Gather tokens from visual_features
    # Expand indices to match feature dimension [B, N, D]
    gathered_tokens = torch.gather(
        visual_features, dim=1, index=safe_indices.unsqueeze(-1).expand(-1, -1, D)
    )

    # 5. Combine results using mask:
    # - Valid positions: gathered tokens
    # - Invalid positions: Gaussian noise
    result = torch.where(
        valid_mask.unsqueeze(-1),  # Expand mask to [B, N, 1]
        gathered_tokens,
        random_tokens,
    )

    return result


def convert_bboxes_list_to_tensor(bboxes: List) -> torch.Tensor:
    """
    Convert a list of bounding boxes to a tensor.
    
    Args:
        bboxes: List of bounding boxes, each represented as [xmin, ymin, xmax, ymax].
    
    Returns:
        Tensor of shape (B, K, 4) where K is the number of bounding boxes.
    """
    if not bboxes:
        return torch.zeros(1, 1, 4, dtype=torch.float32)
    
    if isinstance(bboxes[0], list):
        batch_bboxes = []
        max_bboxes = max(len(bbox) for bbox in bboxes)
        for bbox in bboxes:
            if len(bbox) < max_bboxes:
                bbox += [[0, 0, 0, 0]] * (max_bboxes - len(bbox))
            batch_bboxes.append(torch.tensor(bbox, dtype=torch.float32))
        return torch.stack(batch_bboxes, dim=0)
    else:
        # If bboxes is already a tensor
        return torch.tensor(bboxes, dtype=torch.float32).unsqueeze(0)
    
if __name__ == "__main__":
    # test create_mask_from_bboxes
    import matplotlib.pyplot as plt
    bboxes = [
        [[10, 10, 50, 50], [60, 60, 100, 100]],
        [[20, 20, 40, 40], [70, 70, 90, 90]],
    ]
    
    img_resolution = 128
    
    mask = create_mask_from_bboxes(bboxes, img_resolution)
    
    print("Mask shape:", mask.shape)
    
    print("Mask content:\n", mask)
    
    plt.imsave("mask.png", mask[0].cpu().numpy(), cmap='gray')
    plt.imsave("mask_1.png", mask[1].cpu().numpy(), cmap='gray')