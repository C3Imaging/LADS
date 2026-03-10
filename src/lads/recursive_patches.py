import torch

def subdivide_grid_fast(grid, score_function, dim_factor=2):
    patch_size = grid.shape[-1] // dim_factor
    patches = grid.unsqueeze(0).unsqueeze(0).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).contiguous()
    patches = patches.squeeze(0).squeeze(0)
    powers = score_function(patches)
    return patches, powers 

def subdivide_grid_recur(grid, score_function, min_patch_size=45, score_threshold=0.5, stop_above_thresh=True, depth=0):
    
    patches, powers = subdivide_grid_fast(grid, dim_factor=2, score_function=score_function)
    ny, nx = patches.shape[0], patches.shape[1]
    ph, pw = patches.shape[2], patches.shape[3]

    if ph%2 != 0 or pw%2 != 0: # if patch size is odd, we cannot subdivide further
        return powers
        
    if ph/2 < min_patch_size or pw/2 < min_patch_size: # return if patch size below min
        return powers
    
    if ((powers > score_threshold).all() and stop_above_thresh) or ((powers < score_threshold).all() and not stop_above_thresh):
        # all patches within threshold, no need to subdivide further
        return powers
    
    new_powers = []
    for row in range(ny):
        for col in range(nx):

            patch_power = powers[row, col]
            if (patch_power < score_threshold and stop_above_thresh) or (patch_power > score_threshold and not stop_above_thresh):
                new_patch_powers = subdivide_grid_recur(patches[row, col], min_patch_size=min_patch_size, score_function=score_function,
                                                             score_threshold=score_threshold, 
                                                              stop_above_thresh=stop_above_thresh, depth=depth+1)

            else:
                new_patch_powers = patch_power.view(1,1)
                
            new_powers.append(new_patch_powers)

    # upscale to match new largest
    max_shape = max(new_powers[0].shape[0], new_powers[1].shape[0], 
                    new_powers[2].shape[0], new_powers[3].shape[0])
    for p_i in range(4):
        p = new_powers[p_i]
        if p.shape[0] == 1:
            p = p.repeat(max_shape,max_shape)
            new_powers[p_i] = p
            
        if p.shape[0] != max_shape:
            # print(f"Found patch with shape = {p.shape} when max_shape = {max_shape}")
            while p.shape[0] < max_shape:
                p = p.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
            new_powers[p_i] = p
    
    row1 = torch.cat(new_powers[:2],dim=1)
    row2 = torch.cat(new_powers[2:],dim=1)
    powers = torch.cat((row1,row2), dim=0)
    
    return powers 

