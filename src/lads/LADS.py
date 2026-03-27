from lads.recursive_patches import *
from lads.events_utils import *
from torch.nn.functional import conv2d

class LADS:
    def __init__(self, H, W, device, ts_to_seconds_factor=1, num_bins=1, 
                 decay_func="ER", decay_param = 0.2, 
                 reference_event_rate=16, falloff_rate=None, 
                 do_patch_decay=False, patch_size=None, interpolate_patches=True,
                 min_decay=None,
                 fft_filter_radius=0.05,
                 recursive=False, min_patch_size=0, score_threshold=0.5, stop_above_thresh=True):
        
        self.device = device
        self.W = W
        self.H = H
        self.old_surface = torch.zeros((H,W),dtype=torch.float,device=device)
        self.old_surface_time = 0
        self.surface_time_initialized = False
        self.ts_to_seconds_factor = ts_to_seconds_factor
        self.num_bins = num_bins
        self.interpolate_patches = interpolate_patches
        self.decay_param = decay_param
        self.reference_event_rate = reference_event_rate
        self.recursive = recursive
        self.min_patch_size = min_patch_size
        self.score_threshold = score_threshold
        self.stop_above_thresh = stop_above_thresh
        self.fft_filter_radius = fft_filter_radius
        self.min_decay = min_decay if min_decay is not None else 0 # Minimum decay factor to apply, if given
        decay_func = decay_func.lower()

        decay_funcs = {"global-li", "er", "log", "fft"}
        if decay_func not in decay_funcs:
            raise ValueError(f"decay_func must be one of {decay_funcs}, got {decay_func} instead.")
        self.decay_func = decay_func
        

        self.do_patch_decay = do_patch_decay
        self.patch_size = patch_size
        if self.do_patch_decay and self.patch_size is not None:
            if isinstance(self.patch_size, int):
                self.patch_size = (self.patch_size, self.patch_size)    
            self.patch_score_conv = torch.nn.Conv2d(1, 1, kernel_size=self.patch_size, stride=self.patch_size, padding=0, bias=False, device=device)           
            self.patch_score_conv.weight.data.fill_(1)
            self.patch_score_conv.weight.requires_grad = False
            self.reference_event_rate = reference_event_rate*self.patch_size[0]*self.patch_size[1] # Event rate that sets centres decay at self.decay_param
        else:
            self.reference_event_rate = reference_event_rate*H*W # Event rate that sets centres decay at self.decay_param

        if falloff_rate is None: # Controls the steepness of the score-decay curve
            default_falloffs = {"log": 0.25} # just LoG for now
            falloff_rate = default_falloffs.get(decay_func, None)
        self.falloff_rate = falloff_rate

        if self.decay_func == "log":
            self.laplace_kernel_025 = create_log_kernel(3, 0.25, device=device)
            self.patch_score_conv.weight.data.fill_(1/(self.patch_size[0]*self.patch_size[1]))

        if self.decay_func == "fft":
            patch_filters = {}
            if recursive:
                if patch_size is None:
                    curr_size = self.H/2
                else:
                    curr_size = self.patch_size[0]/2 
                while (curr_size % 1 == 0) and (curr_size >= self.min_patch_size):
                    curr_size = int(curr_size)
                    curr_filter = self.get_fft_filter(curr_size, curr_size, fft_filter_radius, type="circle")
                    patch_filters[curr_size] = curr_filter
                    curr_size = curr_size / 2

            if patch_size is not None:
                curr_filter = self.get_fft_filter(self.patch_size[0], self.patch_size[1], fft_filter_radius, type="circle")
                patch_filters[self.patch_size[0]] = curr_filter
            self.patch_filters = patch_filters


    '''
    ============================= Event-Rate =============================
    '''

    def calc_patch_event_rates(self, events, time_diff_s):
        score_voxel_grid = voxel(events, self.H, self.W, self.device, polarity_mapping=(1, 1))
        patch_event_counts = self.patch_score_conv(score_voxel_grid).squeeze(0)
        patch_event_rates = patch_event_counts/time_diff_s
        return patch_event_rates
    
    def decay_by_event_rate_exp(self, events, time_diff_s, use_patches=False):
        
        if not use_patches:
            if time_diff_s <= 0 or len(events) == 0:
                return 1, 0
            event_rate = len(events) / time_diff_s
            score = event_rate / self.reference_event_rate
            decay_factor = np.exp((-(self.decay_param*score)))
            return decay_factor, score
        else:
            if time_diff_s <= 0 or len(events) == 0:
                return torch.ones((self.H//self.patch_size[0], self.W//self.patch_size[1]), device=self.device), \
                       torch.zeros((self.H//self.patch_size[0], self.W//self.patch_size[1]), device=self.device)
            patch_event_rates = self.calc_patch_event_rates(events, time_diff_s)
            patch_scores = patch_event_rates/self.reference_event_rate
            # decay scales exonentially with score, but unlike event-rate-linear, does not hit 1 when equal to reference rate
            # when equal to reference rate, decay is exp(-decay)
            patch_decay_factors = torch.exp(((-time_diff_s*patch_scores)/self.decay_param))
            return patch_decay_factors, patch_scores
    
    def decay_by_event_rate_linear(self, events, time_diff_s, use_patches=False):
        if not use_patches:
            if time_diff_s <= 0:
                return 1, 0
            event_rate = len(events) / time_diff_s
            score = event_rate / self.reference_event_rate
            decay_factors = 1 - score
            return decay_factors, score
        else:
            if time_diff_s <= 0:
                return torch.ones((self.H//self.patch_size[0], self.W//self.patch_size[1]), device=self.device), \
                       torch.zeros((self.H//self.patch_size[0], self.W//self.patch_size[1]), device=self.device)
            patch_event_rates = self.calc_patch_event_rates(events, time_diff_s)
            patch_scores = patch_event_rates/self.reference_event_rate
            patch_decay_factors = 1 - patch_scores
            return patch_decay_factors, patch_scores
    

    '''
    ============================= Laplace-of-Gaussian =============================
    '''
    def decay_by_LoG(self, new_grid):
        scores = torch.abs(conv2d(new_grid.unsqueeze(0).unsqueeze(0), self.laplace_kernel_025, padding=1))
        if self.patch_size[0] == 1 and self.patch_size[1] == 1:
            patch_scores = scores.squeeze(0).squeeze(0)
        else:
            patch_means = self.patch_score_conv(scores).squeeze(0).squeeze(0)
            patch_scores = patch_means
        
        a = self.falloff_rate
        b = self.decay_param#*100
        patch_decay_factors = (1.0 + np.exp(-a*(b))) / (1.0 + torch.exp(a * (patch_scores - b)))
        return patch_decay_factors, patch_scores

    
    
    '''
    ============================= FFT =============================
    '''

    def get_fft_filter(self, ph, pw, radius, type="circle", order=2):
     
        cy, cx = (ph-1) / 2, (pw-1) / 2
        ys = torch.arange(ph, device=self.device)
        ys = torch.where(ys <= cy, ys, ys - ph+1).view(ph, 1)
        xs = torch.arange(pw, device=self.device)
        xs = torch.where(xs <= cx, xs, xs - ph+1).view(1, pw)
        dist = torch.sqrt(ys.pow(2) + xs.pow(2))
        
        max_radius = dist.max()
        cutoff = radius * max_radius  # Cutoff frequency in pixels

        if type=="butterworth": # BUTTERWORTH:
            eps = torch.finfo(torch.float32).eps
            H_hp = 1.0 / (1.0 + (cutoff / (dist + eps)) ** (2 * order))

        else: # CIRCLE:
            H_hp = torch.ones((ph, pw), dtype=torch.float32, device=self.device)
            mask_area  = dist <= cutoff
            H_hp[mask_area] = 0
            # H_lp = 1.0 - H_hp

        return H_hp
        
    def get_fft_fraction(self, tensor):
        ph, pw = tensor.shape[-2:]
        H_hp = self.patch_filters[ph]

        fft = torch.fft.fft2(tensor)
        P = fft.abs().pow(2)
        
        energy_all = P.sum(dim=[2,3])
        energy_high = (H_hp * P).sum(dim=[2,3])
        R = energy_high / energy_all
        R = torch.where(torch.isnan(R), 1, R)
        return R

    def decay_by_fft(self, score_voxel_grid):

        if self.patch_size is None:
            if self.recursive:
                patch_scores = subdivide_grid_recur(score_voxel_grid, min_patch_size=self.min_patch_size, score_function=self.get_fft_fraction,
                                                        score_threshold=self.score_threshold, stop_above_thresh=self.stop_above_thresh)
            else:
                raise ValueError("Full-frame FFT decay not currently supported, please provide a patch_size or set recursive=True.")

        else:
            ph, pw = self.patch_size
            ny, nx = self.H // ph, self.W // pw
            patches = score_voxel_grid.unsqueeze(0).unsqueeze(0).unfold(2, self.patch_size[0], self.patch_size[0]).unfold(3, self.patch_size[1], self.patch_size[1]).contiguous()
            patches = patches.view(1,1,ny,nx,self.patch_size[0],self.patch_size[1])
                
            if self.recursive:
                patch_scores = self.recurr_with_init_patches(patches)

            else:
                patch_scores = torch.zeros((ny, nx), device=self.device)
                for i in range(ny):
                    for j in range(nx):
                        new_grid_patch = patches[:,:,i,j]
                        patch_score = self.get_fft_fraction(new_grid_patch)
                        patch_scores[i,j] = patch_score

        return patch_scores

        
    '''
    ============================= MISC =============================
    '''
    def _update_surface(self, decay_factor, new_events=None, new_grid=None):
        assert (new_grid is not None or new_events is not None), "Either new_events or new_grid must be provided for update_grid."
        
        if new_grid is None:
            new_grid = voxel(new_events, self.H, self.W, self.device, polarity_mapping=(-1, 1)).squeeze(0)

        self.old_surface = new_grid + self.old_surface*decay_factor
        return self.old_surface

    def integrateEvents(self, events, time_diff_s=None):

        if self.decay_func in ["global-li", "er"]: # Decay modes that require time difference calculation

            ''' TIME INITIALIZATION '''
                        
            if not self.surface_time_initialized:    
                if len(events) > 0:
                    self.old_surface_time = events[0][0]
                    self.surface_time_initialized = True

            if time_diff_s is None:
                if len(events) > 0:
                    if events.dtype.names is None:
                        time_diff_s = (events[-1][0] - self.old_surface_time) / self.ts_to_seconds_factor
                        self.old_surface_time = events[-1][0]
                    else:
                        time_diff_s = (events['t'][-1] - self.old_surface_time) / self.ts_to_seconds_factor
                        self.old_surface_time = events['t'][-1]
            
            elif self.surface_time_initialized:
                self.old_surface_time += time_diff_s
                
            assert time_diff_s >= 0, "Time difference must be positive, got {}".format(time_diff_s)



        with torch.no_grad():

            ''' GLOBAL DECAY '''
            if self.decay_func == "global-li":
                if self.decay_param == 0: # Taken to mean full decay of past events, i.e. Histogram
                    return self._update_surface(0, new_events=events), 0, 0
                assert time_diff_s is not None, "Time difference must be provided for global-li if passing empty event windows."
                decay_factor = torch.exp(-torch.scalar_tensor(time_diff_s / self.decay_param, device=self.device))
                return self._update_surface(decay_factor, new_events=events), time_diff_s, decay_factor
                
            if not self.do_patch_decay:
                if self.decay_func == "er":
                    decay_factor, score = self.decay_by_event_rate_exp(events, time_diff_s, use_patches=False)
                    decay_factor = torch.tensor(decay_factor, device=self.device).clamp(0, 1-self.min_decay)
                    return self._update_surface(decay_factor, new_events=events), score, decay_factor
                
                else:
                    print(f"Warning: {self.decay_func} decay is not implemented non-patched, continuing with default patch params.")
        

            ''' LOCAL DECAY '''
            
            new_grid = voxel(events, self.H, self.W, self.device, polarity_mapping=(-1, 1)).squeeze(0)

            if self.decay_func == "er":
                patch_decay_factors, patch_scores = self.decay_by_event_rate_exp(events, time_diff_s, use_patches=True)

            elif self.decay_func == "log":
                patch_decay_factors, patch_scores = self.decay_by_LoG(new_grid)


            elif self.decay_func == "fft":
                patch_scores = self.decay_by_fft(new_grid)
                patch_decay_factors = patch_scores



            if self.interpolate_patches:
                while len(patch_decay_factors.shape) < 4:
                    patch_decay_factors = patch_decay_factors.unsqueeze(0).unsqueeze(0)
                patch_decay_factors = torch.nn.functional.interpolate(patch_decay_factors, size=(self.H, self.W), 
                                                                    mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                                                                    # mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
            else:
                patch_decay_factors = patch_decay_factors.repeat_interleave(self.H//patch_scores.shape[-2], dim=0)
                patch_decay_factors = patch_decay_factors.repeat_interleave(self.W//patch_scores.shape[-1], dim=1)

            # Clamp decay factors
            patch_decay_factors = patch_decay_factors.clamp(0,1-self.min_decay)
            
            # Apply decay factors to old surface
            self.old_surface = new_grid + self.old_surface*patch_decay_factors

            return self.old_surface, patch_scores, patch_decay_factors
    
    def recurr_with_init_patches(self, patches):
        ph, pw = patches.shape[-2:]
        ny, nx = patches.shape[-4:-2]
        new_powers = []
        new_is = []
        for i in range(ny):
            for j in range(nx):
                new_grid_patch = patches[:,:,i,j] # Add batch and channel dimensions
                patch_score = self.get_fft_fraction(new_grid_patch)
                if (patch_score < self.score_threshold and self.stop_above_thresh) or \
                    (patch_score > self.score_threshold and not self.stop_above_thresh):
                    if not (ph%2 != 0 or pw%2 != 0 or ph/2 < self.min_patch_size or pw/2 < self.min_patch_size): 

                        patch_score = subdivide_grid_recur(new_grid_patch[0,0], min_patch_size=self.min_patch_size, 
                                                                score_function=self.get_fft_fraction,
                                                                score_threshold=self.score_threshold, 
                                                                stop_above_thresh=self.stop_above_thresh)
                new_powers.append(patch_score)
                new_is.append((i,j))

        max_shape = max([p.shape[0] for p in new_powers])
        for p_i in range(len(new_powers)):
            p = new_powers[p_i]
            if p.shape[0] == 1:
                p = p.repeat(max_shape,max_shape)
                new_powers[p_i] = p
                
            if p.shape[0] != max_shape:
                while p.shape[0] < max_shape:
                    p = p.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
                new_powers[p_i] = p

        patch_scores = torch.zeros((max_shape*ny,max_shape*nx), device=self.device)
        for i in range(ny):
            for j in range(nx):
                patch_scores[max_shape*i:max_shape*(i+1),max_shape*j:max_shape*(j+1)] = new_powers[new_is.index((i,j))]
        return patch_scores
                        


def create_log_kernel(kernel_size: int, sigma: float, device='cpu'):
    # Build a (1×1×K×K) LoG kernel
    ax = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size-1)/2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    norm = (xx**2 + yy**2 - 2*sigma**2) / (sigma**4)
    kernel = norm * torch.exp(-(xx**2 + yy**2)/(2*sigma**2))
    kernel -= kernel.mean()
    return kernel.unsqueeze(0).unsqueeze(0)
