import torch
import numpy as np
import pandas as pd
import zipfile
from os.path import splitext
import cv2

def voxel(events, height, width, device, esim=False, polarity_mapping=(-1, 1)):
    # TODO: reimplement multiple time bins
    voxel_grid = torch.zeros(1, height, width, dtype=torch.float32, device=device)
    if len(events) == 0:
        return voxel_grid
    voxel_grid = voxel_grid.flatten()
    if events.dtype.names is None:
        events_torch = torch.from_numpy(events.astype(np.float32)).to(device)

        if esim:
            xs = events_torch[:, 0].long()
            ys = events_torch[:, 1].long()
        else:
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
        pols = events_torch[:, 3].float()
    else:
        
        # events_torch = torch.from_numpy(events).to(device)
        xs = torch.from_numpy(events['x'].astype(np.float32)).to(device)
        ys = torch.from_numpy(events['y'].astype(np.float32)).to(device)
        pols = torch.from_numpy(events['p'].astype(np.float32)).to(device)

    pols[pols == 0] = polarity_mapping[0]
    index1 = (xs + ys*width).long()
    voxel_grid.index_add_(dim=0, index=index1, source=pols) 
    voxel_grid = voxel_grid.view(1, height, width)

    return voxel_grid


def event_image(events, height, width, device, esim=False, polarity_mapping=(-1, 1)):
    
    voxel_grid = torch.zeros(1, height, width, dtype=torch.float32, device=device)
    if len(events) == 0:
        return voxel_grid
    voxel_grid = voxel_grid.flatten()
    if events.dtype.names is None:
        events_torch = torch.from_numpy(events.astype(np.float32)).to(device)

        if esim:
            xs = events_torch[:, 0].long()
            ys = events_torch[:, 1].long()
        else:
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
        pols = events_torch[:, 3].float()
    else:
        
        # events_torch = torch.from_numpy(events).to(device)
        xs = torch.from_numpy(events['x'].astype(np.float32)).to(device)
        ys = torch.from_numpy(events['y'].astype(np.float32)).to(device)
        pols = torch.from_numpy(events['p'].astype(np.float32)).to(device)

    pols[pols == 0] = polarity_mapping[0]
    index1 = (xs + ys*width).long()
    voxel_grid[index1] = pols  # Overwrite instead of adding
    voxel_grid = voxel_grid.view(1, height, width)

    return voxel_grid


def crop_events(events, x1, y1, x2=None, y2=None, width=None, height=None):
    
    assert(x2 is not None or width is not None)
    assert(y2 is not None or height is not None)

    if events.dtype.names is None:
        events = events[np.where(events[:,1]>=x1)[0]]
        events = events[np.where(events[:,2]>=y1)[0]]

        if x2 is not None:
            events = events[np.where(events[:,1]<x2)[0]]
        else:
            events = events[np.where(events[:,1]<x1+width)[0]]

        if y2 is not None:
            events = events[np.where(events[:,2]<y2)[0]]
        else:
            events = events[np.where(events[:,2]<y1+height)[0]]

        events[:,1] -= x1
        events[:,2] -= y1

    else:
        events = events[np.where(events['x']>=x1)[0]]
        events = events[np.where(events['y']>=y1)[0]]

        if x2 is not None:
            events = events[np.where(events['x']<x2)[0]]
        else:
            events = events[np.where(events['x']<x1+width)[0]]
                
        if y2 is not None:
            events = events[np.where(events['y']<y2)[0]]
        else:
            events = events[np.where(events['y']<y1+height)[0]]

        events['x'] -= x1
        events['y'] -= y1

    return events


def pad_events(events, add_to_left=None, add_to_top=None):

    if events.dtype.names is None:
        if add_to_left is not None:
            events[:,1] += add_to_left
        if add_to_top is not None:
            events[:,2] += add_to_top
    else:
        if add_to_left is not None:
            events['x'] += add_to_left
        if add_to_top is not None:
            events['y'] += add_to_top
    return events


class FixedSizeEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path_to_event_file, num_events=10000, start_index=0):
        print('Will use fixed size event windows with {} events'.format(num_events))
        print('Output frame rate: variable')

        self.iterator = pd.read_csv(path_to_event_file, delim_whitespace=True, header=None,
                                    names=['t', 'x', 'y', 'pol'],
                                    dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                                    engine='c',
                                    skiprows=start_index + 1, chunksize=num_events, nrows=None, memory_map=True)

    def __iter__(self):
        return self

    def __next__(self):

        event_window = self.iterator.__next__().values

        return event_window


class FixedDurationEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.

    **Note**: This reader is much slower than the FixedSizeEventReader.
              The reason is that the latter can use Pandas' very efficient cunk-based reading scheme implemented in C.
    """

    # def __init__(self, path_to_event_file, duration_ms=50.0, start_index=0):
    def __init__(self, path_to_event_file, duration_s=1/30, start_index=0, time_to_seconds=1, t_position=0):
        print('Will use fixed duration event windows of size {:.2f}ms'.format(duration_s*1e3))
        print('Output frame rate: {:.1f} Hz'.format(1 / duration_s))
        file_extension = splitext(path_to_event_file)[1]
        assert(file_extension in ['.txt', '.zip'])
        self.is_zip_file = (file_extension == '.zip')

        if self.is_zip_file:  # '.zip'
            self.zip_file = zipfile.ZipFile(path_to_event_file)
            files_in_archive = self.zip_file.namelist()
            assert(len(files_in_archive) == 1)  # make sure there is only one text file in the archive
            self.event_file = self.zip_file.open(files_in_archive[0], 'r')
        else:
            self.event_file = open(path_to_event_file, 'r')

        # ignore the lines before start_index 
        for i in range(start_index):
            self.event_file.readline()

        self.last_stamp = None
        self.duration_s = duration_s*time_to_seconds
        self.time_to_seconds = time_to_seconds
        self.t_position = t_position  # Position of the timestamp value in event (0-3)
    def __iter__(self):
        return self

    def __del__(self):
        if self.is_zip_file:
            self.zip_file.close()

        self.event_file.close()

    def __next__(self):
        event_list = []
        if self.t_position == 0:
            for line in self.event_file:
                if self.is_zip_file:
                    line = line.decode("utf-8")
                t, x, y, pol = line.split(' ')
                t, x, y, pol = float(float(t)), int(float(x)), int(float(y)), int(float(pol))
                event_list.append([t, x, y, pol])
                if self.last_stamp is None:
                    self.last_stamp = t
                if t > self.last_stamp + self.duration_s:
                    self.last_stamp = t
                    event_window = np.array(event_list)
                    return event_window
        elif self.t_position == 3:
            for line in self.event_file:
                if self.is_zip_file:
                    line = line.decode("utf-8")
                x, y, pol, t = line.split(' ')
                t, x, y, pol  = float(float(t)), int(float(x)), int(float(y)), int(float(pol))
                event_list.append([t, x, y, pol])
                if self.last_stamp is None:
                    self.last_stamp = t
                if t > self.last_stamp + self.duration_s:
                    self.last_stamp = t
                    event_window = np.array(event_list)
                    return event_window
        raise StopIteration

def grid_tensor_to_img(integrated, clip_val=5, draw_heatmap=False, draw_grid=False, annotate_score=False, annotate_decay=False, 
                       write_path=None, recursive=False, text_scale=0.8):
        if isinstance(integrated, tuple):
            grid = integrated[0]
            patch_activities, patch_decay_factors = integrated[1]
        else:
            grid = integrated
        if not isinstance(clip_val, tuple):
            clip_val = (-clip_val,clip_val)
        
        grid_ = grid.clamp(clip_val[0],clip_val[1])
        grid_ = (grid_ - clip_val[0]) / (clip_val[1] - clip_val[0]) # normalize to [0,1]
        
        frame = grid_.squeeze(0).detach().cpu().numpy()
        frame = np.stack((frame,frame,frame), axis=2)
        frame = (frame*255).astype(np.uint8)
        frame_orig = frame.copy()
        height, width = frame.shape[0], frame.shape[1]

        if isinstance(integrated, tuple):
            if draw_grid or annotate_score or annotate_decay:
                line_colour = (100, 255, 100)
                if not recursive:
                    patch_size = height//patch_activities.shape[0]
                    patch_activities = patch_activities.detach().cpu().numpy()
                    # patch_activities_laplace = filters.laplace(patch_activities)
                    # patch_activities = local_info_gain(patch_activities)
                    num_patches_y, num_patches_x = patch_activities.shape[-2], patch_activities.shape[-1]
                    # if len(patch_activities.shape) > 2:
                    #     _, num_patches_y, num_patches_x = patch_activities.shape[0], patch_activities.shape[1]
                    # else:
                    # num_patches_y, num_patches_x = patch_activities.shape[1], patch_activities.shape[2]
                    for i in range(num_patches_y):
                        #draw horizontal patch lines at 50% opacity
                        patch_y = i*patch_size
                        cv2.line(frame, (0, patch_y), (width, patch_y), line_colour, 1)

                        for j in range(num_patches_x):
                            # annotate patch activity and decay factor
                            patch_x = j*patch_size
                            cv2.line(frame, (patch_x, 0), (patch_x, height), line_colour, 1)
                            
                            if patch_size >= 30: #otherwise too small for text, just use heatmap
                                if annotate_score:
                                    patch_activity = patch_activities[i,j]
                                    cv2.putText(frame, f"{patch_activity:.2f}", (patch_x+5, patch_y+25), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255,0,0), 1)
                                
                                if annotate_decay:
                                    patch_decay_factor = patch_decay_factors[patch_y+patch_size//2,patch_x+patch_size//2].item()
                                    # Get color from colormap based on decay value
                                    decay_color_val = np.clip(patch_decay_factor, 0, 1)
                                    decay_color_gray = np.array([[[decay_color_val * 255]]], dtype=np.uint8)
                                    decay_color_mapped = cv2.applyColorMap(decay_color_gray, cv2.COLORMAP_JET)[0,0]
                                    # Convert BGR to RGB and make it a tuple for OpenCV
                                    text_color = (int(decay_color_mapped[0]), int(decay_color_mapped[1]), int(decay_color_mapped[2]))
                                    cv2.putText(frame, f"{patch_decay_factor:.2f}", (patch_x+5, patch_y+50 if annotate_score else patch_y+25), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
                else:

                    patch_size = height//patch_activities.shape[0]
                    patch_activities = patch_activities.detach().cpu().numpy()
                    num_patches_y, num_patches_x = patch_activities.shape[-2], patch_activities.shape[-1]

                    for i in range(num_patches_y):
                        patch_y = i*patch_size

                        for j in range(num_patches_x):
                            patch_x = j*patch_size
                            
                            patch_activity = patch_activities[i,j]
                            diff_from_above = False
                            diff_from_left = False
                            # Draw patch lines where activity is different to neighbour (for visualisation of recursive subdivision)
                            if i > 0:
                                if patch_activity != patch_activities[i-1,j]:
                                    diff_from_above = True
                                    # draw horizontal line on top of patch
                                    cv2.line(frame, (patch_x, patch_y), (patch_x+patch_size, patch_y), line_colour, 1)
                            if j > 0:
                                if patch_activity != patch_activities[i,j-1]:
                                    diff_from_left = True
                                    # draw vertical line on left of patch
                                    cv2.line(frame, (patch_x, patch_y), (patch_x, patch_y+patch_size), line_colour, 1)
                            if patch_size >= 30 and annotate_score: #otherwise too small for text, just use heatmap
                                if (diff_from_above and diff_from_left) or (diff_from_left and i == 0) or (diff_from_above and j == 0) or (i == 0 and j == 0):
                                    cv2.putText(frame, f"{patch_activity:.2f}", (patch_x+2, patch_y+14), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255,0,0), 1)

                                # patch_decay_factor = patch_decay_factors[patch_y, patch_x].item()
                                # cv2.putText(frame, f"{patch_decay_factor:.2f}", (patch_x+2, patch_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
                frame = frame_orig*.65 + frame*.35
            if draw_heatmap:
                decay_grid = patch_decay_factors.detach().cpu().numpy()
                decay_grid = np.clip(decay_grid, 0, 1)
                decay_grid = np.stack((decay_grid, decay_grid, decay_grid), axis=2)

                decay_grid = 1-decay_grid  # invert for heatmap
                decay_grid = (decay_grid * 255).astype(np.uint8)
                decay_grid = cv2.applyColorMap(decay_grid, cv2.COLORMAP_JET)
                frame = 0.9*frame + 0.1*decay_grid
                
                frame = frame.astype(np.uint8)
    
        if write_path is not None:
            cv2.imwrite(write_path, frame)
        return frame


def measure_event_rate(events, H, W):
    length = len(events)
    if length == 0:
        return 0
    t_start = events[0][0]
    t_end = events[-1][0]
    duration = t_end - t_start
    sensor_rate = length / duration
    pixel_rate = sensor_rate / (H * W)
    return pixel_rate