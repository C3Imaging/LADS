import cv2
import os
import numpy as np
import argparse
import torch
from event_lads import *
import glob
from tempfile import TemporaryDirectory
from tqdm import tqdm

def parse_args():

    parser = argparse.ArgumentParser(description='Convert events to LADS')
    
    # Event stream 
    parser.add_argument('--events_path', type=str,  help='Path to the events file', default="examples/face.npy")
    parser.add_argument('--ts_to_seconds_factor', type=float, help='Factor for converting event timestamps to seconds', default=1) ## TODO: check
    parser.add_argument('--crop_t', type=int, help='Crop top coordinate', default=0)
    parser.add_argument('--crop_l', type=int, help='Crop left coordinate', default=0)
    parser.add_argument('--height', type=int, help='Height of events to gather for voxel (from crop_t)', default=720)
    parser.add_argument('--width', type=int, help='Width of events to gather for voxel (from crop_l)', default=1280)
    parser.add_argument("--device", default=None, help="Run on CPU or cuda GPU. If None (default), will use cuda if available")
    

    # Representation & decay
    parser.add_argument('--hz', type=int, help='Event accumulation frequency (1/<duration of event windows>)', default=30)
    parser.add_argument('--use_event_count', type=int, help='Override duration-based windowing with fixed event count', default=0)
    parser.add_argument('--representation', help='Representation of the output video', default="timesurface", choices=["timesurface", "histogram"])
    parser.add_argument('--decay_param', type=float, help='Parameter for scaling decay ("τ" in paper), effect changes depending on decay_func', default=0.2)
    parser.add_argument('--patch_size', type=int, help='Patch size for local adaptive decay, None for full height and width', default=80)
    parser.add_argument('--decay_func', type=str.lower, help='Function for decay rate calulation', default="er", 
                        choices=["global-li",
                                 "er",
                                 "fft",
                                 "log",
                                 ])
    parser.add_argument('--use_presets', action='store_true', help='Load preset params for decay function', default=False)
    parser.add_argument('--ref_event_rate', type=float, help='Event rate (events per pixel second) that defines an er-decay factor of exp(-decay)', default=0.5)
    parser.add_argument('--falloff_rate', default=0.5, help='Controls the steepness of the decay curve for LoG ("a" in paaper)')
    parser.add_argument('--min_decay', type=float, help='Optional minimum decay rate (instead of 0 decay for minimum score)', default=None)
    parser.add_argument('--interpolate_patches', type=bool, help='Interpolate patch decay values for per-pixel decay', default=True)
    parser.add_argument('--fft_filter_radius', type=float, help='Radius for FFT filter as a fraction of patch size', default=0.05)

    # Options for recursive patch decay (FFT only)
    parser.add_argument('--recursive_fft', type=bool, help='Recursively subdivide frame in quadrants to save computation (FFT method only)', default=True)
    parser.add_argument('--min_patch_size', type=int, help='Minimum patch size for recursive patches', default=40)
    parser.add_argument('--score_threshold', type=float, help='score threshold for recursive patches', default=0.9)
    
    # Visualisation and saving options
    parser.add_argument('--output_root', type=str, help='Output root directory', default="output")
    parser.add_argument('--output_name', type=str, help='Optionally specify output name (w/o file extension)', default="")
    parser.add_argument('--clip_val', type=float, help='Clip value for the output images/frames', default=5)
    parser.add_argument('--draw_heatmap', type=bool, help='Heatmap of decay values on output frames', default=False)
    parser.add_argument('--draw_grid', type=bool, help='Grid overlay on output frames', default=False)
    parser.add_argument('--annotate_score', type=bool, help='Annotate patches with local activity score used for decay', default=False)
    parser.add_argument('--annotate_decay', type=bool, help='Annotate patches with local decay factor', default=False)
    parser.add_argument('--save_frames', type=bool, help='Save frames as images', default=False)
    parser.add_argument('--save_video', type=bool, help='Save video', default=True)
    parser.add_argument('--overwrite_playback_fps', type=int, help='Playback FPS for the output video, same as FPS if None', default=None)
    parser.add_argument('--start_frame', type=int, help='Skip to start frame', default=0)
    parser.add_argument('--max_frames', type=int, help='Maximum number of frames to process', default=None)    
    
    args = parser.parse_args()
    return args


def load_preset(args):

    match args.decay_func:
        case "fixed-exponential":
            args.patch_size = None
            return args

        case "fft":
            args.patch_size = 80
            return args

        case "log":
            args.decay_param = 5
            args.falloff_rate = 0.5
            return args
    return args

def main(args):

    # Create integrator
    leaky_model = LADS(args.height, args.width, 
                       torch.device(args.device), 
                       ts_to_seconds_factor=args.ts_to_seconds_factor, 
                       decay_param=args.decay_param,
                       decay_func=args.decay_func, 
                       reference_event_rate=args.ref_event_rate,
                       falloff_rate=args.falloff_rate,
                       patch_size=args.patch_size, 
                       interpolate_patches=args.interpolate_patches,
                       min_decay=args.min_decay,
                       fft_filter_radius=args.fft_filter_radius,
                       recursive=args.recursive_fft, min_patch_size=args.min_patch_size,
                       score_threshold=args.score_threshold
                       )

    # Load events
    event_windows = load_event_windows(args)

    # Form output path:
    # video_name = args.events_path.split("\\")[-1].split(".")[0]
    video_name = os.path.basename(args.events_path).split(".")[0]
    if args.output_name != "":
        video_name = args.output_name
        video_suffix = ""
    else:
        video_suffix = generate_filename_suffix(args)

    output_dir = os.path.join(args.output_root, video_name)
    out_video_path = os.path.join(output_dir, video_name + video_suffix + '.mp4')
    print("Saving to: ", output_dir)
    os.makedirs(output_dir, exist_ok=True)


    if args.save_frames:
        frame_dir = video_name+video_suffix
        frame_dir = os.path.join(output_dir, frame_dir)
        os.makedirs(frame_dir, exist_ok=True)
        
    else:
        temp_dir = TemporaryDirectory()
        frame_dir = temp_dir.name

    if args.start_frame > 0:
        skip_progress = tqdm(total=args.start_frame-1, desc="Skipping event windows until start frame")

    if args.max_frames is not None:
        progress_bar = tqdm(total=args.max_frames-1)
    else:
        progress_bar = tqdm(total=len(event_windows)-args.start_frame)

    # Iterate through event windows
    count = 0
    for events in event_windows:
        if count < args.start_frame:
            count += 1
            skip_progress.update(1)
            continue
        progress_bar.update(1)

        if args.crop_t is not None and args.crop_l is not None:
            events = events[np.where(events[:,1]>=args.crop_l)[0]]
            events = events[np.where(events[:,1]<args.crop_l+args.width)[0]]
            events = events[np.where(events[:,2]>=args.crop_t)[0]]
            events = events[np.where(events[:,2]<args.crop_t+args.height)[0]]
            events[:,1] -= args.crop_l
            events[:,2] -= args.crop_t
        

        surface, patch_scores, patch_decay_factors = leaky_model.integrateEvents(events)

        frame = LADS_to_output_frame(surface, patch_scores, patch_decay_factors, 
                                     clip_val=args.clip_val, 
                                     draw_heatmap=args.draw_heatmap, 
                                     draw_grid=args.draw_grid, 
                                     annotate_score=args.annotate_score, 
                                     annotate_decay=args.annotate_decay, 
                                     recursive=(args.recursive_fft and args.decay_func == "fft"),
                                     )
        
        cv2.imwrite(os.path.join(frame_dir,f"{str(count).zfill(4)}.png"), frame)

        if args.max_frames is not None and count >= args.start_frame + args.max_frames - 1:
            break
        count += 1

    if args.save_video:

        images = glob.glob(os.path.join(frame_dir,'*.png'))
        images = sorted(images, key = lambda x: float(x.split('\\')[-1].split('.')[0]))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w, c = cv2.imread(images[0]).shape
        print("Saving video to {}".format(out_video_path))
        if args.overwrite_playback_fps is not None:
            playback_fps = args.overwrite_playback_fps
        else:
            playback_fps = args.hz
        out = cv2.VideoWriter(out_video_path, fourcc, int(playback_fps), (w,h))
    
        for path in images:
            img = cv2.imread(path)
            out.write(img)
        out.release()
    
    if not args.save_frames:
        temp_dir.cleanup()


def generate_filename_suffix(args):
    video_suffix = ""
    if args.use_event_count > 0:
        video_suffix += f"_{args.use_event_count}evs"
    video_suffix += f"_{args.hz}fps"
    if args.representation == "timesurface":
        if args.decay_func == "fixed-exponential":
            video_suffix += f"_fixed-decay"+str(args.decay_param).replace(".","-")
        elif  "event-rate" in args.decay_func:
            video_suffix += "_event-rate"+str(args.ref_event_rate).replace(".","-")+f"_{args.decay_func.replace('event-rate-','')}"+str(args.decay_param).replace(".","")
        elif args.decay_func == "fft":
            video_suffix += f"_fft"+str(args.fft_filter_radius).replace(".","-")
        else:
            video_suffix += "_"+args.decay_func+str(args.decay_param).replace(".","-")
    else:
        video_suffix += "_histogram"

    if args.do_patch_decay:
        if args.patch_size is not None or args.min_patch_size is not None:
            video_suffix += f"_patch{args.patch_size}" if args.patch_size is not None else f"_patch{args.min_patch_size}"
        video_suffix += "_heatmap" if args.draw_heatmap else ""
    return video_suffix

def load_event_windows(args):
    
    duration = 1/args.hz
    event_windows = []
    if args.use_event_count > 0:
        if args.events_path.split(".")[-1] == "txt":
            event_reader = FixedSizeEventReader(args.events_path, num_events=args.use_event_count)
            for w in iter(event_reader):
                event_windows.append(w)
        
        elif args.events_path.split(".")[-1] == "npy":
            all_events = np.load(args.events_path, allow_pickle=False)
            if all_events.dtype.names is not None:
                all_events = np.array([all_events[name] for name in all_events.dtype.names]).T
            
            num_events = len(all_events)
            num_windows = num_events//args.use_event_count
            remainder = num_events%args.use_event_count
            if remainder == 0:
                num_windows -= 1

            event_windows = np.array_split(all_events[:-remainder], num_windows, axis=0)
            
            if remainder > 0:
                event_windows.append(all_events[-remainder:])
            
        else:
            print("Event count windowing only implemented for npy and txt files.")
            exit()
    else:
        if args.events_path.split(".")[-1] == "txt":
            event_reader = FixedDurationEventReader(args.events_path, duration, time_to_seconds=args.ts_to_seconds_factor)
            for w in iter(event_reader):
                event_windows.append(w)
        elif args.events_path.split(".")[-1] == "npy":
            all_events = np.load(args.events_path, allow_pickle=False)
            if all_events.dtype.names is not None:
                all_events = np.array([all_events[name] for name in all_events.dtype.names]).T
            max_t = all_events[-1,0]
            timestamps = np.arange(all_events[0,0],max_t,duration)
            
            # print("Event rate pps:", measure_event_rate(all_events,args.height,args.width))
            for count in range(len(timestamps)-1):
                ev_st = np.searchsorted(all_events[:,0], timestamps[count], side="left")
                ev_en = np.searchsorted(all_events[:,0], timestamps[count+1], side="left")
                events = all_events[ev_st:ev_en]
                event_windows.append(events)
            events = all_events[ev_en:]
            event_windows.append(events)
    print(len(event_windows), "event windows")

    return event_windows


if __name__ == '__main__':
    args = parse_args()
    
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    if args.use_presets:
        args = load_preset(args)
    
    do_patch_decay = None if (args.patch_size is None and not args.recursive_fft) else True

    if args.representation == "histogram":
        args.decay_func = "global-li" # histogram implemented as leaky time surface with full decay every frame
        args.decay_param = 0
    
    if args.decay_func == "global-li":
        do_patch_decay = False
    setattr(args, 'do_patch_decay', do_patch_decay)
    
    main(args)