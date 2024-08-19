import os
import math
import torch
import av
import cv2
import imageio
import decord
from decord import VideoReader, cpu
# decord.bridge.set_bridge('torch')
import numpy as np
from PIL import Image

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample



def sample_frames(num_frames, video_len, sample='uniform', fix_start=-1):
    if num_frames >= video_len:
        return range(video_len)

    intv = np.linspace(start=0, stop=video_len, num=num_frames+1).astype(int)
    if sample == 'rand':
        frame_ids = [random.randrange(intv[i], intv[i+1]) for i in range(len(intv)-1)]
    elif fix_start >= 0:
        fix_start = int(fix_start)
        frame_ids = [intv[i]+fix_start for i in range(len(intv)-1)]
    elif sample == 'uniform':
        frame_ids = [(intv[i]+intv[i+1]-1) // 2 for i in range(len(intv)-1)]
    else:
        raise NotImplementedError
    return frame_ids


def read_videos(video_path, num_frames=-1, sample='uniform', decode='decord', dtype='numpy', fps=None, trim=1., fix_start=-1, keyframe=False, start_ratio=0.0, end_ratio=1.0):
    
    # load video
    if decode == 'decord':
        # video_reader = decord.VideoReader(video_path, height=224, width=224)
        # video_reader = decord.VideoReader(video_path, num_threads=0)
        video_reader = decord.VideoReader(video_path, ctx=cpu(0))
        vlen = len(video_reader)
    elif decode == 'cv2':
        video = cv2.VideoCapture(video_path)
        success, image = video.read()
        success = True
        frames = []
        while success:
            frames.append(image)
            success, image = video.read()
        vlen = len(frames)
    elif decode == 'av':
        video = av.open(video_path)
        frames = []

        if fps is not None:
            avg_fps = int(video.streams.video[0].averate_rate)
            if fps <= avg_fps:
                step = avg_fps
                for idx, frame in enumerate(video.decode(video=0)):
                    if idx % step == 0:
                        frame = frame.to_ndarray(format='rgb24')
                        frames.append(frame)

        for frame in video.decode(video=0):
            frame = frame.to_ndarray(format='rgb24')
            frames.append(frame)
        vlen = len(frames)

    # sample frames
    ori_indices = list(range(vlen))
    if trim < 1.:
        remain = (1. - trim) / 2
        start, end = int(vlen * remain), int(vlen * (1 - remain))
        indices = ori_indices[start:end]
    if keyframe:
        start, end = int(vlen*start_ratio), int(vlen*end_ratio)+1
        indices = ori_indices[start:end]
    if num_frames > 0:
        while vlen < num_frames: # duplicate frames
            ori_indices = [f for ind in ori_indices for f in (ind, ind)]
            vlen = len(ori_indices)
        frame_ids = sample_frames(num_frames, vlen, sample, fix_start)
        indices = [ori_indices[ii] for ii in frame_ids]
    
    if decode == 'decord':
        if dtype == 'torch':
            frames = video_reader.get_batch(indices).permute(3,0,1,2).float() # T H W C -> C T H W
        elif dtype == 'numpy':
            frames = video_reader.get_batch(indices).numpy() # T H W C
    elif decode == 'cv2':
        if dtype == 'torch':
            frames = torch.from_numpy(np.stack([frames[x] for x in indices], axis=0)).permute(3,0,1,2).float() # T H W C -> C T H W
        elif dtype == 'numpy':
            frames = np.stack([frames[x] for x in indices], axis=0) # T H W C
        video.release()
    elif decode == 'av':
        if dtype == 'torch':
            frames = torch.from_numpy(np.stack([frames[x] for x in indices], axis=0)).permute(3,0,1,2).float() # T H W C -> C T H W
        elif dtype == 'numpy':
            frames = np.stack([frames[x] for x in indices], axis=0)
    return frames



def load_video(
        video_path,
        video_decode_backend='decord',
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8,
        fps=None,
        max_frames=16,
):
    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec).numpy()

    elif video_decode_backend == 'decord':
        # decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        duration = len(decord_vr)
        
        if fps:
            avg_fps = decord_vr.get_avg_fps()
            secs = duration / avg_fps
            new_duration = math.ceil(secs * fps)
            # num_frames = max(8, new_duration-new_duration%8)
            num_frames = new_duration
            num_frames = min(num_frames, max_frames) # TODO: maximum frames
        
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list.tolist()).numpy() # (T, H, W, C)

    elif video_decode_backend == 'opencv':
        cv2_vr = cv2.VideoCapture(video_path)
        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id_list = np.linspace(0, duration-5, num_frames, dtype=int)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(1, frame_idx)
            ret, frame = cv2_vr.read()
            if not ret:
                raise ValueError(f'video error at {video_path}')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(frame)
        cv2_vr.release()
        video_data = np.stack(video_data, dim=0) # (T, H, W, C)

    elif video_decode_backend == 'av':
        av_vr = av.open(video_path)
        frames = [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]
        vlen = len(frames)
        video_stream = av_vr.streams.video[0]
        if video_stream.duration == math.inf: video_duration = math.inf
        else:
            video_duration = int(video_stream.duration - video_stream.start_time) * video_stream.time_base
        fps = vlen / float(video_duration)
        frame_indices = np.linspace(0, vlen-1, num_frames, dtype=int)
        video_data = np.stack([frames[idx] for idx in frame_indices]) # T H W C
        
    
    elif video_decode_backend == 'gif':
        if video_path.startswith('s3') or video_path.startswith('p2'):
            video_bytes = client.get(video_path)
            gif = imageio.get_reader(io.BytesIO(video_bytes))
        else:
            gif = imageio.get_reader(video_path)
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) for frame in gif]
        vlen = len(frames)
        frame_indices = np.linspace(0, vlen-1, num_frames, dtype=int)
        video_data = np.stack([frames[idx] for idx in frame_indices])
        

    elif video_decode_backend == 'frame':
        max_frame = len(os.listdir(video_path))
        image_groups = list()
        frame_indices = np.linspace(1, max_frame, num_frames, dtype=int)
        for ind in frame_indices:
            img = Image.open(os.path.join(video_path, f"{ind:05d}.jpg"))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image_groups.append(img)
        # formulate images
        video_data = np.stack(image_groups)
    
    
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_data

