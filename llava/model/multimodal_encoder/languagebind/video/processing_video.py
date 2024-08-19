import os
import math
import torch
import av
import cv2
import imageio
import decord
from decord import VideoReader, cpu
decord.bridge.set_bridge('torch')
import numpy as np
from PIL import Image

from torchvision import transforms
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo

from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

def get_video_transform(config):
    config = config.vision_config
    if config.video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(config.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )

    elif config.video_decode_backend == 'decord':

        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    elif config.video_decode_backend == 'opencv':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return transform


def load_and_transform_video(
        video_path,
        transform,
        video_decode_backend='decord',
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8,
        fps=None,
):
    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'decord':
        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        duration = len(decord_vr)
        
        if fps:
            avg_fps = decord_vr.get_avg_fps()
            secs = duration / avg_fps
            new_duration = math.ceil(secs * fps)
            num_frames = max(8, new_duration-new_duration%8)
            num_frames = min(num_frames, 512) # TODO: maximum frames
        
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_outputs = transform(video_data)

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
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        cv2_vr.release()
        video_data = torch.stack(video_data, dim=1)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'av':
        av_vr = av.open(video_path)
        frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
        vlen = len(frames)
        video_stream = av_vr.streams.video[0]
        if video_stream.duration == math.inf: video_duration = math.inf
        else:
            video_duration = int(video_stream.duration - video_stream.start_time) * video_stream.time_base
        fps = vlen / float(video_duration)
        frame_indices = np.linspace(0, vlen-1, num_frames, dtype=int)
        frames = torch.stack([frames[idx] for idx in frame_indices])
        frames = frames.permute(3, 0, 1, 2) # C T H W 
        video_outputs = transform(frames)
        
    
    elif video_decode_backend == 'gif':
        if video_path.startswith('s3') or video_path.startswith('p2'):
            video_bytes = client.get(video_path)
            gif = imageio.get_reader(io.BytesIO(video_bytes))
        else:
            gif = imageio.get_reader(video_path)
        frames = [torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)) for frame in gif]
        vlen = len(frames)
        frame_indices = np.linspace(0, vlen-1, num_frames, dtype=int)
        frames = torch.stack([frames[idx] for idx in frame_indices])
        frames = frames.permute(3, 0, 1, 2) # T H W C -> C T H W
        # vlen = len(gif)
        # frame_indices = np.linspace(0, vlen-1, num_frames, dtype=int)
        # frames = []
        # for frame_idx in frame_indices:
        #     frame = gif[frame_idx]
        #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGB)
        #     frame = torch.from_numpy(frame)
        #     frame = frame.permute(2,0,1)
        #     frames.append(frame)

        # for index, frame in enumerate(gif):
        #     # for index in frame_idxs:
        #     if index in frame_indices:
        #         frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        #         # frame = torch.from_numpy(frame).byte()
        #         frame = torch.from_numpy(frame)
        #         # # (H x W x C) to (C x H x W)
        #         frame = frame.permute(2, 0, 1)
        #         frames.append(frame)
        # frames = torch.stack(frames, dim=1)  # .float() / 255
        # print(frames.shape)
        video_outputs = transform(frames)

    elif video_decode_backend == 'frame':
        max_frame = len(os.listdir(video_path))
        image_groups = list()
        frame_indices = np.linspace(1, max_frame, num_frames, dtype=int)
        for ind in frame_indices:
            img = Image.open(os.path.join(video_path, f"{ind:05d}.jpg"))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = torch.from_numpy(np.array(img))
            image_groups.append(img)
        # formulate images
        frames = torch.stack(image_groups)
        frames = frames.permute(3, 0, 1, 2)
        video_outputs = transform(frames)
    
    
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_outputs

class LanguageBindVideoProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindVideoTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # self.config.vision_config.video_decode_backend = 'decord'
        # self.config.vision_config.num_frames = 16 # default 8
        self.transform = get_video_transform(config)
        self.image_processor = load_and_transform_video
        self.tokenizer = tokenizer
        

    def __call__(self, videos=None, text=None, context_length=77, return_tensors=None, fps=None, **kwargs):
        if text is None and videos is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                      truncation=True, return_tensors=return_tensors, **kwargs)

        if videos is not None:
            videos = make_list_of_images(videos)
            image_features = []
            for video in videos:
                # print(video)
                if os.path.isdir(video):
                    image_features.append(self.image_processor(video, self.transform,
                                                               video_decode_backend="frame",
                                                               num_frames=self.config.vision_config.num_frames))
                
                elif video.endswith(".gif"):
                    image_features.append(self.image_processor(video, self.transform, 
                                                    video_decode_backend="gif", 
                                                    num_frames=self.config.vision_config.num_frames))
                
                else:
                    image_features.append(self.image_processor(video, self.transform,
                                                video_decode_backend=self.config.vision_config.video_decode_backend,
                                                num_frames=self.config.vision_config.num_frames, fps=fps)) # iteratively preprocess image
                
            # image_features = [self.image_processor(video, self.transform,
            #                                        video_decode_backend=self.config.vision_config.video_decode_backend,
            #                                        num_frames=self.config.vision_config.num_frames) for video in videos] # iteratively preprocess image
            # # image_features = [torch.rand(3, 8, 224, 224) for image in images]
            image_features = torch.stack(image_features)

            # print("="*32)
            # print(image_features.shape)
            # print("="*32)

        if text is not None and videos is not None:
            encoding["pixel_values"] = image_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return {"pixel_values": image_features}

    def preprocess(self, images, return_tensors):
        return self.__call__(images=images, return_tensors=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
