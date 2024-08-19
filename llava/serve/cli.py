import argparse
import requests
from io import BytesIO

import torch
from transformers import TextStreamer

from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import DEFAULT_X_START_TOKEN, DEFAULT_X_TOKEN, DEFAULT_X_END_TOKEN, X_TOKEN_INDEX
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, tokenizer_x_token, KeywordsStoppingCriteria
from llava.vid_utils import read_videos
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.train.train import smart_tokenizer_and_embedding_resize

from PIL import Image
from decord import VideoReader, cpu
import decord
decord.bridge.set_bridge('torch')




def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.num_frames, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    video_tensor = processor['VIDEO'](args.video_file, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    
    fid = True
    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        
        if fid:
            if model.config.mm_use_x_start_end:
                inp = DEFAULT_X_START_TOKEN['VIDEO'] + DEFAULT_X_TOKEN['VIDEO'] + DEFAULT_X_END_TOKEN['VIDEO'] + '\n' + inp
            else:
                inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
            fid = False
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # TODO
        input_ids = tokenizer_x_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).to(args.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                X=[video_tensor],
                X_modalities=["VIDEO"],
                X_sizes=[None],
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                # streamer=streamer,
                cache_position=None,
                stopping_criteria=[stopping_criteria])
        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        
        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        else:
            print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-frames", type=float, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
