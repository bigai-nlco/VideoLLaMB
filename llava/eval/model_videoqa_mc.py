import math
import os
import argparse
import json

import torch
import transformers
from tqdm import tqdm
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import DEFAULT_X_START_TOKEN, DEFAULT_X_TOKEN, DEFAULT_X_END_TOKEN, X_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, tokenizer_x_token, KeywordsStoppingCriteria
from llava.vid_utils import read_videos
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.train.train import smart_tokenizer_and_embedding_resize

OPTIONS = ["A", "B", "C", "D", "E"]
# OPTIONS = ["(1)", "(2)", "(3)", "(4)", "(5)"]

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    return parser.parse_args()

def get_model_output(model, video_processor, tokenizer, video, qs, args):
    if model.config.mm_use_x_start_end:
        qs = DEFAULT_X_START_TOKEN['VIDEO'] + DEFAULT_X_TOKEN['VIDEO'] + DEFAULT_X_END_TOKEN['VIDEO'] + '\n' + qs
    else:
        qs = DEFAULT_X_TOKEN['VIDEO'] + '\n' + qs

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # print('==========prompt==========')
    # print(prompt)
    # print('==========prompt==========')

    # video = read_videos(video, 32)
    # video_tensor = video_processor.preprocess(list(video), return_tensors='pt')['pixel_values'][0].half().to(args.device)
    video_tensor = video_processor(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    # print(video_tensor.shape)
    input_ids = tokenizer_x_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).to(args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    '''
    images (X_modalities) [
            [img_feature, img_feature, video_feature, audio_feature],
            ['image', 'image', 'video', 'audio']
            ]
    '''

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
            cache_position=None,
            stopping_criteria=[stopping_criteria],
            # output_scores=True,
            # return_dict_in_generate=True,
            )


    # input_token_len = input_ids.shape[1]
    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    # if n_diff_input_output > 0:
    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    # print(outputs)
    
    # constrained_ids = tokenizer(' '.join(OPTIONS))["input_ids"][1:]
    # # print(input_ids)
    # logits = output_ids.scores[1]
    # constrained_logits = logits[:, constrained_ids]
    # index = constrained_logits.squeeze().argmax().item()
    # outputs = OPTIONS[index]

    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.num_frames)
    model = model.to(args.device)

    # Load both ground truth file containing questions and answers
    # with open(args.gt_file_question) as file:
    #     gt_questions = json.load(file)
    # with open(args.gt_file_answers) as file:
    #     gt_answers = json.load(file)


    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.gt_file_answers, "r"))
    gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results


    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    acc = 0
    
    for sample in tqdm(gt_questions):
        video_name = sample['video_name']
        idx = sample['question_id']
        answer = gt_answers[index]['answer']
        if "type" in gt_answers[index]:
            typeid = gt_answers[index]["type"]
        else:
            typeid = None
        index += 1

        options = sample['option']

        # # sevila prompt
        # question = sample['question']
        # question += '\n'
        # for i, op in enumerate(options.values()):
        #     option = "Option " + OPTIONS[i]
        #     question += f" {option}: {op}\n"
        #     # question += f" {OPTIONS[i]}. {op}\n"
        # question += " Select the correct answer from the options. Answer with the option's letter only. "
        # # question += " Answer with the option's letter from the given choices directly. "
        
        # # gemini prompt
        # question = sample['question']
        # prompt = "You will be given a question about a video and five possible answer options, where C refers to the person wearing the camera. You will be provided frames from the video, sampled evenly across the video."
        # question = prompt + "\n" + "Question: " + question + '\n'
        # question += "Possible answer choices:\n"
        # for i, op in enumerate(options.values()):
        #     option = OPTIONS[i]
        #     question += f"{option} {op}\n"
        #     # question += f" {OPTIONS[i]}. {op}\n"
        # question += ' Output the final answer in the format "(X)" where X is the correct digit choice. DO NOT OUTPUT with the full answer text or any other words.'
        
        # llava prompt
        question = sample['question']
        question += "\n"
        for i, op in enumerate(options.values()):
            option = OPTIONS[i]
            question += f"{option}. {op}\n"
        question += "Answer with the option's letter from the given choices directly."


        # # videochat prompt
        # question = sample['question']
        # question += "\nOptions:\n"
        # for i, op in enumerate(options.values()):
        #     option = OPTIONS[i]
        #     question += f"({option}) {op}\n"
        # question += "Answer: "

        answer = OPTIONS[answer]
        sample_set = {'id': idx, 'question': question, 'answer': answer}
        if typeid: sample_set["type"] = typeid

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                # try:
                # Run inference on the video and add the output to the list
                output = get_model_output(model, processor['VIDEO'], tokenizer, video_path, question, args).split('.')[0]
                if output == answer: acc += 1
                # if index % 1 == 0:
                #     print('=================')
                #     # print(question) 
                #     print("predict: ", output)
                #     print("answer: ", answer)
                #     print('=================')
                sample_set['pred'] = output
                output_list.append(sample_set)
                # except Exception as e:
                #     print(f"Error processing video file '{video_name}': {e}")
                ans_file.write(json.dumps(sample_set) + "\n")
                break

        # # # debug
        # if index % 5 == 0:
        #     # print("acc: ", acc/index)
        #     break
        # if index % 200 == 0:
        #     break
        #     print("acc: ", acc/index)


    ans_file.close()
    
    # Save the output list to a JSON file
    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
