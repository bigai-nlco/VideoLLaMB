import math
import os
import argparse
import json
import string

import torch
from torch.nn import CrossEntropyLoss
import transformers
from tqdm import tqdm
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import DEFAULT_X_START_TOKEN, DEFAULT_X_TOKEN, DEFAULT_X_END_TOKEN, X_TOKEN_INDEX, IGNORE_INDEX
from llava.mm_utils import get_model_name_from_path, tokenizer_x_token, KeywordsStoppingCriteria
from llava.vid_utils import read_videos
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.train.train import smart_tokenizer_and_embedding_resize, preprocess_multimodal, preprocess

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

def get_model_output(model, video_processor, tokenizer, video, qs, options, args):
    # if model.config.mm_use_x_start_end:
    #     qs = DEFAULT_X_START_TOKEN['VIDEO'] + DEFAULT_X_TOKEN['VIDEO'] + DEFAULT_X_END_TOKEN['VIDEO'] + '\n' + qs
    # else:
    #     qs = DEFAULT_X_TOKEN['VIDEO'] + '\n' + qs

    # conv_mode = "llava_v1"
    # args.conv_mode = conv_mode

    # conv = conv_templates[args.conv_mode].copy()
    # conv.append_message(conv.roles[0], qs)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()

    # # print('==========prompt==========')
    # # print(prompt)
    # # print('==========prompt==========')

    # # video = read_videos(video, 32)
    # # video_tensor = video_processor.preprocess(list(video), return_tensors='pt')['pixel_values'][0].half().to(args.device)
    # video_tensor = video_processor(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    # # print(video_tensor.shape)
    # input_ids = tokenizer_x_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).to(args.device)

    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # keywords = [stop_str]
    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # '''
    # images (X_modalities) [
    #         [img_feature, img_feature, video_feature, audio_feature],
    #         ['image', 'image', 'video', 'audio']
    #         ]
    # '''

    # argmin actions loss
    input_ids = []
    labels = []
    for option in options:
        # print(option)
        sources = [[{"from":"human", "value": DEFAULT_X_TOKEN["VIDEO"]+"\n"+qs}, {"from":"gpt", "value":option}]]
        data_dict = preprocess(
            sources,
            tokenizer,
            'VIDEO'
        )
        input_ids.append(data_dict["input_ids"][0].to(args.device))
        labels.append(data_dict["labels"][0].to(args.device))
    
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=IGNORE_INDEX
    )
    input_ids = input_ids[:, :tokenizer.model_max_length]
    labels = labels[:, :tokenizer.model_max_length]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    video = video_processor(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    X, X_modalities = [], []
    for option in options:
        X.append(video)
        X_modalities.append('VIDEO')

    with torch.inference_mode():
        all_logits = model(
            input_ids,
            attention_mask,
            labels=None,
            X=X,
            X_modalities=X_modalities,
            X_sizes=[None]*len(options),
            return_dict=True
        )["logits"]

        _, _, _, _, _, labels = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=labels,
            X=X,
            X_modalities=X_modalities,
            X_sizes=[None]*len(options)
        )

        shift_logits = all_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(all_logits.size(0), -1)
        loss_mask = loss != 0
        all_losses = loss.sum(1)/loss_mask.sum(1)

    # print(all_losses)
    
    predicted_choice_idx = torch.argmin(all_losses, dim=-1).item()
    predicted_choice = chr(ord('A')+predicted_choice_idx)


    return predicted_choice


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

        # question = sample['question']
        task_goal = sample['task_goal']
        task_goal = task_goal.strip(string.punctuation + " ").lower()
        if "goal" in task_goal:
            task_goal = task_goal.split("to", 1)[1].strip()
        words = task_goal.split()
        if words[0].endswith("ing"):
            question_pattern = "I am tasked with {}. " \
                               "The task's progress is demonstrated in the provided video. " \
                               "My current field of view is shown in the provided image. " \
                               "What should be my next action? " \
                               "Please output the most reasonable action you think, expressed in a short phrase."
        else:
            question_pattern = "My current task is to {}. " \
                               "The task's progress is demonstrated in the provided video. " \
                               "My current field of view is shown in the provided image. " \
                               "What should be my next action? " \
                               "Please output the most reasonable action you think, expressed in a short phrase."
        question = question_pattern.format(task_goal)

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
                output = get_model_output(model, processor['VIDEO'], tokenizer, video_path, question, list(options.values()), args).split('.')[0]
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

        # debug
        # # break
        # if index % 20 == 0:
        #     print("acc: ", acc/index)
            # break
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
