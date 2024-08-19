import openai
import os
import argparse
import json
import ast
import time
from multiprocessing.pool import Pool
from tqdm import tqdm

import time, random
from openai import AzureOpenAI, OpenAI

REGIONS = {
        "gpt-35-turbo-0125": ["canadaeast", "northcentralus", "southcentralus"],
        "gpt-4-0125-preview": ["eastus", "eastus2", "northcentralus", "southcentralus"],
        "gpt-4-vision-preview": ["australiaeast", "japaneast", "westus"]
    }

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", default=r'', help="The path to file containing prediction.")
    parser.add_argument("--output_dir", default=r'', help="The path to save annotation json files.")
    parser.add_argument("--output_json", default=r'', help="The path to save annotation final combined json file.")
    parser.add_argument("--api_key", default="", help="OpenAI API key.")
    parser.add_argument("--api_base", default="", type=str, help="OpenAI API base.")
    parser.add_argument("--num_tasks", default=1, type=int, help="Number of splits.")
    args = parser.parse_args()
    return args



def openai_api_1(api_key, api_base, model, messages):
    client = OpenAI(
        api_key=api_key,
        api_base=api_base
    )
    response = client.chat.completion.create(
        model=model,
        messages=messages
    )
    response = response.choices[0].message.content
    return response

def openai_api_0(api_key, api_base, model, messages):
    # Compute the correctness score
    openai.api_key = api_key
    openai.api_base = api_base
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    # Convert response to a Python dictionary.
    response_message = completion["choices"][0]["message"]["content"]
    return response_message

def azureopenai_api(api_key, api_base, model, messages):
    
    
    # API_BASE = "https://api.tonggpt.mybigai.ac.cn/proxy"

    region = random.choice(REGIONS[model])
    endpoint = f"{api_base}/{region}"
    client = AzureOpenAI(
        api_key = api_key,
        api_version = "2024-02-01",
        azure_endpoint = endpoint,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        response = response.choices[0].message.content
        # print(response)
    except Exception as e: # FIXME: prevent azure content filter
        if e.code == 'content_filter':
            response = "{'pred': '', 'score': 0.0}"
            print("stupid azure, ", e)
        else:
            raise e

    
    # print(response.model_dump_json(indent=2))
    # print(response.choices[0].message.content)
    
    return response
    # API_KEY = "ba4cc9feab3e89b99ab1e226c1603d46"
    # MODEL = "gpt-35-turbo-0125"
    # print(azure_api(API_KEY, MODEL))

def annotate(prediction_set, caption_files, output_dir, args):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    # configure


    for file in caption_files:
        key = file[:-5] # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['pred']
        # if len(set(pred.split())) < len(pred.split()) // 2: pred = pred.split()[0] # prevent from repetitive error
        try:
            # Compute the correctness score
            messages = [
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for evaluating the contextual understanding of generative outputs for video-based question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if the generated response aligns with the overall context of the video content. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Evaluate whether the predicted answer aligns with the overall context of the video content. It should not provide information that is out of context or misaligned.\n"
                            "- The predicted answer must capture the main themes and sentiments of the video.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Provide your evaluation of the contextual understanding of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a contextual understanding score where the contextual understanding score is an integer value between 0 and 5, with 5 indicating the highest level of contextual understanding. "
                            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is contextual understanding score in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {''score': 4.8}."
                    }
                ]
            
            if 'bigai' in args.api_base:
                response_message = azureopenai_api(args.api_key, args.api_base, model='gpt-35-turbo-0125', messages=messages)
            else:
                # Set the OpenAI API key.
                # response_message = openai_api_0(args.api_key, args.api_base, model="gpt-3.5-turbo-0125", messages=messages)
                response_message = openai_api_1(args.api_key, args.api_base, model="gpt-3.5-turbo-0125", messages=messages)
            
            # Convert response to a Python dictionary.
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = [response_dict, qa_set]

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    file = open(args.pred_path)
    # new_pred_contents = [eval(i.strip()) for i in file.readlines()]
    new_pred_contents = []
    video_id_counts = {}
    for i in file.readlines():
        sample = eval(i.strip())
        video_id = sample["video_name"]
        video_id_counts[video_id] = video_id_counts.get(video_id, 0) + 1
        id = f"{video_id}_{video_id_counts[video_id]}"
        sample['id'] = id
        new_pred_contents.append(sample)

    '''
    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        video_id = sample['video_name']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)
    '''
    # Generating list of id's and corresponding files
    id_list = [x['id'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['id']
        question = sample['Q']
        answer = sample['A']
        # pred = sample['pred']
        pred = sample['pred'].split("</s>")[0]
        # pred = sample['pred'].split('.')[0]
        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set

    num_tasks = args.num_tasks

    start = time.time()
    # While loop to ensure that all captions are processed.
    while True:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir, args) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")

    end = time.time()
    eval_hs = (end-start) // 3600
    eval_mins = (end-start) % 3600 // 60
    print(f"Evaluation takes {eval_hs} hours {eval_mins} minutes")


    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    for key, result in tqdm(combined_contents.items()):
        try:
            # Computing score
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score
        except:
            print(result)

    average_score = score_sum / count
    print("Average score for contextual understanding:", average_score)


if __name__ == "__main__":
    main()

