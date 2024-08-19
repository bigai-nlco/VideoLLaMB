import os
import json
import argparse
import collections

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--dst", type=str)
args = parser.parse_args()


all_answers = []
acc, total = 0, 0
type_dct = collections.defaultdict(list)
for line_idx, line in enumerate(open(args.src)):
    res = json.loads(line)
    question_id = res['question']
    answer = res['answer']
    text = res['pred'].strip('.')
    if answer == text:
        acc += 1
    total += 1
    typeid = res['type'][0]
    type_dct[typeid].append(int(answer == text))
    # all_answers.append({
    #     "questionId": question_id, 
    #     "prediction": text,
    #     "answer": answer,
    #     })
print(text, answer)
print('Accuracy: ', acc/total)
for typeid, typeacc in type_dct.items():
    typeacc = sum(typeacc) / len(typeacc)
    print(f"Accuracy for Type {typeid}: {typeacc}")

# with open(args.dst, 'w') as f:
#     json.dump(all_answers, f)
