import os
import json
from tqdm import tqdm
import argparse

def check_ans(pred, gt):
    flag = False

    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[1], ' '.join(pred_list[1:])
    if pred_option.replace('.', '').replace('(', '').replace(')', '') in gt.lower():
        flag = True
    elif gt.lower() in pred_option:
        flag = True

    return flag


def main(args):
    result_files = os.listdir(args.output_dir)

    correct = 0
    total = 0
    acc_dict = {}

    for file in tqdm(result_files):
        if file.endswith('.json'):
            json_file = os.path.join(args.output_dir, file)
            json_data = json.load(open(json_file))
            video_name = json_data['video_name']
            task_type = json_data['task_type']
            pred = json_data['pred']
            gt_answer = json_data['A']
            question = json_data['Q']

            if task_type not in acc_dict:
                acc_dict[task_type] = [0, 0]  # correct, total
            acc_dict[task_type][1] += 1
            total += 1
            if check_ans(pred=pred, gt=gt_answer):
                acc_dict[task_type][0] += 1
                correct += 1

    print(f"Total Acc: {correct / total * 100 :.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="mvbench_eval")
    args = parser.parse_args()
    main(args)
