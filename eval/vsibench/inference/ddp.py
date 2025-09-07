import json
import pandas as pd
from torch.utils.data import Dataset
import torch
import subprocess
from langdc.constants import *
from eval.video_encoding import _get_rawvideo_dec, read_frame_mod, read_gif_mod

def extract_words(text, num):
    words = text.split()
    first_1000_words = words[:num]
    return ' '.join(first_1000_words)

class EvalDatasetVSIBench(Dataset):
    def __init__(self, annotation_path, video_dir, image_processor, video_processor):
        self.gt_contents = []
        self.table = pd.read_parquet(annotation_path)
        task_types = [
            "object_rel_direction_hard",
            "object_rel_direction_easy",
            "route_planning",
            "obj_appearance_order",
            "object_rel_distance"
        ]
        self.table = self.table[self.table["question_type"].isin(task_types)]
        self.video_dir = video_dir
        self.image_processor = image_processor
        self.video_processor = video_processor

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        sample = self.table.iloc[idx]

        task_type = sample['question_type']

        # {'obj_appearance_order', 'object_counting', 'object_rel_direction_hard', 'object_abs_distance', 'object_size_estimation', 'object_rel_direction_easy', 'room_size_estimation', 'object_rel_direction_medium', 'route_planning', 'object_rel_distance'}
        video_name = sample['scene_name']+".mp4"
        prefix = sample['dataset']
        video_path = os.path.join(self.video_dir, prefix, video_name)
        if os.path.exists(video_path):
            video_frames, context_frames, slice_len = (
                        _get_rawvideo_dec(video_path, self.image_processor, self.video_processor,
                                          max_frames=NUM_FRAMES, image_resolution=224,
                                          num_video_frames=NUM_FRAMES, num_context_images=NUM_CONTEXT_IMAGES))
        else:
            video_frames, slice_len = "None", 0
            print('Video not found:', video_path)
        

        sample_set = {}
        question, answer = qa_template(sample)
        sample_set['video_name'] = f'{prefix}_{video_name}'
        sample_set['Q'] = question
        sample_set['A'] = answer
        sample_set['task_type'] = task_type

        return idx, [sample_set], video_frames, context_frames, slice_len


def qa_template(data):
    question = f"Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option. Question: {data['question']}\n"
    question += "Options:\n"
    answer = data['ground_truth']
    answer_idx = -1
    for idx, c in enumerate(data['options']):
        question += f"({chr(ord('A') + idx)}) {c}\n"
        if c == answer:
            answer_idx = idx
    question = question.rstrip()
    answer = f"({chr(ord('A') + answer_idx)}) {answer}"

    # Add the instruction to question
    question_prompt = "\nOnly give the best option."  # to change
    question += question_prompt

    return question, answer


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
        print('Using distributed mode: 1')
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '3460')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
        print('Using distributed mode: slurm')
        print(f"world: {os.environ['WORLD_SIZE']}, rank:{os.environ['RANK']},"
              f" local_rank{os.environ['LOCAL_RANK']}, local_size{os.environ['LOCAL_SIZE']}")
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
