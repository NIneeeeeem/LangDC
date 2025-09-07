import argparse
from tqdm import tqdm
import shortuuid
from langdc.conversation import conv_templates
from langdc.model.builder import load_pretrained_vqamodel
from langdc.mm_utils import tokenizer_image_token, get_model_name_from_path
import os, json, torch
from langdc.constants import *
from eval.video_encoding import _get_rawvideo_dec
from eval.vcgbench.inference.ddp import init_distributed_mode
from torch.utils.data import Dataset, DataLoader, DistributedSampler, SequentialSampler
import subprocess

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


class EvalDatasetQA(Dataset):
    def __init__(self, q_path,a_path, video_dir, image_processor, video_processor):
        with open(q_path) as file:
            self.gt_questions = json.load(file)
        with open(a_path) as file:
            self.gt_answers = json.load(file)
        self.video_dir = video_dir
        self.image_processor = image_processor
        self.video_processor = video_processor

        self.video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __len__(self):
        return len(self.gt_questions)

    def __getitem__(self, idx):
        sample = self.gt_questions[idx]
        video_name = sample['video_name']
        sample_set = sample
        answer = self.gt_answers[idx]['answer']
        video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
        for fmt in self.video_formats:  # Added this line
            temp_path = os.path.join(self.video_dir, f"{video_name}{fmt}")
            
            if os.path.exists(temp_path):
                # print(temp_path)
                video_path = temp_path
                break
            anet_temp_path = os.path.join(self.video_dir, f"v_{video_name}{fmt}")
            if os.path.exists(anet_temp_path):
                video_path = anet_temp_path
                break

        # Load the video file
        # video_path = os.path.join(self.video_dir, video_path)

        # Check if the video exists
        if os.path.exists(video_path):  # Modified this line
            video_frames, context_frames, slice_len = _get_rawvideo_dec(video_path, self.image_processor,
                                                                        self.video_processor,
                                                                        max_frames=NUM_FRAMES,
                                                                        image_resolution=224,
                                                                        num_video_frames=NUM_FRAMES,
                                                                        num_context_images=NUM_CONTEXT_IMAGES)

        else:
            print(f'Video {video_path} not found')
            video_frames, context_frames, slice_len = "None", "None", 0

        return idx, sample_set,answer, video_frames, context_frames, slice_len

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--model-path", type=str, default="results/VQA_3b_0923_action_layer_3")
    parser.add_argument("--model-base", type=str, default=".cache/qwen2_5/3b_instruction")
    parser.add_argument("--cap-light-llm", type=str, default=".cache/qwen2_5/stage1_weight_0920")
    parser.add_argument("--mm_projector_path", type=str, default="results/multi_vcap_0920_qwen2_5/non_lora_trainables.bin")
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument("--cap_mm_projector_path", type=str, default=None)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    # parser.add_argument("--question-file", type=str, default=".cache/instruction_data/annotations/caption_videochat.json")
    parser.add_argument("--conv-mode", type=str, default="qwen2_cap")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--hidden_state_layer", type=int, default=-1)
    parser.add_argument("--is_pool_branch", type=bool, default=False)
    parser.add_argument("--is_cap_branch", type=bool, default=False)

    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int)
    
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)


    return parser.parse_args()


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_vqamodel(model_path, args.model_base, model_name, args.cap_light_llm, args.mm_projector_path, args.cap_mm_projector_path)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    vision_tower.load_model(model.config.mm_vision_tower)
    video_processor = vision_tower.image_processor

    image_vision_tower = model.get_image_vision_tower()
    image_vision_tower.load_model()
    image_processor = image_vision_tower.image_processor

    model = model.to("cuda")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode
    stop_str = "<|im_end|>"

    # video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    clip_num = 4
        # if light_llm is not qwen2: change here                                                                              Please provide a brief description of the video, focusing on the main subjects, their actions, the background scenes.
    cap_prompt = '<|im_start|>system\nYou are a helpful AI assistant.<|im_end|><|im_start|>user\n'+'<image>'*clip_num +'\nPlease provide a brief description of the video, focusing on the main subjects, their actions, the background scenes.\n<|im_end|><|im_start|>assistant\n'
        # cap_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "+"<image>"*clip_num +"\nPlease provide a description of the video, focusing on the main subjects, their actions, the background scenes in 50 words.\n ASSISTANT:"
    cap_input_ids = tokenizer_image_token(cap_prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda() # turn to 2-dimension        input_ids_caps = [cap_input_ids.clone() for _ in range(4)]
        # input_ids_caps = [cap_input_ids.clone() for _ in range(4)]
    input_ids_caps = torch.tile(cap_input_ids, (4, 1))
    # for sample in tqdm(gt_questions):
    dataset = EvalDatasetQA(args.gt_file_question,args.gt_file_answers, args.video_dir, image_processor, video_processor)
    distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, num_workers=4, sampler=distributed_sampler)

    for (idx, sample_set,answer, video_frames, context_frames, slice_len) in tqdm(dataloader):
        idx, sample_set,answer, video_frames, context_frames, slice_len = int(idx[0]), sample_set,answer, video_frames, context_frames, int(slice_len[0])
        sample = sample_set
        video_name = sample['video_name']
        question = sample['question']
        id = sample['question_id']
        # answer = gt_answers[index]['answer']
        index += 1

        sample_set = {'id': id[0], 'question': question[0], 'answer': answer[0]}

        try:
            conv = conv_templates[args.conv_mode].copy()
            question = DEFAULT_IMAGE_TOKEN * slice_len + '\n' +question[0]
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            # cap_start_inputs_ids = tokenizer_image_token(conv.roles[1], tokenizer, return_tensors='pt')
            # '<|system|>\nYou are a helpful AI assistant.<|im_end|><|im_start|>user\n<image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image>\nWrite an exhaustive depiction of the given video, capturing its essence and key moments.<|im_end|><|im_start|>assistant\n'
            # change inputs_ids
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).cuda()
            # Run inference on the video and add the output to the list
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images = torch.cat(video_frames, dim=0).half().cuda(),
                    # images = torch.cat([tensor.unsqueeze(0) for tensor in video_frames], dim=0).half().cuda(),
                    context_images = torch.cat(context_frames, dim=0).half().cuda(),
                    # context_images = torch.cat([tensor.unsqueeze(0) for tensor in context_frames], dim=0).half().cuda(),
                    input_ids_caps = input_ids_caps,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    hidden_state_layer=args.hidden_state_layer,
                    max_new_tokens=1024,
                    use_cache=True,
                    repetition_penalty = 1.0)
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            outputs = outputs.replace("<|im_end|>", '')
            outputs = outputs.strip()
            sample_set['pred'] = outputs
            output_list.append(sample_set)
            with open(f"{args.output_dir}/{id[0]}_{idx}.json", "w") as f:
                json.dump(sample_set, f)
        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")

    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    init_distributed_mode(args)
    run_inference(args)