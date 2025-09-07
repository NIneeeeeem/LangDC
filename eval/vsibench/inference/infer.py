import argparse
from tqdm import tqdm
import shortuuid
from langdc.conversation import conv_templates
from langdc.model.builder import load_pretrained_vqamodel
from langdc.mm_utils import tokenizer_image_token, get_model_name_from_path
from eval.vsibench.inference.ddp import *
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import SequentialSampler
import traceback
import time
import transformers
latency_list = []


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)


def eval_model(args):
    # Model
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
    cap_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.cap_light_llm,
        padding_side="right",
        use_fast=False,
    )

    vision_tower = model.get_vision_tower()
    vision_tower.load_model(model.config.mm_vision_tower)
    video_processor = vision_tower.image_processor

    image_vision_tower = model.get_image_vision_tower()
    image_vision_tower.load_model()
    image_processor = image_vision_tower.image_processor

    model = model.to("cuda")

    dataset = EvalDatasetVSIBench(args.qa_path, args.video_folder, image_processor,
                                 video_processor)
    # distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
    distributed_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, num_workers=4, sampler=distributed_sampler)

    for (idx, sample_set, video_frames, context_frames, slice_len) in tqdm(dataloader):
        idx, sample_set, video_frames, context_frames, slice_len = int(idx[0]), sample_set[
            0], video_frames, context_frames, int(slice_len[0])

        sample = sample_set
        qs = sample['Q'][0]
        
        clip_num = 4
        cap_prompt = '<|im_start|>system\nYou are a helpful AI assistant.<|im_end|><|im_start|>user\n'+'<image>'*clip_num +'\nPlease provide a brief description of the video, focusing on the main subjects, their actions, the background scenes.\n<|im_end|><|im_start|>assistant\n'
        cap_input_ids = tokenizer_image_token(cap_prompt, cap_tokenizer, return_tensors='pt').unsqueeze(0).cuda() # turn to 2-dimension        input_ids_caps = [cap_input_ids.clone() for _ in range(4)]
        # input_ids_caps = [cap_input_ids.clone() for _ in range(4)]
        input_ids_caps = torch.tile(cap_input_ids, (4, 1))

        try:
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * slice_len + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN * slice_len + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).cuda()
            
            # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stop_str = "<|im_end|>"
            start_time = time.time()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=torch.cat(video_frames, dim=0).half().cuda(),
                    context_images=torch.cat(context_frames, dim=0).half().cuda(),
                    input_ids_caps = input_ids_caps,
                    hidden_state_layer=args.hidden_state_layer,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=30,
                    use_cache=True,
                    repetition_penalty = 1.0)
            end_time = time.time()
            latency = end_time - start_time
            latency_list.append(latency)
            

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
            print(outputs)
            print(latency)
            print("=====================")

            ans_id = shortuuid.uuid()
            video_json_name = sample['video_name'][0].replace('/', '_')
            if len(video_json_name) > 100:
                video_json_name = video_json_name[50:]

            results = {'video_name': sample['video_name'][0],
                       "prompt": cur_prompt,
                       "pred": outputs,
                       "answer_id": ans_id,
                       "Q": sample_set['Q'][0],
                       "task_type": sample['task_type'][0],
                       "A": sample['A'][0]}
            with open(f"{args.output_dir}/{video_json_name}_{idx}.json", "w") as f:
                json.dump(results, f)
        except Exception as e:
            trace = traceback.format_exc()
            print(f"Error processing video file '{sample['video_name'][0]}': {e}")
            print("Detailed traceback:")
            print(trace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="weights/mvbench_weights/dual_branch_3b")
    parser.add_argument("--model-base", type=str, default=".cache/qwen2_5/3b_instructiont")
    parser.add_argument("--cap-light-llm", type=str, default="new_weights/vcap_weights/multi_vcap_1024_qwen2_5_merged")
    parser.add_argument("--cap_mm_projector_path", type=str, default=None)
    parser.add_argument("--mm_projector_path", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="/ssd1/wxc/video_data/VSI-Bench")
    parser.add_argument("--qa_path", type=str, default="/ssd1/wxc/video_data/VSI-Bench/test-00000-of-00001.parquet")
    parser.add_argument("--output-dir", type=str, default="eval_json/langdc/mvbench_eval")
    parser.add_argument("--conv-mode", type=str, default="qwen2_cap")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--hidden_state_layer", type=int, default=-1)
    parser.add_argument("--is_pool_branch", type=bool, default=False)
    parser.add_argument("--is_cap_branch", type=bool, default=False)
    parser.add_argument("--use_subtitles", type=bool, default=False)

    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--debug', default=False, type=bool)

    args = parser.parse_args()

    init_distributed_mode(args)

    os.makedirs(args.output_dir, exist_ok=True)

    eval_model(args)
