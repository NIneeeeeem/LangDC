from random import choice
from torch.utils.data import Dataset
import os
import decord
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import torch
import pandas as pd
import json
from langdc.constants import *
from eval.video_encoding import _get_rawvideo_dec, read_frame_mod, read_gif_mod

def number_to_excel_column(n):
    """
    Convert a positive integer to its corresponding Excel-style column name.
    1 -> 'A', 2 -> 'B', ..., 26 -> 'Z', 27 -> 'AA', 28 -> 'AB', ...
    """
    if n < 0:
        raise ValueError("Input must be a positive integer.")
    if n == 0:
        return 'A'
    elif n == 1:
        return 'B'
    elif n == 2:
        return 'C'
    elif n == 3:
        return 'D'  
    elif n == 4:
        return 'E'
    elif n == 5:
        return 'F'
    else:
        # For numbers greater than 5, we can use a simple conversion
        # to get the corresponding letter.
        # This is a simplified version and may not cover all cases.
        return chr(ord('A') + n - 6)

def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(':')
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds

def load_video(video_file, duration, max_num_frames=16):
    from decord import VideoReader
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]
    
    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]
    
    return [Image.fromarray(fr).convert("RGB") for fr in frames], frame_timestamps

def insert_subtitles(subtitles):
    interleaved_list = []
    cur_i = 0
    
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            subtitle_text = subtitle["text"]
        else:
            subtitle_text = subtitle["line"]

        interleaved_list.append(subtitle_text)

    return interleaved_list
        
def insert_subtitles_into_frames(frames, frame_timestamps, subtitles, 
                                 starting_timestamp_for_subtitles, duration):
    interleaved_list = []
    cur_i = 0
    
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration
                
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles
            
            
            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles
            
            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]

        
        for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
                if frame_timestamp <= subtitle_timestamp:
                    #print("frame:", frame_timestamp)
                    interleaved_list.append(frame)
                    cur_i += 1
                else:
                    break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame, frame_timestamp in zip(frames, frame_timestamps):
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break
        #
        if covering_frames:
            #print("subtitle:", subtitle_timestamp, start, end)
            interleaved_list.append(subtitle_text)
        else:
            pass
            #print("leaving out subtitle:", start, end)
        
    for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
        #print(frame_timestamp)
        interleaved_list.append(frame)
        
    return interleaved_list
    
class LongVideoBenchDataset(Dataset):
    def __init__(self,
                 data_path,
                 annotation_file,
                 image_processor,
                 video_processor,
                 max_num_frames=256,
                 insert_text=True,
                 insert_frame=True,
                ):
        super().__init__()
        self.data_path = data_path
        self.insert_text = insert_text

        with open(os.path.join(data_path, annotation_file)) as f:
            self.data = json.load(f)
        self.max_num_frames = max_num_frames
        self.image_processor = image_processor
        self.video_processor = video_processor   
    
    def __getitem__(self, index):
        di = self.data[index]
        prefix = di["video_path"]
        video_name = di["video_path"]
        video_path = os.path.join(self.data_path, "videos", di["video_path"])
        video_frames, context_frames, slice_len = (
                        _get_rawvideo_dec(video_path, self.image_processor, self.video_processor,
                                          max_frames=NUM_FRAMES, image_resolution=224,
                                          num_video_frames=NUM_FRAMES, num_context_images=NUM_CONTEXT_IMAGES))
        # {"video_id": "86CxyhFV9MI", "question": "In the video, which subtitles appear at the same time as the man with black hair, dressed in grey clothes with black sleeves, on stage?", "question_wo_referring_query": "Which subtitles appear at the same time?", "candidates": ["promisc has come to an end, in and run away countless times, i was just scared, i still", "run away countless times, i was just scared, i still and front of our crown, like a world of souls,", "promisc has come to an end, in and front of our crown, like a world of souls,", "promisc has come to an end, in and captain of the godson, three three three three three three"], "correct_choice": 1, "position": [854, 948, 1373], "topic_category": "NP-News-Programs", "question_category": "TOS", "level": "L2-Relation", "id": "86CxyhFV9MI_0", "video_path": "86CxyhFV9MI.mp4", "subtitle_path": "86CxyhFV9MI_en.json", "duration_group": 600, "starting_timestamp_for_subtitles": 0, "duration": 190.16, "view_count": 259852}
            
        # with open(os.path.join(self.data_path, "subtitles", di["subtitle_path"])) as f:
        #     subtitles = json.load(f)
        # inputs = []
        # if self.insert_text:
        #     inputs = insert_subtitles_into_frames(frames, frame_timestamps, subtitles, di["starting_timestamp_for_subtitles"], di["duration"])
        # else:
        #     inputs = frames

        ##### YOU MAY MODIFY THE FOLLOWING PART TO ADAPT TO YOUR MODEL #####
        # inputs += ["Question: " + di["question"]]
        # inputs += [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(di["candidates"])]
        # inputs += ["Answer with the option's letter from the given choices directly."]
        # print(inputs)
        choices = [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(di["candidates"])]
        # question = "The video shows that "+subtitles+". Question: " + di["question"] + " ".join(choices) +"Answer with the option's letter from the given choices directly."
        question = "Question: " + di["question"] + " ".join(choices) +"Answer with the option's letter from the given choices directly."
        # print(question)
        sample_set = {}
        # question, answer = qa_template(sample)
        
        answer = di["correct_choice"]
        sample_set['video_name'] = f'{prefix}_{video_name}'
        sample_set['Q'] = question
        sample_set['A'] = number_to_excel_column(int(answer))
        sample_set['task_type'] = di["question_category"]
        sample_set['duration'] = di["duration"]
        ##### YOU MAY MODIFY THE PREVIOUS PART TO ADAPT TO YOUR MODEL #####

        ##### CORRECT CHOICE WILL BE "@" FOR TEST SET SAMPLES #####
        # return {"inputs": inputs, "correct_choice": chr(ord("A")+di.get("correct_choice", -1)), "id": di["id"]}
        return index, [sample_set], video_frames, context_frames, slice_len
    
    def __len__(self):
        return len(self.data)
    
    def get_id(self, index):
        return self.data[index]["id"]
        
if __name__ == "__main__":
    db = LongVideoBenchDataset("../", "lvb_val.json")
    for i in range(10):
        print([ele for ele in db[i]["inputs"] if not isinstance(ele, str)])