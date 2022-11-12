import torch
import os
from resemblyzer import preprocess_wav, VoiceEncoder
from pydub import AudioSegment
import soundfile as sf
import numpy as np
from skimage import io, transform
import librosa
import glob
import cv2

from make_it_talk.utils.audio_utils import match_target_amplitude

def parse_img_tensor(filepath):
    img = io.imread(filepath)
    # img_resized = transform.resize(img, (256, 256))
    return torch.tensor(img)

def parse_lb_tensor(filepath):
    return torch.tensor(preprocess_wav(filepath))

def parse_sf_tensor(file_dir_path, filename):
    file_path = os.path.join(file_dir_path, filename)
    sound = AudioSegment.from_file(file_path + ".wav")

    audio_file_tmp1 = os.path.join(file_dir_path, 'tmp.wav')
    audio_file_tmp2 = os.path.join(file_dir_path, 'tmp2.wav')

    normalized_sound = match_target_amplitude(sound, -20.0)
    normalized_sound.export(audio_file_tmp1, format='wav')

    sf.write(audio_file_tmp2, sf.read(audio_file_tmp1)[0], 22050)
    audio = torch.tensor(sf.read(audio_file_tmp2)[0])

    os.remove(audio_file_tmp1)
    os.remove(audio_file_tmp2)
    return audio

def parse_video_tensor(filepath):
    video = cv2.VideoCapture(filepath)
    if (video.isOpened() == False):
        print('Unable to open video file')
        exit(0)
        
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Process Video {}, len: {}, FPS: {:.2f}, W X H: {} x {}'.format(filepath, length, fps, w, h))

    frames = []
    ret = True
    while ret:
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.resize(frame, (256, 256))
        frames.append(torch.tensor(frame))

    cv2.destroyAllWindows()
    return torch.stack(frames)

def index_to_str(idx):
    assert idx < 100000
    str_idx = str(idx)
    return '0' * (5 - len(str_idx)) + str_idx

class AudioVideoImageTensorDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, 
            audio_dir,
            video_dir,
            image_dir,
        ):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.image_dir = image_dir
    
    def __len__(self):
        return len(glob.glob1(self.image_dir, '*.jpg'))

    def __getitem__(self, idx):
        filename = index_to_str(idx)
        audio_file_path = os.path.join(self.audio_dir, filename + '.wav')
        video_file_path = os.path.join(self.video_dir, filename +'.mp4') 
        image_file_path = os.path.join(self.image_dir, filename + '.jpg') 

        speaker_emb_tens = parse_lb_tensor(audio_file_path)
        content_emb_tens = parse_sf_tensor(self.audio_dir, filename)
        image_tens = parse_img_tensor(image_file_path)
        video_tens = parse_video_tensor(video_file_path)

        sample = {
            'video': video_tens, 
            'audio_content': content_emb_tens,
            'audio_speacker': speaker_emb_tens,
            'start_image': image_tens,  
            }

        return sample