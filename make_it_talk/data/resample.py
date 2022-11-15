from pydub import AudioSegment
from pathlib import Path

target_freq = 16000

def resample_file(path_from, path_to):
    sound = AudioSegment.from_file(path_from)
    sound_w_new_fs = sound.set_frame_rate(target_freq)
    sound_w_new_fs.export(path_to, format="wav")

def resample(input_dir_path):
    paths = Path(input_dir_path).glob('**/*.m4a')
    for path in paths:
        resample_file(path, path[:-3] + 'wav')
