from pathlib import Path
import subprocess
from tqdm import tqdm
import torchaudio
from torchaudio.transforms import Resample

paths = Path('datasets/voxceleb2/aac').glob('**/*.m4a')
for path in tqdm(paths):
    p = str(path)
    p_new = p[:-4] + '.wav'
    subprocess.run(["ffmpeg", "-i", p, p_new])
    waveform, sample_rate = torchaudio.load(p_new)
    resamp = Resample(sample_rate, 16000)
    wav = resamp(waveform)
    torchaudio.save(p_new, wav, 16000)

