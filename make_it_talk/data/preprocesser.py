import torch
import os
from pathlib import Path
from resemblyzer import preprocess_wav, VoiceEncoder
from pydub import AudioSegment
import soundfile as sf
import numpy as np
# from skimage import io, transform
# import librosa
# import glob
# import cv2
from math import ceil

from make_it_talk.utils.audio_utils import match_target_amplitude, extract_f0_func_audiofile, quantize_f0_interp
from make_it_talk.models.audio_to_embedding import Generator

class AudioPreprocesser:
    def __init__(self, 
            root_audio_dir= os.path.join('datasets', 'voxceleb2', 'aac'),
            # video_dir='mp4',
            checkpoints_dir='.'
    ):
        self.root_audio_dir = root_audio_dir
        # self.video_dir = video_dir
        self.checkpoints_dir = checkpoints_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(16, 256, 512, 16).eval().to(self.device)
        self.emb_obama = torch.zeros(256)

    def load_autovc_weights(self):
        weights_path = os.path.join(self.checkpoints_dir, 'checkpoints', 'audio', 'ckpt_autovc.pth')
        print('pretrained weights: ', weights_path)

        g_checkpoint = torch.load(weights_path, map_location=self.device)
        self.generator.load_state_dict(g_checkpoint['model'])

    def load_obama_embs(self):
        obama_path = os.path.join(self.checkpoints_dir, 'checkpoints', 'audio', 'obama_emb.txt')

        print('Obama embeds: ', obama_path)
        emb = np.loadtxt(obama_path)
        self.emb_obama = torch.from_numpy(emb.astype('float32')).to(self.device)

    def Parse(self):
        files = list(Path(self.root_audio_dir).glob('**/*.wav'))
        print('Found ', len(list(Path(self.root_audio_dir).glob('**/*.wav'))), ' folders')
        print('Root path: ', str(Path(self.root_audio_dir)))
        # for filename in files:
        #     print('filename: ', filename)
        # files = glob.glob1(self.root_audio_dir, '*.wav')
        self.load_autovc_weights()
        self.load_obama_embs()
        for filename in files:
            file_name = str(filename)
            if file_name.endswith('tmp.wav'):
                continue
            spk_tens = self.parse_speacker_tensor(file_name)
            cont_tens = self.parse_content_tensor(file_name)
            # print('save paths: ' + os.path.join(self.root_audio_dir, 'speacker_' + file_name[:-3] + 'pt'))
            # print('shape: ', spk_tens)
            torch.save(spk_tens, file_name[:-4] + '_speacker.pt')
            torch.save(cont_tens,  file_name[:-4] + '_content.pt')

    def parse_speacker_tensor(self, filename):
        file_path = filename
        wav = preprocess_wav(file_path)
        
        resemblyzer_encoder = VoiceEncoder()
        segment_len=960000
        l = len(wav) // segment_len # segment_len = 16000 * 60
        l = np.max([1, l])
        all_embeds = []
        for i in range(l):
            mean_embeds, cont_embeds, wav_splits = resemblyzer_encoder.embed_utterance(
                wav[segment_len * i:segment_len* (i + 1)], return_partials=True, rate=2)
            all_embeds.append(mean_embeds)
        all_embeds = np.array(all_embeds)

        return torch.tensor(np.mean(all_embeds, axis=0))

    def parse_content_tensor(self, filename):

        def pad_seq(x, base=32):
                len_out = int(base * ceil(float(x.shape[0]) / base))
                len_pad = len_out - x.shape[0]
                assert len_pad >= 0
                return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad

        file_path = filename
        sound = AudioSegment.from_file(file_path)

        audio_file_tmp1 = str(filename)[:-4] + 'tmp.wav'

        normalized_sound = match_target_amplitude(sound, -20.0)
        normalized_sound.export(audio_file_tmp1, format='wav')

        # sf.write(audio_file_tmp2, sf.read(audio_file_tmp1)[0], 22050)
        audio_read = sf.read(audio_file_tmp1)

        emb_trg = (self.emb_obama[np.newaxis, :]).to(self.device)
        x_real_src, f0_norm = extract_f0_func_audiofile(audio_read[0], audio_read[1], 'F')
        f0_org_src = quantize_f0_interp(f0_norm)

        l = x_real_src.shape[0]
        x_identic_psnt = []
        step = 4096
        for i in range(0, l, step):
            x_real = x_real_src[i:i + step]
            f0_org = f0_org_src[i:i + step]

            x_real, len_pad = pad_seq(x_real.astype('float32'))
            f0_org, _ = pad_seq(f0_org.astype('float32'))
            x_real = torch.from_numpy(x_real[np.newaxis, :].astype('float32')).to(self.device)
            emb_org = self.emb_obama[np.newaxis, :].to(self.device)
            # emb_trg = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(device)
            f0_org = torch.from_numpy(f0_org[np.newaxis, :].astype('float32')).to(self.device)

            with torch.no_grad():
                x_identic, x_identic_psnt_i, code_real = self.generator(x_real, emb_org, f0_org, emb_trg, f0_org)
                x_identic_psnt.append(x_identic_psnt_i)

        x_identic_psnt = torch.cat(x_identic_psnt, dim=1)

        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, :, :]
        else:
            uttr_trg = x_identic_psnt[0, :-len_pad, :]

        os.remove(audio_file_tmp1)
        return uttr_trg

