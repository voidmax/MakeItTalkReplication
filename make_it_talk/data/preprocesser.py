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
from math import ceil

from make_it_talk.utils.audio_utils import match_target_amplitude, extract_f0_func_audiofile, quantize_f0_interp
from make_it_talk.models.audio_to_embedding import Generator



class AudioPreprocesser:
    def __init__(self, 
            input_dir,
            output_dir,
            root_dir='.'
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.root_dir = root_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(16, 256, 512, 16).eval().to(self.device)
        self.emb_obama = torch.zeros(256)

    def load_autovc_weights(self):
        weights_path = os.path.join(self.root_dir, 'checkpoints', 'audio', 'ckpt_autovc.pth')

        g_checkpoint = torch.load(weights_path, map_location=self.device)
        self.generator.load_state_dict(g_checkpoint['model'])

    def load_obama_embs(self):
        obama_path = os.path.join(self.root_dir, 'checkpoints', 'audio', 'obama_emb.txt')
        emb = np.loadtxt(obama_path)
        self.emb_obama = torch.from_numpy(emb.astype('float32')).to(self.device)

    def Parse(self):
        files = glob.glob1(self.input_dir, '*.wav')
        self.load_autovc_weights()
        self.load_obama_embs()
        for file_name in files:
            if file_name.startswith('tmp'):
                continue
            spk_tens = self.parse_speacker_tensor(self.input_dir, file_name)
            cont_tens = self.parse_content_tensor(self.input_dir, file_name)
            torch.save(spk_tens, os.path.join(self.output_dir, 'speacker', file_name[:-3] + 'pt'))
            torch.save(cont_tens, os.path.join(self.output_dir, 'content', file_name[:-3] + 'pt'))

    def parse_speacker_tensor(self, file_dir_path, filename):
        file_path = os.path.join(file_dir_path, filename)
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

    def parse_content_tensor(self, file_dir_path, filename):

        def pad_seq(x, base=32):
                len_out = int(base * ceil(float(x.shape[0]) / base))
                len_pad = len_out - x.shape[0]
                assert len_pad >= 0
                return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad

        file_path = os.path.join(file_dir_path, filename)
        sound = AudioSegment.from_file(file_path)

        audio_file_tmp1 = os.path.join(file_dir_path, 'tmp.wav')

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

