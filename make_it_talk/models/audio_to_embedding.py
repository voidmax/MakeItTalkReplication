from resemblyzer import preprocess_wav, VoiceEncoder
from pydub import AudioSegment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
import shutil
import soundfile as sf
from math import ceil

from make_it_talk.utils.audio_utils import match_target_amplitude, extract_f0_func_audiofile, quantize_f0_interp

class AudioToEmbedding(nn.Module):
    def __init__(self, root_dir='.') -> None:
        super().__init__()
        self.speaker_embs = None
        self.content_extracter = None
        self.root_dir = root_dir 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = Generator(16, 256, 512, 16).eval().to(self.device)

    def forward(self, input):
        x_lb, x_sf = input
        
        speaker = torch.stack([self.get_speaker_embedding(x) for x in x_lb])
        content = torch.stack([self.convert_single_wav_to_autovc_input(x) for x in x_sf])

        return content, speaker

    def load_autovc_weights(self, weights_path=None):
        if weights_path is None:
            weights_path = os.path.join(self.root_dir, 'checkpoints', 'audio', 'ckpt_autovc.pth')

        g_checkpoint = torch.load(weights_path, map_location=self.device)
        self.G.load_state_dict(g_checkpoint['model'])

    def get_content_embedding(self, audio_dir_path):
        au_data = []
        ains = glob.glob1(audio_dir_path, '*.wav')
        ains = [item for item in ains if item is not 'tmp.wav']
        ains.sort()
        for ain in ains:
            os.system('ffmpeg -y -loglevel error -i {}/{} -ar 16000 {}/tmp.wav'.format(audio_dir_path, ain, audio_dir_path))
            shutil.copyfile('{}/tmp.wav', '{}/{}'.format(audio_dir_path, audio_dir_path, ain))

            au_data.append(self.convert_single_wav_to_autovc_input(audio_filename=os.path.join(audio_dir_path, ain)))

        if(os.path.isfile('{}/tmp.wav'.format(audio_dir_path))):
            os.remove('{}/tmp.wav'.format(audio_dir_path))


    def get_speaker_embedding(self, lb_tensor):
        # wav = preprocess_wav(lb_tensor)
        wav = lb_tensor.cpu().numpy()
        
        resemblyzer_encoder = VoiceEncoder(device=self.device)
        segment_len=960000
        l = len(wav) // segment_len # segment_len = 16000 * 60
        l = np.max([1, l])
        all_embeds = []
        for i in range(l):
            mean_embeds, cont_embeds, wav_splits = resemblyzer_encoder.embed_utterance(
                wav[segment_len * i:segment_len* (i + 1)], return_partials=True, rate=2)
            all_embeds.append(mean_embeds)
        all_embeds = np.array(all_embeds)
        self.speaker_embs = torch.tensor(np.mean(all_embeds, axis=0))

        return self.speaker_embs

    def convert_single_wav_to_autovc_input(self, input):

        def pad_seq(x, base=32):
            len_out = int(base * ceil(float(x.shape[0]) / base))
            len_pad = len_out - x.shape[0]
            assert len_pad >= 0
            return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad

        
        assert self.speaker_embs is not None
        emb_trg = (self.speaker_embs[np.newaxis, :]).to(self.device)

        x_real_src, f0_norm = extract_f0_func_audiofile(input, 22050, 'F')
        f0_org_src = quantize_f0_interp(f0_norm)

        ''' long split version '''
        l = x_real_src.shape[0]
        x_identic_psnt = []
        step = 4096
        for i in range(0, l, step):
            x_real = x_real_src[i:i + step]
            f0_org = f0_org_src[i:i + step]

            x_real, len_pad = pad_seq(x_real.astype('float32'))
            f0_org, _ = pad_seq(f0_org.astype('float32'))
            x_real = torch.from_numpy(x_real[np.newaxis, :].astype('float32')).to(self.device)
            emb_org = self.speaker_embs[np.newaxis, :].to(self.device)
            # emb_trg = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(self.device)
            f0_org = torch.from_numpy(f0_org[np.newaxis, :].astype('float32')).to(self.device)
            print('source shape:', x_real.shape, emb_org.shape, emb_trg.shape, f0_org.shape)

            with torch.no_grad():
                x_identic, x_identic_psnt_i, code_real = self.G(x_real, emb_org, f0_org, emb_trg, f0_org)
                x_identic_psnt.append(x_identic_psnt_i)

        x_identic_psnt = torch.cat(x_identic_psnt, dim=1)
        print('converted shape:', x_identic_psnt.shape, code_real.shape)

        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, :, :].detach()
        else:
            uttr_trg = x_identic_psnt[0, :-len_pad, :].detach()

        return uttr_trg



dim_enc = 512
dim_freq = 80
dim_f0 = 257
num_grp = 32
dim_dec = 512

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal
    
    


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        #self.dropout = nn.Dropout(0.0)
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_freq+dim_emb if i==0 else dim_enc,
                         dim_enc,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(num_grp, dim_enc))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(dim_enc, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x):
                
        for conv in self.convolutions:
            #x = self.dropout(F.relu(conv(x)))
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        #self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        return codes
      
      
        
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm = nn.LSTM(dim_neck*2+dim_emb+dim_f0, dim_dec, 3, batch_first=True)
        
        self.linear_projection = LinearNorm(dim_dec, dim_freq)

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()
        
        outputs, _ = self.lstm(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
    
    
    
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        #self.dropout = nn.Dropout(0.0)
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(dim_freq, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.GroupNorm(num_grp, 512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.GroupNorm(num_grp, 512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, dim_freq,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.GroupNorm(5, dim_freq))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            #x = self.dropout(torch.tanh(self.convolutions[i](x)))
            x = torch.tanh(self.convolutions[i](x))

        #x = self.dropout(self.convolutions[-1](x))
        x = self.convolutions[-1](x)

        return x    
    

    
    
class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()
        self.freq = freq


    def forward(self, x, c_org, f0_org=None, c_trg=None, f0_trg=None, enc_on=False):
        
        x = x.transpose(2,1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
                
        codes = self.encoder(x)
        if enc_on:
            return torch.cat(codes, dim=-1)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,self.freq,-1))
        code_exp = torch.cat(tmp, dim=1)
        
        encoder_outputs = torch.cat((code_exp, 
                                     c_trg.unsqueeze(1).expand(-1,x.size(-1),-1),
                                     f0_trg), dim=-1)
        
        mel_outputs = self.decoder(encoder_outputs)
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)


### PLUG FOR TESTING ###
class AudioToEmbeddingPlug(nn.Module):
    def __init__(self, audio_dim, out_hs, out_speaker_hs):
        super(AudioToEmbeddingPlug, self).__init__()
        self.linear = nn.Linear(audio_dim, out_hs)
        self.speaker_linear = nn.Linear(audio_dim, out_speaker_hs)

    def forward(self, input):
        x = self.linear(input)
        return x, self.speaker_linear(input).sum(1)
    