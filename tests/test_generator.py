import pytest
import librosa
import torch
from skimage import io


def check_shape(tensor, constants, names):
    assert len(tensor.shape) == len(names), "Dimension mismatch"
    for sz, n in zip(tensor.shape, names):
        if n == "???":
            continue
        elif n not in constants:
            constants[n] = sz
        elif sz != constants[n]:
            err_msg = "Wrong tensor size value (%s). Expected %d, got %d" % (n, constants[n], sz)
            raise ValueError(err_msg)


def run_audio_to_embedding(input, constants):
    from make_it_talk import AudioToEmbedding
    model = AudioToEmbedding()

    check_shape(input, constants, ["batch_size", "???"])

    output = model(input)

    assert len(output) == 2
    check_shape(output[0], constants, ["batch_size", "frames", "audio_dim"])
    check_shape(output[1], constants, ["batch_size", "speaker_dim"])
    
    return output


def run_LSTM_speech_content(input, constants):
    from make_it_talk import LSTMSpeechContent
    model = LSTMSpeechContent()

    check_shape(input, constants, ["batch_size", "frames", "audio_dim"])

    output = model(input)

    check_shape(output, constants, ["batch_size", "frames", "hidden_size_1"])

    return output


def run_LSTM_speaker_aware(input, constants):
    from make_it_talk import LSTMSpeakerAware
    model = LSTMSpeakerAware()

    check_shape(input, constants, ["batch_size", "frames", "audio_dim"])

    output = model(input)

    check_shape(output, constants, ["batch_size", "hidden_size_2"])

    return output


def run_MLP_speaker_embedding(input, constants):
    from make_it_talk import MLPSpeakerEmbedding
    model = MLPSpeakerEmbedding()

    check_shape(input, constants, ["batch_size", "hidden_size_2"])

    output = model(input)

    check_shape(output, constants, ["batch_size", "hidden_size_3"])

    return output


def run_self_attention_encoder(input, constants):
    from make_it_talk import SelfAttentionEncoder
    model = SelfAttentionEncoder()

    assert len(input) == 2
    check_shape(input, constants, ["batch_size", "hidden_size_2"])
    check_shape(input, constants, ["batch_size", "hidden_size_3"])

    output = model(*input)

    check_shape(output, constants, ["batch_size", "hidden_size_4"])

    return output


def run_facial_landmarks_extractor(input, constants):
    from make_it_talk import FacialLandmarksExtractor
    model = FacialLandmarksExtractor()

    check_shape(input, constants, ["batch_size", "???", "???"])

    output = model(input)

    check_shape(output, constants, ["batch_size", "landmarks_dim"])

    return output


def run_MLP_speaker_aware(input, constants):
    from make_it_talk import MLPSpeakerAware
    model = MLPSpeakerAware()

    assert len(input) == 2
    check_shape(input[0], constants, ["batch_size", "frames", "hidden_size_4"])
    check_shape(input[1], constants, ["batch_size", "landmarks_dim"])

    output = model(input)

    check_shape(output, constants, ["batch_size", "frames", "landmarks_dim"])

    return output


def run_MLP_speech_content(input, constants):
    from make_it_talk import MLPSpeechContent
    model = MLPSpeechContent()

    assert len(input) == 2
    check_shape(input[0], constants, ["batch_size", "frames", "hidden_size_4"])
    check_shape(input[1], constants, ["batch_size", "landmarks_dim"])

    output = model(input)

    check_shape(output, constants, ["batch_size", "frames", "landmarks_dim"])

    return output


def run_landmarks_predictor(input, constants):
    from make_it_talk import LandmarksPredictor
    model = LandmarksPredictor()

    assert len(input) == 3
    check_shape(input[0], constants, ["batch_size", "landmarks_dim"])
    check_shape(input[1], constants, ["batch_size", "frames", "landmarks_dim"])
    check_shape(input[2], constants, ["batch_size", "frames", "landmarks_dim"])

    output = model(input)

    check_shape(output, constants, ["batch_size", "frames", "landmarks_dim"])

    return output


def load_test():
    image = torch.Tensor(io.imread("example_image.jpg"))
    audio = torch.Tensor(librosa.load("example_audio.wav")[1])
    
    batch_size = 4
    image_batch = torch.stack([image] * batch_size)
    audio_batch = torch.stack([audio] * batch_size)
    
    return image_batch, audio_batch


def test_generator():
    constants = dict()

    image_batch, audio_batch = load_test()

    audio_content, audio_speaker = run_audio_to_embedding(audio_batch, constants)
    features_content = run_LSTM_speech_content(audio_content, constants)
    audio_speaker_aware_embedding = run_LSTM_speaker_aware(audio_content, constants)
    features_speaker_aware = run_self_attention_encoder(
        (audio_speaker_aware_embedding, audio_speaker), constants
    )

    lambmarks = run_facial_landmarks_extractor(image_batch, constants)
    landmarks_content_diff = run_MLP_speech_content((features_content, lambmarks), constants)
    landmarks_speaker_aware_diff = run_MLP_speaker_aware((features_speaker_aware, lambmarks), constants)

    landmarks_prediction = run_landmarks_predictor(
        (lambmarks, landmarks_content_diff, landmarks_speaker_aware_diff), constants
    )

