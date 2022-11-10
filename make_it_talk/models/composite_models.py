import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLandmarkDeltasPredictor(nn.Module):
    def __init__(
            self,
            lstm_speech_content,
            mlp_speech_content,
    ):
        super(ContentLandmarkDeltasPredictor, self).__init__()
        self.lstm_speech_content = lstm_speech_content
        self.mlp_speech_content = mlp_speech_content

    def forward(self, audio_embeddings, landmarks):
        audio_embeddings = audio_embeddings[0]  # drop speaker embeddings
        processed_audios = self.lstm_speech_content(audio_embeddings)
        predicted_deltas = self.mlp_speech_content(processed_audios, landmarks)
        return predicted_deltas


class SpeakerAwareLandmarkDeltasPredictor(nn.Module):
    def __init__(
            self,
            lstm_speaker_aware,
            mlp_speaker_embedding,
            self_attention_encoder,
            mlp_speaker_aware,
            landmarks_predictor,
    ):
        self.lstm_speaker_aware = lstm_speaker_aware
        self.mlp_speaker_embedding = mlp_speaker_embedding
        self.self_attention_encoder = self_attention_encoder
        self.mlp_speaker_aware = mlp_speaker_aware
        self.landmarks_predictor = landmarks_predictor

    def forward(
            self,
            audio_embeddings,
            landmarks,
            return_discriminator_inputs=True,
    ):

        audio_content, speaker = audio_embeddings

        batch_size, n_tokens, audio_emb_dim = audio_content.shape
        speaker_dim = speaker.shape[-1]
        landmarks_dim = landmarks.shape[-1]

        personal_processed = self.lstm_speaker_aware(audio_content)
        speaker_processed = self.mlp_speaker_embedding(speaker)

        personal_and_speaker_processed = self.self_attention_encoder(speaker_processed, personal_processed)

        personal_landmark_deltas = self.mlp_speaker_aware(personal_and_speaker_processed, landmarks)

        if return_discriminator_inputs:
            return personal_landmark_deltas, (personal_processed, speaker_processed)
        else:
            return personal_landmark_deltas


class TalkingHeadPipeline(nn.Module):
    def __init__(
            self,
            audio_to_embedding: nn.Module,
            facial_landmarks_extractor: nn.Module,
            content_landmarks_predictor: ContentLandmarkDeltasPredictor,
            personal_landmarks_predictor: SpeakerAwareLandmarkDeltasPredictor,
            discriminator: nn.Module,
    ):
        super(TalkingHeadPipeline, self).__init__()
        self.audio_to_embedding = audio_to_embedding,
        self.facial_landmarks_extractor = facial_landmarks_extractor
        self.content_landmarks_predictor = content_landmarks_predictor
        self.personal_landmarks_predictor = personal_landmarks_predictor
        self.discriminator = discriminator

    def forward(
            self,
            audios,
            pictures,
            return_discriminator_inputs=True,
    ):
        audio_embeddings = self.audio_to_embedding(audios)
        landmarks = self.facial_landmarks_extractor(pictures)

        content_deltas = self.content_landmarks_predictor(audio_embeddings, landmarks)

        if return_discriminator_inputs:
            personal_deltas, additional_discriminator_inputs = self.personal_landmarks_predictor(audio_embeddings,
                                                                                                 landmarks,
                                                                                                 return_discriminator_inputs)
        else:
            personal_deltas = self.personal_landmarks_predictor(audio_embeddings,
                                                                landmarks,
                                                                return_discriminator_inputs)

        predicted_landmarks = landmarks.reshape(landmarks.shape[0], 1,
                                                landmarks.shape[1]) + content_deltas + personal_deltas

        if return_discriminator_inputs:
            return predicted_landmarks, (predicted_landmarks, *additional_discriminator_inputs)
        else:
            return predicted_landmarks


def make_talking_head_pipeline(
        audio_to_embedding,
        lstm_speech_content,
        lstm_speaker_aware,
        mlp_speaker_embedding,
        self_attention_encoder,
        facial_landmarks_extractor,
        mlp_speaker_aware,
        mlp_speech_content,
        landmarks_predictor,
        discriminator,
):
    content_landmarks_predictor = ContentLandmarkDeltasPredictor(
        lstm_speech_content,
        mlp_speech_content,
    )

    generator = SpeakerAwareLandmarkDeltasPredictor(
        lstm_speaker_aware=lstm_speaker_aware,
        mlp_speaker_embedding=mlp_speaker_embedding,
        self_attention_encoder=self_attention_encoder,
        mlp_speaker_aware=mlp_speaker_aware,
        landmarks_predictor=landmarks_predictor,
    )

    return TalkingHeadPipeline(
        audio_to_embedding,
        facial_landmarks_extractor,
        content_landmarks_predictor,
        generator,
        discriminator,
    )
