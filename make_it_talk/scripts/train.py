import torch
import torch.nn as nn
import torch.nn.functional as F

from MakeItScream.MakeItTalkReplication.make_it_talk import TalkingHeadPipeline


def train_content_landmarks_predictor(
        talking_head_pipeline: TalkingHeadPipeline,
        optimizer,
        dataloader,
        n_epochs,
        device,
        loss_function,
        metrics_list,
        training_log=None,
):
    if training_log is None:
        training_log = []

    talking_head_pipeline = talking_head_pipeline.to(device)

    talking_head_pipeline.audio_to_embedding.requires_grad = False
    talking_head_pipeline.facial_landmarks_extractor.requires_grad = False
    talking_head_pipeline.generator.requires_grad = False
    talking_head_pipeline.discriminator.requires_grad = False

    talking_head_pipeline.content_landmarks_predictor.requires_grad = True

    for epoch in range(n_epochs):
        results = {}
        results['loss'] = []
        results['metrics'] = []
        for i, batch in enumerate(dataloader):
            initial_pictures, audios, true_videos = batch

            initial_pictures = initial_pictures.to(device)
            audios = audios.to(device)
            true_videos = true_videos.to(device)

            initial_landmarks = talking_head_pipeline.facial_landmarks_extractor(initial_pictures)
            true_landmarks = talking_head_pipeline.facial_landmarks_extractor(true_videos)
            audio_embeddings = talking_head_pipeline.audio_to_embedding(audios)

            predicted_deltas = talking_head_pipeline.content_landmarks_predictor(audio_embeddings, initial_landmarks)
            batch_size, landmarks_dim = initial_landmarks.shape
            predicted_landmarks = predicted_deltas + initial_landmarks.reshape(batch_size, 1, landmarks_dim)

            # TODO: discuss this
            loss = loss_function(predicted_landmarks, true_landmarks)
            results['loss'] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_results = []
            for metric in metrics_list:
                metric_results.append(metric(predicted_landmarks, true_landmarks))
            results['metrics'] = metric_results

        training_log.append(results)


def train_pipeline(
        talking_head_pipeline,
        generator_optimizer,
        discriminator_optimizer,
        train_dataloader,
        device,
        generator_loss_function,
        discriminator_loss_function,
        metrics_list,
        training_log=None,
):
    if training_log is None:
        training_log = []

    talking_head_pipeline = talking_head_pipeline.to(device)
    talking_head_pipeline.train()

    for i, batch in enumerate(train_dataloader):

        training_generator_now = i % 2

        # TODO: find out the format of the batch and change accordingly
        # whatever is returned should already be torch.Tensors to put to device
        audios, pictures, true_videos = batch

        audios = audios.to(device)
        pictures = pictures.to(device)
        true_videos = true_videos.to(device)

        predicted_landmarks, discriminator_input = talking_head_pipeline.generator(audios, pictures)

        if training_generator_now:
            loss = generator_loss_function
        predicted_landmarks, personal_processed, speaker_processed = discriminator_input

        true_landmarks = talking_head_pipeline.facial_landmarks_extractor(true_videos)
