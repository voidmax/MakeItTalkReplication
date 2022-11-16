import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython.display import clear_output
from tqdm.auto import tqdm

from make_it_talk.models import *


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
    talking_head_pipeline.personal_landmarks_predictor.requires_grad = False
    talking_head_pipeline.discriminator.requires_grad = False

    talking_head_pipeline.content_landmarks_predictor.requires_grad = True

    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1} started...')
        results = {}
        results['loss'] = []
        results['metrics'] = []
        for i, batch in tqdm(enumerate(dataloader)):

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

    return training_log


def plot_losses(log):
    losses = {'generator_loss': [], 'discriminator_loss': []}

    for epoch in log:
        for name in epoch:
            if name == 'metrics':
                continue
            if epoch[name]:
                losses[name].append(epoch[name])

    # check that losses drop with time

    plt.figure()
    plt.plot(losses['generator_loss'])
    plt.title('generator loss')
    plt.grid()
    plt.show()
    plt.figure()
    plt.plot(losses['discriminator_loss'])
    plt.title('discriminator loss')
    plt.grid()
    plt.show()


def train_pipeline(
        talking_head_pipeline,
        generator_optimizer,
        discriminator_optimizer,
        train_dataloader,
        n_epochs,
        device,
        generator_loss_function,
        discriminator_loss_function,
        metrics_list,
        training_log=None,
):
    if training_log is None:
        training_log = []

    talking_head_pipeline = talking_head_pipeline.to(device)

    talking_head_pipeline.audio_to_embedding.requires_grad = False
    talking_head_pipeline.facial_landmarks_extractor.requires_grad = False
    talking_head_pipeline.content_landmarks_predictor.requires_grad = False

    talking_head_pipeline.discriminator.requires_grad = True
    talking_head_pipeline.personal_landmarks_predictor.requires_grad = True

    talking_head_pipeline.train()

    for epoch in range(n_epochs):

        print(f'Epoch {epoch + 1} started...')

        training_generator_now = epoch % 2

        for i, batch in enumerate(train_dataloader):
            results = {}

            start_landmark = batch['start_landmark'].to(device)
            content = batch['content'].to(device)
            speaker = batch['speaker'].to(device)
            true_landmarks = batch['landmarks'].to(device)

            batch_size, time = true_landmarks.shape[0], true_landmarks.shape[1]
            start_landmark = start_landmark.reshape(batch_size, -1)
            true_landmarks = true_landmarks.reshape(batch_size, time, -1)

            #true_landmarks = talking_head_pipeline.facial_landmarks_extractor(true_videos)

            predicted_landmarks, discriminator_inputs = talking_head_pipeline(content, speaker, start_landmark)

            predicted_landmarks, personal_processed, speaker_processed = discriminator_inputs
            realism_predicted = talking_head_pipeline.discriminator(predicted_landmarks, personal_processed,
                                                                    speaker_processed)
            realism_true = talking_head_pipeline.discriminator(true_landmarks, personal_processed, speaker_processed)

            if training_generator_now:
                loss = generator_loss_function(predicted_landmarks, realism_predicted, true_landmarks)
                results['generator_loss'] = loss.item()
                results['discriminator_loss'] = None
                generator_optimizer.zero_grad()
                loss.backward()
                generator_optimizer.step()
            else:
                predicted_landmarks, personal_processed, speaker_processed = discriminator_inputs
                realism_predicted = talking_head_pipeline.discriminator(predicted_landmarks, personal_processed, speaker_processed)
                realism_true = talking_head_pipeline.discriminator(true_landmarks, personal_processed, speaker_processed)

                loss = discriminator_loss_function(realism_predicted, realism_true)
                results['generator_loss'] = None
                results['discriminator_loss'] = loss.item()
                discriminator_optimizer.zero_grad()
                loss.backward()
                discriminator_optimizer.step()

            metric_results = []
            for metric in metrics_list:
                metric_results.append(metric(predicted_landmarks, true_landmarks))
            results['metrics'] = metric_results

        training_log.append(results)
        clear_output()
        plot_losses(training_log)

    return training_log


