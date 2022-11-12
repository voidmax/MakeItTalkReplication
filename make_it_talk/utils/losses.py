import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import landmark_classes

class LossForContentPredictedLandmarks(nn.Module):
    def __init__(self, lambda_classes = 1):
        super(LossForContentPredictedLandmarks, self).__init__()
        self.lambda_classes = lambda_classes

    def forward(self, predicted_landmarks, true_landmarks):
        '''
        Computes MSE + lambda_c * MSE_inside_class
        EXPECTS BATCH __AND__ TIME IN INPUTS
        TODO: change this to handle more general case later
        '''

        batch_size = predicted_landmarks.shape[0]
        time = predicted_landmarks.shape[1]
        if len(predicted_landmarks.shape) == 3:
            predicted_landmarks = predicted_landmarks.reshape(batch_size, time, 68, 3)
        if len(true_landmarks.shape) == 3:
            true_landmarks = true_landmarks.reshape(batch_size, time, 68, 3)

        mse_total = F.mse_loss(predicted_landmarks, true_landmarks)

        mse_classes = []
        n_classes = len(landmark_classes)

        for class_idx in landmark_classes:
            class_len = len(class_idx)
            class_mask = torch.zeros_like(predicted_landmarks)
            class_mask.requires_grad = False
            class_mask[:, :, class_idx, :] += 1.0

            preds = predicted_landmarks * class_mask
            true = true_landmarks * class_mask

            preds -= preds.sum(dim=2, keepdim=True) / class_len
            true -= preds.sum(dim=2, keepdim=True) / class_len

            mse_classes.append(F.mse_loss(preds, true, reduction='sum') / class_mask.sum().item())

        mse_classes_mean = torch.mean(torch.stack(mse_classes, dim=0))

        return mse_total + self.lambda_classes * mse_classes_mean

class LossForDiscriminator(nn.Module):
    def __init__(self):
        super(LossForDiscriminator, self).__init__()

    def forward(self, realism_pred, realism_true):
        return ((realism_true - 1) ** 2).mean() + (realism_pred ** 2).mean()

class LossForGenerator(nn.Module):
    def __init__(self, lambda_classes, mu_discriminator):
        super(LossForGenerator, self).__init__()
        self.lambda_classes = lambda_classes
        self.mu_discriminator = mu_discriminator

    def forward(self, predicted_landmarks, predicted_realism, true_landmarks):
        realism_loss = ((predicted_realism - 1) ** 2).mean()

        batch_size = predicted_landmarks.shape[0]
        time = predicted_landmarks.shape[1]
        if len(predicted_landmarks.shape) == 3:
            predicted_landmarks = predicted_landmarks.reshape(batch_size, time, 68, 3)
        if len(true_landmarks.shape) == 3:
            true_landmarks = true_landmarks.reshape(batch_size, time, 68, 3)

        mse_total = mse_total = F.mse_loss(predicted_landmarks, true_landmarks)

        mse_classes = []
        n_classes = len(landmark_classes)

        for class_idx in landmark_classes:
            class_len = len(class_idx)
            class_mask = torch.zeros_like(predicted_landmarks)
            class_mask.requires_grad = False
            class_mask[:, :, class_idx, :] += 1.0

            preds = predicted_landmarks * class_mask
            true = true_landmarks * class_mask

            preds -= preds.sum(dim=2, keepdim=True) / class_len
            true -= preds.sum(dim=2, keepdim=True) / class_len

            mse_classes.append(F.mse_loss(preds, true, reduction='sum') / class_mask.sum().item())

        mse_classes_mean = torch.mean(torch.stack(mse_classes, dim=0))

        return mse_total + self.lambda_classes * mse_classes_mean + self.mu_discriminator * realism_loss