def metric_DL(predicted_landmarks, true_landmarks):
    mse = ((predicted_landmarks - true_landmarks)**2).mean() * 3
    width = ((true_landmarks[:, 0] - true_landmarks[:, 1])**2).sum(axis=-1)
    return (mse / width).sqrt()

def metric_DV(predicted_landmarks, true_landmarks):
    predicted_velocity = predicted_landmarks[1:] - predicted_landmarks[:-1] 
    true_velocity = true_landmarks[1:] - true_landmarks[:-1] 
    mse = ((predicted_velocity - true_velocity)**2).mean() * 3
    width = ((true_landmarks[:, 0] - true_landmarks[:, 1])**2).sum(axis=-1)
    return (mse / width).sqrt()