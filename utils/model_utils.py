import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

# Define the model architecture (ResNeXt50 + LSTM) as you used during training
class DeepFakeDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepFakeDetectionModel, self).__init__()
        self.base_model = models.resnext50_32x4d(pretrained=False)
        self.base_model.fc = nn.Identity()  # Remove the final FC layer
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # Check if input has 4 or 5 dimensions
        if len(x.size()) == 4:
            # Input shape is (batch_size, c, h, w) - Single frame input
            x = x.unsqueeze(1)  # Add a sequence length dimension: (batch_size, 1, c, h, w)

        batch_size, seq_len, c, h, w = x.size()
        
        x = x.view(batch_size * seq_len, c, h, w)  # Reshape for ResNeXt input
        features = self.base_model(x)
        features = features.view(batch_size, seq_len, -1)  # (Batch, Sequence, Features)
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]  # Take only the output from the last LSTM cell
        out = self.classifier(lstm_out)
        return out

def load_model(model_path):
    """Load the pre-trained model and weights."""
    # Initialize the model with the same architecture used during training
    model = DeepFakeDetectionModel(num_classes=2)
    
    # Load the saved weights from the checkpoint file into the model
    checkpoint = torch.load(model_path, map_location=torch.device('mps'))  # Use MPS for Apple Silicon
    model.load_state_dict(checkpoint)  # Load weights into the model
    
    model.to('mps')  # Move model to MPS device
    model.eval()  # Set the model to evaluation mode
    return model

def predict(model, frames):
    """
    Run the model inference on extracted frames.
    Frames should be a 4D tensor: (batch_size, channels, height, width).
    """
    with torch.no_grad():
        # Ensure input is a float tensor and move it to the MPS device
        inputs = torch.tensor(frames).float().to('mps')
        outputs = model(inputs)  # Forward pass through the model
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Get class probabilities
        predictions = torch.argmax(probabilities, dim=1)  # Get predicted class (0 or 1)

        # Use majority voting: count the number of predictions for each class (0 or 1)
        # majority_vote = predictions.sum().item()  # Sum gives the number of 1s (fakes)
        mean_prob = probabilities[:, 1].mean().item()  # Average probability of 'Fake'
        return 1 if mean_prob > 0.5 else 0  # Fake if probability > 50%
        # # If majority of frames are fake (1), return 1 (fake), otherwise 0 (real)
        # return 1 if majority_vote > len(predictions) // 2 else 0
