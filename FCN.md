> write a FCN network with pytorch for image segmentation task. and you need to give a train function with MiOU for judging accuracies.

To create a Fully Convolutional Network (FCN) using PyTorch for an image segmentation task, we need to define the
following components:

1. A **ConvNet Encoder**: This is typically a pre-trained ConvNet like VGG or ResNet which extracts feature maps
   from the input image. We will use the pretrained ResNet50 model as our encoder in this example.
2. An **Atrous Spatial Pyramid Pooling (ASPP) Module**: This module helps capture different spatial context
   information by applying convolutions with different dilation rates. We will create a simplified version of ASPP
   for this task.
3. A **Segmentation Decoder**: The decoder upsamples the feature maps from the encoder and ASPP module to obtain
   segmentation results. In our example, we will use bilinear upsampling followed by a 1x1 convolution layer with
   sigmoid activation to produce the segmentation mask.
4. A **Loss Function**: We will use the Binary Cross Entropy Loss as it is commonly used for binary image
   segmentation problems.
5. An **Optimizer**: In this example, we will use Adam optimizer which is a popular choice for training deep
   learning models.
6. A **Train function** with Mean Intersection-over-Union (MIoU) metric: We will write a train function that
   iterates over the dataset multiple times and calculates MIoU to assess the segmentation performance of our model
   on validation data.

Let's start by defining each component and creating our FCN network in PyTorch.

```python
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, jaccard_score

# Define the ConvNet Encoder - ResNet50 (Pretrained)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)  # Load pretrained ResNet50 model

    def forward(self, x):
        return self.resnet(x)  # Output feature maps from the last layer

# Define Atrous Spatial Pyramid Pooling (ASPP) Module
class ASPP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Conv layer with dilation rate of 1
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)  # Dilation rate
of 6
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)  # Dilation
rate of 12
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)  # Dilation
rate of 18
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling to reduce spatial dimensions

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  # Output from conv layer with dilation rate of 1
        x2 = F.relu(self.conv2(x))  # Output from conv layer with dilation rate of 6
        x3 = F.relu(self.conv3(x))  # Output from conv layer with dilation rate of 12
        x4 = F.relu(self.conv4(x))  # Output from conv layer with dilation rate of 18

        # Global average pooling to reduce spatial dimensions and concatenate the results
        out = torch.cat([self.global_avg_pool(i) for i in [x1, x2, x3, x4]], dim=1)

        return out

# Define Segmentation Decoder
class Decoder(nn.Module):
    def __init__(self, in_channels=768, num_classes=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)  # Convolution layer to reduce
channels from ASPP output
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  # Bilinear upsampling
to increase spatial dimensions
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1)  # Output layer

    def forward(self, x):
        x = F.relu(self.conv(x))  # Apply convolution and ReLU activation
        x = self.upsample(x)  # Upsample to increase spatial dimensions
        out = torch.sigmoid(self.conv2(x))  # Output layer with sigmoid activation

        return out

# Define the Fully Convolutional Network (FCN)
class FCN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = Encoder()
        self.aspp = ASPP()
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)  # Get feature maps from the encoder
        x = self.aspp(x)  # Pass the feature maps through the ASPP module
        out = self.decoder(x)  # Obtain segmentation mask

        return out

# Define Loss Function (Binary Cross Entropy Loss)
class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        return F.binary_cross_entropy(input, target, weight=self.weight)

# Define Optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define Train function with Mean Intersection-over-Union (MIoU) metric
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        # Training loop
        total_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Clear gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            total_loss += loss.item()  # Accumulate loss

            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters

        print(f"Training Loss: {total_loss / len(train_loader)}")

        # Validation loop
        with torch.no_grad():
            model.eval()  # Set the model to evaluation mode
            total_iou = 0.0

            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)  # Forward pass
                pred = torch.round(outputs).detach().cpu()  # Get predicted segmentation mask

                # Calculate intersection-over-union (IoU) for each class and average them
                iou_list = []
                for j in range(num_classes):
                    true_pos = ((pred == j) & (labels == j)).sum().item()
                    false_neg = (((pred != j) | (labels != j)) & (labels == j)).sum().item()

                    # Calculate intersection and union
                    iou = true_pos / (true_pos + false_neg)
                    iou_list.append(iou)

                total_iou += sum(iou_list) / len(val_loader)

            print(f"Validation IoU: {total_iou}")

    return model
```
