import torch
import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        # Convolutional layers
        
        self.conv1 = nn.Conv2d(3, 16, (3,3), padding=1) # 224x224x3->224x224x28
        self.conv2 = nn.Conv2d(16, 2*16, (3,3), padding=1) # 112x112x28->112x112x56
        self.conv3 = nn.Conv2d(16*2, 4*16, (3,3), padding=1)# 56x56x56->56x56x(28*4)
        self.conv4 = nn.Conv2d(16*4, 8*16, (3,3), padding=1)# 28x28x56->28x28x(28*4)
        self.maxpool = nn.MaxPool2d((2,2))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        
        # Fully connected layers
        
        self.fc1 = nn.Linear(14*14*(8*16), 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Activation functions and batch normalization
        
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.BatchNorm2D_1 = nn.BatchNorm2d(16)
        self.BatchNorm2D_2 = nn.BatchNorm2d(2*16)
        self.BatchNorm2D_3 = nn.BatchNorm2d(4*16)
        self.BatchNorm1D = nn.BatchNorm1d(256)     

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        
        # Feature extraction
        
        x = self.maxpool(self.LeakyReLU(self.conv1(x)))
        x = self.BatchNorm2D_1(x)
        x = self.maxpool(self.LeakyReLU(self.conv2(x)))
        x = self.BatchNorm2D_2(x)
        x = self.maxpool(self.LeakyReLU(self.conv3(x)))
        x = self.BatchNorm2D_3(x)
        x = self.maxpool(self.LeakyReLU(self.conv4(x)))
        x = x.view(-1, (8*16) * 14 * 14)     
        
        # linear layers
        
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.LeakyReLU(self.fc1(x))
        x = self.BatchNorm1D(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
