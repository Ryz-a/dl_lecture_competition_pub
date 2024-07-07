import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torchvision import models

'''
class Encoder(nn.Module):
    def __init__(self, in_channels: int, hid_dim: int = 128):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.blocks(X)

class Classifier(nn.Module):
    def __init__(self, num_classes: int, hid_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.head(X)

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, hid_dim)
        self.classifier = Classifier(num_classes, hid_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.encoder(X)
        return self.classifier(X)
'''
class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)

"""  
class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # 最後の全結合層を除外

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # フラット化
        return x
    
class R_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(2048, num_classes)  # ResNet50の特徴量の次元数に合わせる

    def forward(self, x):
        return self.fc(x)
    
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.encoder = ResNetEncoder()
        self.classifier = R_Classifier(num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
"""

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)   

class Pretrain_Classifier(nn.Module):
    def __init__(self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
        ) -> None:
            super().__init__()
            self.encoder_custom = BasicConvClassifier(num_classes,seq_len,in_channels)
            self.encoder_resnet = ResNetClassifier(num_classes)
            # Assuming the output of ResNetEncoder is 2048 and Encoder is encoder_hid_dim * some_factor
            self.classifier = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x, y):
        x_custom = self.encoder_custom(x)
        x_resnet = self.encoder_resnet(y)
        x_combined = torch.cat((x_custom, x_resnet), dim=1)
        x_out = self.classifier(x_combined)
        return x_out
    
class UsePretrainConvClassifier(nn.Module):
    def __init__(
        self,
        encoder: BasicConvClassifier,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.blocks = encoder

        self.head = nn.Linear(num_classes, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)