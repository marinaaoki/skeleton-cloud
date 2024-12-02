import torch
import torch.nn as nn

class AE(nn.Module):

    def __init__(self, **kwargs):
        super(AE, self).__init__()

        self.in_features = kwargs['in_features']
        self.out_features = kwargs['out_features']

        self.motion_encoder = EM(in_features=self.in_features, out_features=self.out_features)
        self.privacy_encoder = PM(in_features=self.in_features, out_features=self.out_features)

        self.decoder = Decoder(in_features=self.out_features*2, out_features=self.in_features)

    def forward(self, x):
        motion_embedding = self.motion_encoder(x)
        privacy_embedding = self.privacy_encoder(x)

        x = torch.cat((motion_embedding, privacy_embedding), 1)
        x = self.decoder(x)

        return x


class Encoder(nn.Module):
    """
    The encoder network. 
    Both the motion encoder and the privacy encoder have the same architecture and inherit from this class.
    """

    def __init__(self, **kwargs):
        super(Encoder, self).__init__()

        self.in_features = kwargs['in_features']
        self.out_features = kwargs['out_features']

        self.conv1 = nn.Conv2d(in_channels=self.in_features, out_channels=12, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=256, kernel_size=3, stride=1)

        self.lr = nn.LeakyReLU()
        self.mp = nn.MaxPool2d(kernel_size=3, stride=1)
        self.rp = nn.ReflectionPad2d(1)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x = self.rp(self.mp(self.lr(self.conv1(x))))
        x = self.rp(self.mp(self.lr(self.conv2(x))))
        x = self.rp(self.mp(self.lr(self.conv3(x))))
        x = self.rp(self.mp(self.lr(self.conv4(x))))

        return x
    

class Decoder(nn.Module):
    """
    The decoder network.
    This takes the concatenated motion embedding and privacy embedding as inputs and reconstructs the skeleton motion.
    """

    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

        self.in_features = kwargs['in_features']
        self.out_features = kwargs['out_features']

        self.convt1 = nn.ConvTranspose2d(in_channels=self.in_features, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.convt2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.convt3 = nn.ConvTranspose2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.convt4 = nn.ConvTranspose2d(in_channels=96, out_channels=self.out_features, kernel_size=3, stride=1, padding=1)

        self.lr = nn.LeakyReLU()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.rp = nn.ReflectionPad2d(1)

    def forward(self, x):
        x = self.rp(self.up(self.lr(self.convt1(x))))
        x = self.rp(self.up(self.lr(self.convt2(x))))
        x = self.rp(self.up(self.lr(self.convt3(x))))
        x = self.rp(self.up(self.lr(self.convt4(x))))

        return x


class EM(Encoder):
    """
    The motion encoder network.
    Defined as a subclass in case we want to add more functionality in the future.
    """
    
    def __init__(self, **kwargs):
        super(EM, self).__init__(**kwargs)

class PM(Encoder):
    """
    The privacy encoder network.
    Defined as a subclass in case we want to add more functionality in the future.
    """
    
    def __init__(self, **kwargs):
        super(PM, self).__init__(**kwargs)
