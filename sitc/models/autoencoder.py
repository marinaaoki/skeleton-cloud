import torch
import torch.nn as nn

class PairedAE(nn.Module):
    def __init__(self, **kwargs):
        super(PairedAE, self).__init__()

        self.dummy_em = EM()
        self.dummy_pm = PM()

        self.orig_em = EM()
        self.orig_pm = PM()

        self.decoder = Decoder()

    def forward(self, original, dummy):
        dummy_clone = dummy.clone()
        orig_clone = original.clone()

        dummy_motion_embedding = self.dummy_em(dummy)
        dummy_privacy_embedding = self.dummy_pm(dummy_clone)

        orig_motion_embedding = self.orig_em(original)
        orig_privacy_embedding = self.orig_pm(orig_clone)

        x = torch.cat((orig_motion_embedding, dummy_privacy_embedding), 1)
        x = self.decoder(x)

        return x

class AE(nn.Module):
    def __init__(self, **kwargs):
        super(AE, self).__init__()

        self.motion_encoder = EM()
        self.privacy_encoder = PM()

        self.decoder = Decoder()

    def forward(self, x):
        encoder_input = x.clone()
        motion_embedding = self.motion_encoder(x)
        privacy_embedding = self.privacy_encoder(encoder_input)

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


        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=256, kernel_size=3, stride=1)

        self.lr = nn.LeakyReLU()
        self.mp = nn.MaxPool2d(kernel_size=3, stride=1)
        self.rp = nn.ReflectionPad2d(2)
    
    def forward(self, x):

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

        self.lr = nn.LeakyReLU()
        self.up = nn.Upsample(scale_factor=(0.98, 0.95), mode='bilinear')
        self.rp = nn.ReflectionPad2d(1)

        self.convt1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.convt2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.convt3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.convt4 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x):

        x = self.convt1(x)
        x = self.lr(x)
        x = self.up(x)
        x = self.rp(x)

        x = self.convt2(x)
        x = self.lr(x)
        x = self.up(x)
        x = self.rp(x)

        x = self.convt3(x)
        x = self.lr(x)
        x = self.up(x)
        x = self.rp(x)

        x = self.convt4(x)
        x = self.lr(x)
        x = self.up(x)
        x = self.rp(x)

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