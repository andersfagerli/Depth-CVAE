import torch
import torch.nn as nn

from .blocks import ResDownCat, ResUp, ResInCat, ResCat, OutConv


class CVAE(nn.Module):
    def __init__(self, cfg, bilinear=True):
        super(CVAE, self).__init__()
        self.e_ch = [1, 64, 64, 128, 256, 512]
        self.d_ch = [512, 256, 128, 64, 64, 1]
        self.latent_dim = cfg.MODEL.CVAE.LATENT.DIMENSIONS
        self.latent_input_dim = cfg.MODEL.CVAE.LATENT.INPUT_DIM

        # Encoder
        self.down1 = ResInCat(1, 64)
        self.down2 = ResDownCat(64, 64) 
        self.down3 = ResDownCat(64, 128) 
        self.down4 = ResDownCat(128, 256)  
        self.down5 = ResDownCat(256, 512) 

        # Fully connected layers for code mean and log variance
        self.mu = nn.Linear(self.e_ch[-1]*self.latent_input_dim[0]*self.latent_input_dim[1], self.latent_dim)
        self.logvar = nn.Linear(self.e_ch[-1]*self.latent_input_dim[0]*self.latent_input_dim[1], self.latent_dim)

        # Fully connected layer from code to decoder
        self.decoder_in = nn.Linear(self.latent_dim, self.d_ch[0]*self.latent_input_dim[0]*self.latent_input_dim[1])

        # Decoder
        self.d_inc = ResCat(512, 512)
        self.up1 = ResUp(512, 256, bilinear)
        self.up2 = ResUp(256, 128, bilinear)
        self.up3 = ResUp(128, 64, bilinear)
        self.up4 = ResUp(64, 64, bilinear)
        self.out = nn.Sequential(
            OutConv(64, 1),
            nn.Sigmoid()
        )  
        
    def encode(self, feature_maps, x):
        x1, x2, x3, x4, x5 = feature_maps
        
        x = self.down1(x, x5)
        x = self.down2(x, x4)
        x = self.down3(x, x3)
        x = self.down4(x, x2)
        x = self.down5(x, x1)

        return x

    def decode(self, feature_maps, z):
        x1, x2, x3, x4, x5 = feature_maps

        x_flattened = self.decoder_in(z)
        x = x_flattened.view(-1, self.d_ch[0], self.latent_input_dim[0], self.latent_input_dim[1])

        x1_out = self.d_inc(x, x1)
        x2_out = self.up1(x1_out, x2)
        x3_out = self.up2(x2_out, x3)
        x4_out = self.up3(x3_out, x4)
        x5_out = self.up4(x4_out, x5)

        out = self.out(x5_out)

        return out

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar*0.5)
        eps = torch.randn_like(std)

        return mu + std * eps
    
    def sample(self, zero_code=True, batch_size=1):
        if zero_code:
            z = torch.zeros((batch_size, self.latent_dim))
        else:
            z = torch.randn((batch_size, self.latent_dim))
        return z.cuda() if torch.cuda.is_available() else z

    def forward(self, feature_maps, x=None):
        is_training = x is not None
        if is_training:
            x = self.encode(feature_maps, x)

            x_flattened = torch.flatten(x, start_dim=1)
            
            mu = self.mu(x_flattened)
            logvar = self.logvar(x_flattened)
            z = self.reparameterize(mu, logvar)

            out = self.decode(feature_maps, z)
        else:
            batch_size = feature_maps[0].size()[0]

            z = self.sample(zero_code=True, batch_size=batch_size)
            out = self.decode(feature_maps, z)

            mu = torch.zeros((batch_size, self.latent_dim))
            logvar = torch.zeros((batch_size, self.latent_dim))
            
        return out, mu, logvar