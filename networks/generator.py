import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, opt, num_channels):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 128, 3, stride=1, padding=1)),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 64, 3, stride=1, padding=1)),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, num_channels[opt.dataset], 3, stride=1, padding=1)),
            nn.Tanh(),
            nn.BatchNorm2d(num_channels[opt.dataset], affine=False),
        )

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, a=-0.08, b=0.08)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.08, b=0.08)
                nn.init.zeros_(m.bias)
        
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        img = self.conv_blocks3(img)
        return img

class MemoryGenerator(nn.Module):
    def __init__(self, opt, num_channels):
        super(MemoryGenerator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks3 = nn.Sequential(
            nn.Conv2d(64, num_channels[opt.dataset], 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(num_channels[opt.dataset], affine=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, a=-0.08, b=0.08)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.08, b=0.08)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        img = self.conv_blocks3(img)
        return img

class Encoder(nn.Module):
    def __init__(self, opt, num_channels):
        super(Encoder, self).__init__()
        self.init_size = opt.img_size // 4
        self.conv_blocks0 = nn.Sequential(
            nn.Conv2d(num_channels[opt.dataset], 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
        )
        self.fc_mu = nn.Linear(32*self.init_size**2, opt.latent_dim)
        self.fc_var = nn.Linear(32*self.init_size**2, opt.latent_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, a=-0.08, b=0.08)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.08, b=0.08)
                nn.init.zeros_(m.bias)

    def forward(self, img):
        img = F.normalize(img, p=2, dim=2)
        out = self.conv_blocks0(img)
        out = self.conv_blocks1(out)
        out = nn.functional.interpolate(out,scale_factor=0.5)
        out = self.conv_blocks2(out)
        out = nn.functional.interpolate(out,scale_factor=0.5)
        out = out.view(out.shape[0], 32*self.init_size**2)
        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return (eps * std + mu), mu, log_var

