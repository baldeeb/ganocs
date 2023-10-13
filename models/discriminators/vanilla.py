from torch import nn

class Discriminator(nn.Module):
    '''Discriminator model for NOCS images.
    ref: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html'''
    def __init__(self, in_ch=3, feat_ch=64, out_ch=1):
        super().__init__()

        # a 4x4 kernel model. closer to DCGAN.
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, 3, 2, 1, bias=False),  # 28 -> 14
            nn.BatchNorm2d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),

            # NOTE: removed this layer to reduce params when using multiple
            # nn.Conv2d(feat_ch, feat_ch, 3, 1, 1, bias=False), # 14 -> 14
            # nn.BatchNorm2d(feat_ch),
            # nn.LeakyReLU(0.2, inplace=True),
            
            # nn.Conv2d(feat_ch, feat_ch, 3, 1, 1, bias=False), # 14 -> 14
            # nn.BatchNorm2d(feat_ch),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_ch,   2*feat_ch, 3, 2, 1, bias=False), # 14 -> 7
            nn.BatchNorm2d(2*feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(2*feat_ch, 4*feat_ch, 3, 2, 1, bias=False), # 7 -> 4
            nn.BatchNorm2d(4*feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(4*feat_ch, 8*feat_ch, 3, 2, 1, bias=False), # 4 -> 2
            nn.BatchNorm2d(8*feat_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(1),
            nn.Linear(8*feat_ch*4, out_ch),

            # nn.Conv2d(8*feat_ch, 8*feat_ch, 2, 1, 0, bias=False), # 2 -> 1
            # nn.BatchNorm2d(8*feat_ch),
            # nn.LeakyReLU(0.2, inplace=True),
            
            # nn.Conv2d(8*feat_ch, out_ch, 1, 1, 0, bias=False), # 1 -> 1
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
    def forward(self, x):
        return self.model(x)
    
    @property
    def properties(self): return []