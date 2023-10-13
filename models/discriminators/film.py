from torch import nn

class FiLMLayer(nn.Module):
    '''A convolutional layer with feature modulation.'''
    def __init__(self, in_ch, out_ch, 
                 kernel_size=3, 
                 stride=1, 
                 padding=0,
                 dilation=1,
                 dim_ctx=64,
                 activation=nn.ReLU(inplace=True),
                 batch_norm=nn.BatchNorm2d):
        super().__init__()
        self.layer = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation)]
        if batch_norm is not None: self.layer.append(batch_norm(out_ch))
        if activation is not None: self.layer.append(activation)
        
        self.layer = nn.Sequential(*self.layer)
        self.bias = nn.Linear(dim_ctx, out_ch, bias=False)
        self.gate = nn.Linear(dim_ctx, out_ch)

    def forward(self, x, ctx):
        '''x: input feature map
        ctx: context vector'''
        x = self.layer(x)

        # FiLM
        gamma = self.gate(ctx).unsqueeze(-1).unsqueeze(-1)
        beta = self.bias(ctx).unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        return x