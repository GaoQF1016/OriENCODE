from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, depth, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        # 使用 nn.TransformerEncoder 和 nn.TransformerEncoderLayer
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layers, depth)
        
        # 添加层归一化
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.transformer(x)
        return self.norm(x)
    
class x_Conv(nn.Module):
    def __init__(self, nf):
        super(x_Conv, self).__init__()
        
        # 创建卷积层
        self.conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=False)
        
        # 创建权重掩码
        self.mask = torch.tensor([[[[1, 0, 1],
                                    [1, 1, 1],
                                    [1, 0, 1]]]], dtype=torch.float32)
        
        # 扩展掩码，以适应输入和输出通道
        self.mask = self.mask.repeat(nf, nf, 1, 1)
        # 将卷积权重随机初始化，但应用掩码
        nn.init.xavier_uniform_(self.conv.weight)
        self.conv.weight.data *= self.mask

    def forward(self, x):
        # 应用掩码在每次前向传播之前
        self.conv.weight.data *= self.mask.to(self.conv.weight.device)
        return self.conv(x)
    
class y_Conv(nn.Module):
    def __init__(self, nf):
        super(y_Conv, self).__init__()
        
        # 创建卷积层
        self.conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=False)
        
        # 创建权重掩码
        self.mask = torch.tensor([[[[1, 1, 1],
                                    [0, 1, 0],
                                    [1, 1, 1]]]], dtype=torch.float32)
        
        # 扩展掩码，以适应输入和输出通道
        self.mask = self.mask.repeat(nf, nf, 1, 1)
        
        # 将卷积权重随机初始化，但应用掩码
        nn.init.xavier_uniform_(self.conv.weight)
        self.conv.weight.data *= self.mask

    def forward(self, x):
        # 应用掩码在每次前向传播之前
        self.conv.weight.data *= self.mask.to(self.conv.weight.device)
        return self.conv(x)
    
# Attention Branch
class AttentionBranch(nn.Module):

    def __init__(self, nf, k_size=3):

        super(AttentionBranch, self).__init__()
        self.k1 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        '''self.x_conv = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.y_conv = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.x_conv.weight = nn.Parameter(torch.tensor([[[[-1,0,1],
                                                          [-2,0,2],
                                                          [-1,0,1]]]], dtype=torch.float).repeat([nf,nf,1,1]), requires_grad=False)
        self.y_conv.weight = nn.Parameter(torch.tensor([[[[1,2,1],
                                                          [0,0,0],
                                                          [-1,-2,-1]]]], dtype=torch.float).repeat([nf,nf,1,1]), requires_grad=False)'''
        self.x_conv = x_Conv(nf)
        self.y_conv = y_Conv(nf)
        #self.channel_expand = nn.Conv2d(nf, nf // 2, kernel_size=1, padding=(k_size - 1) // 2, bias=False)
  
    def forward(self, x):
        
        y = self.k1(x)
        y = self.lrelu(y)
        x_branch = self.x_conv(y)
        y_branch = self.y_conv(y)
        y = self.sigmoid(x_branch + y_branch)
        


        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out
    

class AAB(nn.Module):

    def __init__(self, nf):
        super(AAB, self).__init__()

        self.conv_first = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.conv_last = nn.Conv2d(nf, nf, kernel_size=1, bias=False)        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)    
        # attention branch
        self.attention = AttentionBranch(nf)  
        
        # non-attention branch
        # 3x3 conv for A2N
        self.non_attention = nn.Conv2d(nf, nf, kernel_size=3, padding=(3 - 1) // 2, bias=False)         
        # 1x1 conv for A2N-M (Recommended, fewer parameters)
        # self.non_attention = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.weight1 = nn.Parameter(torch.tensor(0.5))
        self.weight2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        residual = x
        a, b, c, d = x.shape

        x = self.conv_first(x)
        x = self.lrelu(x)

        attention = self.attention(x)
        non_attention = self.non_attention(x)
    
        x = attention * self.weight1 + non_attention * self.weight2
        x = self.lrelu(x)

        out = self.conv_last(x)
        #out += residual

        return out
    
def get_activation(activation):
    """ Get activation from str or nn.Module
    """
    if activation is None:
        return None
    elif isinstance(activation, str):
        activation = getattr(nn, activation)()
    else:
        activation = activation()
        assert isinstance(activation, nn.Module)
    return activation


class Upsample(nn.Module):
    """ Upsample the input and change the number of channels
    via 1x1 Convolution if a different number of input/output channels is specified.
    """

    def __init__(self, scale_factor, mode='nearest',
                 in_channels=None, out_channels=None, align_corners=False,
                 ndim=3):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        if in_channels != out_channels:
            if ndim == 2:
                self.conv = nn.Conv2d(in_channels, out_channels, 1)
            elif ndim == 3:
                self.conv = nn.Conv3d(in_channels, out_channels, 1)
            else:
                raise ValueError("Only 2d and 3d supported")
        else:
            self.conv = None

    def forward(self, input):
        x = F.interpolate(input, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        if self.conv is not None:
            return self.conv(x)
        else:
            return x
        
class UNetBase(nn.Module):
    """ UNet Base class implementation

    Deriving classes must implement
    - _conv_block(in_channels, out_channels, level, part)
        return conv block for a U-Net level
    - _pooler(level)
        return pooling operation used for downsampling in-between encoders
    - _upsampler(in_channels, out_channels, level)
        return upsampling operation used for upsampling in-between decoders
    - _out_conv(in_channels, out_channels)
        return output conv layer

    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      depth: depth of the network
      initial_features: number of features after first convolution
      gain: growth factor of features
      pad_convs: whether to use padded convolutions
      norm: whether to use batch-norm, group-norm or None
      p_dropout: dropout probability
      final_activation: activation applied to the network output
    """
    norms = ('BatchNorm', 'GroupNorm')
    pool_modules = ('MaxPool', 'StrideConv')

    def __init__(self, in_channels, out_channels, depth=4, initial_features=64, gain=2, pad_convs=False, norm=None,
                 norm_groups=None, p_dropout=None, final_activation=None, activation=nn.ReLU(), pool_mode='MaxPool',
                 skip_gn_level=None, upsample_mode='bilinear'):
        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad_convs = pad_convs
        if norm is not None:
            assert norm in self.norms
        assert pool_mode in self.pool_modules
        self.pool_mode = pool_mode
        self.norm = norm
        self.norm_groups = norm_groups
        if p_dropout is not None:
            assert isinstance(p_dropout, (float, dict))
        self.p_dropout = p_dropout
        self.skip_gn_level=skip_gn_level

        # modules of the encoder path
        n_features = [in_channels] + [initial_features * gain ** level
                                      for level in range(self.depth)]
        self.features_per_level = n_features
        self.encoder = nn.ModuleList([self._conv_block(n_features[level], n_features[level + 1],
                                                       level, part='encoder', activation=activation)
                                      for level in range(self.depth)])
        self.aab_modules = nn.ModuleList([
                  AAB(n_features[level + 1]) for level in range(self.depth)
              ])
        # the base convolution block
        self.base = self._conv_block(n_features[-1], gain * n_features[-1],
                                     part='base', level=depth, activation=activation)

        # modules of the decoder path
        n_features = [initial_features * gain ** level
                      for level in range(self.depth + 1)]
        n_features = n_features[::-1]
        self.decoder = nn.ModuleList([self._conv_block(n_features[level], n_features[level + 1],
                                                       self.depth - level - 1, part='decoder', activation=activation)
                                      for level in range(self.depth)])

        # the pooling layers;
        self.poolers = nn.ModuleList([self._pooler(level + 1) for level in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList([self._upsampler(n_features[level],
                                                         n_features[level + 1],
                                                         self.depth - level - 1,
                                                         mode=upsample_mode)
                                         for level in range(self.depth)])
        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        
        self.out_conv = self._out_conv(n_features[-1], out_channels)
        self.activation = get_activation(final_activation)


    @staticmethod
    def _crop_tensor(input_, shape_to_crop):
        input_shape = input_.shape
        # get the difference between the shapes
        shape_diff = tuple((ish - csh) // 2
                           for ish, csh in zip(input_shape, shape_to_crop))
        # if input_.size() == shape_to_crop:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if all(sd == 0 for sd in shape_diff):
                return input_
        # calculate the crop
        crop = tuple(slice(sd, sh - sd)
                     for sd, sh in zip(shape_diff, input_shape))
        return input_[crop]

    # crop the `from_encoder` tensor and concatenate both
    def _crop_and_concat(self, from_decoder, from_encoder):
        cropped = self._crop_tensor(from_encoder, from_decoder.shape)
        return torch.cat((cropped, from_decoder), dim=1)

    def forward_parts(self, parts):
        if 'encoder' in parts:
            x = input
            # apply encoder path
            encoder_out = []
            for level in range(self.depth):
                x = self.encoder[level](x)
                encoder_out.append(x)
                x = self.poolers[level](x)

        if 'base' in parts:
            x = self.base(x)

        if 'decoder' in parts:
            # apply decoder path
            encoder_out = encoder_out[::-1]
            for level in range(self.depth):
                x = self.upsamplers[level](x)
                x = self.decoder[level](self._crop_and_concat(x,
                                                              encoder_out[level]))

            # apply output conv and activation (if given)
            x = self.out_conv(x)
            if self.activation is not None:
                x = self.activation(x)

        return x

    def forward(self, input):
        x = input
        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            x = self.aab_modules[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)

        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.decoder[level](self._crop_and_concat(x,
                                                          encoder_out[level]))

        # apply output conv and activation (if given)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class UNet2d(UNetBase):
    """ 2d U-Net for segmentation as described in
    https://arxiv.org/abs/1505.04597
    """

    # Convolutional block for single layer of the decoder / encoder
    # we apply to 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels, level, part, activation=nn.ReLU()):
        """
        Returns a 'double' conv block as described in the paper.
        Group Norm can be skipped until specified level
        :param in_channels:
        :param out_channels:
        :param level:
        :param part:
        :param activation:
        :return:
        """
        padding = 1 if self.pad_convs else 0
        if self.norm is not None:
            num_groups1 = min(in_channels, self.norm_groups)
            num_groups2 = min(out_channels, self.norm_groups)
        else:
            num_groups1 = None
            num_groups2 = None
        if self.norm is None or (self.skip_gn_level is not None and self.skip_gn_level >= level):
            sequence = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                               kernel_size=3, padding=padding),
                                     activation,
                                     nn.Conv2d(out_channels, out_channels,
                                               kernel_size=3, padding=padding),
                                     activation)
        elif self.norm == 'GroupNorm':
            sequence = nn.Sequential(nn.GroupNorm(num_groups1, in_channels),
                                     nn.Conv2d(in_channels, out_channels,
                                               kernel_size=3, padding=padding),
                                     activation,
                                     nn.GroupNorm(num_groups2, out_channels),
                                     nn.Conv2d(out_channels, out_channels,
                                               kernel_size=3, padding=padding),
                                     activation)

        if self.p_dropout is not None:
            sequence.add_module('droupout', nn.Dropout2d(p=self.p_dropout))

        return sequence

    # upsampling via transposed 2d convolutions
    def _upsampler(self, in_channels, out_channels, level, mode):
        # use bilinear upsampling + 1x1 convolutions
        return Upsample(in_channels=in_channels,
                        out_channels=out_channels,
                        scale_factor=2, mode=mode, ndim=2,
                        align_corners=False if mode == 'bilinear' else None)

    # pooling via maxpool2d
    def _pooler(self, level):
        if self.pool_mode == 'MaxPool':
            return nn.MaxPool2d(2)
        elif self.pool_mode == 'StrideConv':
            return nn.Conv2d(self.features_per_level[level], self.features_per_level[level],
                             kernel_size=2, stride=2, padding=0)

    def _out_conv(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, 1)
    
class MLTHeads(nn.Module):
    def __init__(self, in_channels, out_channels, last_kernel, norm, norm_groups, padding, activation):
        super().__init__()
        self.norm = norm
        self.norm_groups = norm_groups
        if self.norm is not None:
            groups_1 = min(in_channels, self.norm_groups)
            groups_2 = min(1, self.norm_groups)
        else:
            groups_1 = None
            groups_2 = None

        padding = padding

        self.core = self._make_core(in_channels, groups_1, groups_2, activation, padding, self.norm)
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=last_kernel, padding=False)

    def forward(self, x):
        o = self.core.forward(x)
        o = self.out_conv.forward(o)

        return o

    @staticmethod
    def _make_core(in_channels, groups_1, groups_2, activation, padding, norm):
        if norm == 'GroupNorm':
            return nn.Sequential(nn.GroupNorm(groups_1, in_channels),
                                 nn.Conv2d(in_channels, in_channels,
                                           kernel_size=3, padding=padding),
                                 activation,
                                 # nn.GroupNorm(groups_2, in_channels)
                                 )
        elif norm is None:
            return nn.Sequential(nn.Conv2d(in_channels, in_channels,
                                           kernel_size=3, padding=padding),
                                 activation)
        else:
            raise NotImplementedError
        
class DoubleMUnet(nn.Module):
    def __init__(self, ch_in, ch_out, ext_features=0, depth_shared=3, depth_union=3, initial_features=64,
                 inter_features=64,
                 activation=nn.ReLU(), use_last_nl=True, norm=None, norm_groups=None, norm_head=None,
                 norm_head_groups=None, pool_mode='Conv2d', upsample_mode='bilinear', skip_gn_level=None):
        super().__init__()

        self.unet_shared = UNet2d(1 + ext_features, inter_features, depth=depth_shared, pad_convs=True,
                                             initial_features=initial_features,
                                             activation=activation, norm=norm, norm_groups=norm_groups,
                                             pool_mode=pool_mode, upsample_mode=upsample_mode,
                                             skip_gn_level=skip_gn_level)

        self.unet_union = UNet2d(ch_in * inter_features, inter_features, depth=depth_union, pad_convs=True,
                                            initial_features=initial_features,
                                            activation=activation, norm=norm, norm_groups=norm_groups,
                                            pool_mode=pool_mode, upsample_mode=upsample_mode,
                                            skip_gn_level=skip_gn_level)

        assert ch_in in (1, 3)
        # assert ch_out in (5, 6)
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.mt_heads = nn.ModuleList(
            [MLTHeads(inter_features, out_channels=1, last_kernel=1,
                      norm=norm_head, norm_groups=norm_head_groups,
                      padding=True, activation=activation) for _ in range(self.ch_out)])

        self._use_last_nl = use_last_nl

        self.p_nl = torch.sigmoid  # only in inference, during training
        self.phot_nl = torch.sigmoid
        self.xyz_nl = torch.tanh
        self.bg_nl = torch.sigmoid
    
    def forward(self, x, force_no_p_nl=False):
        """

        Args:
            x:
            force_no_p_nl:

        Returns:

        """
        o = self._forward_core(x)

        o_head = []
        for i in range(self.ch_out):
            o_head.append(self.mt_heads[i].forward(o))
        o = torch.cat(o_head, 1)

        """Apply the final non-linearities"""
        if not self.training and not force_no_p_nl:
            o[:, [0]] = self.p_nl(o[:, [0]])

        if self._use_last_nl:
            o = self.apply_nonlin(o)

        return o
    
    def _forward_core(self, x) -> torch.Tensor:
        #print(self.ch_in)
        if self.ch_in == 3:
            x0 = x[:, [0]]
            #print(x.shape)
            x1 = x[:, [1]]
            x2 = x[:, [2]]

            o0 = self.unet_shared.forward(x0)
            o1 = self.unet_shared.forward(x1)
            o2 = self.unet_shared.forward(x2)

            o = torch.cat((o0, o1, o2), 1)

        elif self.ch_in == 1:
            o = self.unet_shared.forward(x)

        o = self.unet_union.forward(o)

        return o

class New_model(DoubleMUnet):
    
    ch_out = 10
    out_channels_heads = (1, 4, 4, 1)  # p head, phot,xyz_mu head, phot,xyz_sig head, bg head

    sigmoid_ch_ix = [0, 1, 5, 6, 7, 8, 9]  # channel indices with respective activation function
    tanh_ch_ix = [2, 3, 4]

    p_ch_ix = [0]  # channel indices of the respective parameters
    pxyz_mu_ch_ix = slice(1, 5)
    pxyz_sig_ch_ix = slice(5, 9)
    #crlb_sig_ch_ix = slice(9, 11)
    bg_ch_ix = [9]
    sigma_eps_default = 0.001


    def __init__(self, ch_in: int, *, depth_shared: int, depth_union: int, initial_features: int, inter_features: int,
                 norm=None, norm_groups=None, norm_head=None, norm_head_groups=None, pool_mode='StrideConv',
                 upsample_mode='bilinear', skip_gn_level: Union[None, bool] = None,
                 activation=nn.ReLU(), kaiming_normal=True):

        super().__init__(ch_in=ch_in, ch_out=self.ch_out, depth_shared=depth_shared, depth_union=depth_union,
                         initial_features=initial_features, inter_features=inter_features,
                         norm=norm, norm_groups=norm_groups, norm_head=norm_head,
                         norm_head_groups=norm_head_groups, pool_mode=pool_mode,
                         upsample_mode=upsample_mode,
                         skip_gn_level=skip_gn_level, activation=activation,
                         use_last_nl=False)

        self.transformer_encoder = TransformerEncoder(64, 2, 128, 3, 0.1)

        # 嵌入层，将 U-Net 输出维度映射到 embed_size
        self.embedding = nn.Conv2d(inter_features, 64, kernel_size=1)


        self.mt_heads = torch.nn.ModuleList(
            [MLTHeads(in_channels=64, out_channels=ch_out,
                                  activation=activation, last_kernel=1, padding=True,
                                  norm=norm_head, norm_groups=norm_head_groups)
             for ch_out in self.out_channels_heads]
        )

        """Register sigma as parameter such that it is stored in the models state dict and loaded correctly."""
        self.register_parameter('sigma_eps', torch.nn.Parameter(torch.tensor([self.sigma_eps_default]),
                                                                requires_grad=False))

        if kaiming_normal:
            self.apply(self.weight_init)

            # custom
            torch.nn.init.kaiming_normal_(self.mt_heads[0].core[0].weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.mt_heads[0].out_conv.weight, mode='fan_in', nonlinearity='linear')
            torch.nn.init.constant_(self.mt_heads[0].out_conv.bias, -6.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_core(x)
        
        o = self.embedding(x)  # 使用卷积层进行嵌入
        batch_size, channels, height, width = o.shape

        # 准备输入 Transformer
        o = o.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

        # 调用 Transformer Encoder
        o = self.transformer_encoder(o)

        # 将输出形状恢复为图像格式
        x = o.reshape(batch_size, channels, height, width)
        
        """Forward through the respective heads"""
        x_heads = [mt_head.forward(x) for mt_head in self.mt_heads]
        x = torch.cat(x_heads, dim=1)

        """Clamp prob before sigmoid"""
        x[:, [0]] = torch.clamp(x[:, [0]], min=-8., max=8.)

        """Apply non linearities"""
        x[:, self.sigmoid_ch_ix] = torch.sigmoid(x[:, self.sigmoid_ch_ix])
        x[:, self.tanh_ch_ix] = torch.tanh(x[:, self.tanh_ch_ix])

        """Add epsilon to sigmas and rescale"""
        x[:, self.pxyz_sig_ch_ix] = x[:, self.pxyz_sig_ch_ix] * 3 + self.sigma_eps
        #x[:, self.crlb_sig_ch_ix] = x[:, self.crlb_sig_ch_ix] * 3 + self.sigma_eps

        return x

    def apply_detection_nonlin(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply_nonlin(self, o: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def parse(cls, param, **kwargs):

        activation = getattr(torch.nn, param.HyperParameter.arch_param.activation)
        activation = activation()
        return cls(
            ch_in=param.HyperParameter.channels_in,
            depth_shared=param.HyperParameter.arch_param.depth_shared,
            depth_union=param.HyperParameter.arch_param.depth_union,
            initial_features=param.HyperParameter.arch_param.initial_features,
            inter_features=param.HyperParameter.arch_param.inter_features,
            activation=activation,
            norm=param.HyperParameter.arch_param.norm,
            norm_groups=param.HyperParameter.arch_param.norm_groups,
            norm_head=param.HyperParameter.arch_param.norm_head,
            norm_head_groups=param.HyperParameter.arch_param.norm_head_groups,
            pool_mode=param.HyperParameter.arch_param.pool_mode,
            upsample_mode=param.HyperParameter.arch_param.upsample_mode,
            skip_gn_level=param.HyperParameter.arch_param.skip_gn_level,
            kaiming_normal=param.HyperParameter.arch_param.init_custom
        )

    @staticmethod
    def weight_init(m):
        """
        Apply Kaiming normal init. Call this recursively by model.apply(model.weight_init)

        Args:
            m: model

        """
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')