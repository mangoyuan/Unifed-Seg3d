#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.generic_UNet import (StackedConvLayers, Upsample, ConvDropoutNormNonlin)
import torch.nn.functional

from nnunet.paths import m_dim


class SELayer(nn.Module):
    def __init__(self, channels, conv_op, reduction=2):
        super(SELayer, self).__init__()
        self.conv_op = conv_op
        if conv_op == nn.Conv2d:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        s = x.size()
        y = self.avg_pool(x).view(s[0], s[1])
        if self.conv_op == nn.Conv2d:
            y = self.fc(y).view(s[0], s[1], 1, 1)
        else:
            y = self.fc(y).view(s[0], s[1], 1, 1, 1)
        return x * y.expand_as(x)


class Idt(nn.Module):
    def __init__(self):
        super(Idt, self).__init__()

    def forward(self, x):
        return x


class UAGAN(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 # Add New Args.
                 reduction=2, atten_bitmap=None):
        """
        The Unified Attentional GAN Architecture.
        Include Two Parts: Translation and Segmentation. Each part uses half base_num_features.
        We must keep the input args unchanged in order to use the nnunet framework. Orz...
        @c_dim: the number of the modality, default=4.
        """
        super(UAGAN, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
             nonlin_kwargs = {'negative_slope': 1e-2, 'inplace':True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self.do_ds = deep_supervision
        self.reduction = reduction

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        # use bitmap to determine which levels of upsampling using attention
        if atten_bitmap is None:
            self.atten_bitmap = [1] * len(self.pool_op_kernel_sizes)
        else:
            self.atten_bitmap = atten_bitmap
        assert len(self.atten_bitmap) == len(self.pool_op_kernel_sizes)

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        self.conv_blocks_context = []
        self.td = []
        # The module list of segmentation. The bottleneck block is seg_conv_blocks_context[-1].
        self.seg_conv_blocks_localization = []
        self.seg_tu = []
        self.cross_atten = []
        self.seg_outputs = []

        # The module list of translation.
        self.tsl_conv_blocks_localization = []
        self.tsl_tu = []
        self.tsl_outputs = []

        output_features = base_num_features
        input_features = input_channels

        # -----------------------------------------------------
        #                Down Sample Path
        # -----------------------------------------------------
        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d-1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(
                StackedConvLayers(input_features, output_features, num_conv_per_stage, self.conv_op,
                                  self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, first_stride))

            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            if self.conv_op == nn.Conv3d:
                output_features = min(output_features, self.MAX_NUM_FILTERS_3D)
            else:
                output_features = min(output_features, self.MAX_FILTERS_2D)

        # -----------------------------------------------------
        #                Bottle Neck Blocks
        # -----------------------------------------------------
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # -----------------------------------------------------
        #                   Up Sample Path
        # -----------------------------------------------------
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            # self.conv_blocks_context[-1] is bottleneck, so start with -2
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.seg_tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u+1)],
                                              pool_op_kernel_sizes[-(u+1)], bias=False))
                self.tsl_tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u+1)],
                                              pool_op_kernel_sizes[-(u+1)], bias=False))
                # self.seg_tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
                # self.tsl_tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.seg_tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u+1)],
                                              pool_op_kernel_sizes[-(u+1)], bias=False))
                self.tsl_tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u+1)],
                                              pool_op_kernel_sizes[-(u+1)], bias=False))

            # Stacked Convolution Blocks
            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u+1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u+1)]
            self.seg_conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs)
            ))
            self.tsl_conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs)
            ))

            # Adding Attentional Blocks
            if self.atten_bitmap[u] == 1:
                self.cross_atten.append(SELayer(final_num_features, conv_op, reduction=self.reduction))
            else:
                self.cross_atten.append(Idt())

        for ds in range(len(self.seg_conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.seg_conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, False))
            self.tsl_outputs.append(conv_op(self.tsl_conv_blocks_localization[ds][-1].output_channels, 1,
                                            1, 1, 0, 1, 1, False))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl+1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.seg_conv_blocks_localization = nn.ModuleList(self.seg_conv_blocks_localization)
        self.seg_tu = nn.ModuleList(self.seg_tu)
        self.cross_atten = nn.ModuleList(self.cross_atten)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)

        self.tsl_conv_blocks_localization = nn.ModuleList(self.tsl_conv_blocks_localization)
        self.tsl_tu = nn.ModuleList(self.tsl_tu)
        self.tsl_outputs = nn.ModuleList(self.tsl_outputs)

        if self.upscale_logits:
            # lambda x:x is not a Module so we need to distinguish here
            self.upscale_logits_ops = nn.ModuleList(self.upscale_logits_ops)

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

    def forward(self, x, vec_org=None, vec_trg=None, output_tsl=False):
        skips = []
        seg_outputs, tsl_outputs = [], []

        if self.conv_op == nn.Conv3d:
            b, _, d, h, w = x.size()
            if vec_org is None or vec_trg is None:
                diff_vec = torch.zeros((b, m_dim, d, h, w)).cuda()
            else:
                diff_vec = vec_trg - vec_org
                diff_vec = diff_vec.view(diff_vec.size(0), diff_vec.size(1), 1, 1, 1)
                diff_vec = diff_vec.repeat(1, 1, d, h, w)
        else:
            b, _, h, w = x.size()
            if vec_org is None or vec_trg is None:
                diff_vec = torch.zeros((b, m_dim, h, w)).cuda()
            else:
                diff_vec = vec_trg - vec_org
                diff_vec = diff_vec.view(diff_vec.size(0), diff_vec.size(1), 1, 1)
                diff_vec = diff_vec.repeat(1, 1, h, w)

        x = torch.cat([x, diff_vec], dim=1)

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x_tsl = x_seg = self.conv_blocks_context[-1](x)

        for u in range(len(self.seg_tu)):
            x_tsl, x_seg = self.tsl_tu[u](x_tsl), self.seg_tu[u](x_seg)
            x_tsl, x_seg = torch.cat((x_tsl, skips[-(u+1)]), dim=1), torch.cat((x_seg, skips[-(u+1)]), dim=1)

            x_tsl, x_seg = self.tsl_conv_blocks_localization[u](x_tsl), self.seg_conv_blocks_localization[u](x_seg)
            if self.atten_bitmap[u] == 1:
                x_seg = x_seg + self.cross_atten[u](x_tsl)
            else:
                pass

            tsl_outputs.append(torch.tanh(self.tsl_outputs[u](x_tsl)))
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x_seg)))

        # if self.do_ds:
        #     return tuple([seg_outputs[-1]] + [i(j) for i, j in
        #                                       zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        if output_tsl:
            return seg_outputs[-1], tsl_outputs[-1]
        else:
            return seg_outputs[-1]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :return:
        """

        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64(5 * np.prod(map_size) * base_num_features + num_modalities * np.prod(map_size) + \
              num_classes * np.prod(map_size))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = 5 if p < (npool -1) else 2 # 2 + 2 for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size) * num_feat
            # print(p, map_size, num_feat, tmp)
        return tmp


class Discriminator(nn.Module):
    def __init__(self, input_size, input_channels=1, base_num_features=32, c_dim=4,
                 repeat_num=5, weightInitializer=None):
        super(Discriminator, self).__init__()
        if len(input_size) == 3:
            conv_op = nn.Conv3d
            ks = [4, 4, 4]
            stride = [2, 2, 2]
        else:
            conv_op = nn.Conv2d
            ks = [4, 4]
            stride = [2, 2]

        cur_size = [(ipsz + 1) // 2 for ipsz in input_size]
        ks = [ks[i] if cur_size[i] != 1 else 2 for i in range(len(input_size))]
        stride = [stride[i] if cur_size[i] != 1 else 2 for i in range(len(input_size))]
        layers = [conv_op(input_channels, base_num_features, kernel_size=tuple(ks), stride=tuple(stride), padding=1),
                  nn.LeakyReLU(0.001, inplace=True)]

        cur_num_features = base_num_features
        for i in range(1, repeat_num):
            cur_size = [(ipsz + 1) // 2 for ipsz in cur_size]
            ks = [ks[i] if cur_size[i] != 1 else 2 for i in range(len(input_size))]
            stride = [stride[i] if cur_size[i] != 1 else 2 for i in range(len(input_size))]

            layers.append(conv_op(cur_num_features, cur_num_features * 2,
                                  kernel_size=tuple(ks), stride=tuple(stride), padding=1))
            layers.append(nn.LeakyReLU(0.001, inplace=True))
            cur_num_features = cur_num_features * 2

        ks = [int(size / np.power(2, repeat_num)) for size in input_size]
        ks = [k if k != 0 else 1 for k in ks]

        self.main = nn.Sequential(*layers)
        self.d_src = conv_op(cur_num_features, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.d_cls = conv_op(cur_num_features, c_dim, kernel_size=tuple(ks), stride=1, bias=True)

        if weightInitializer is not None:
            self.apply(weightInitializer)

    def forward(self, x):
        hidden = self.main(x)
        out_src = self.d_src(hidden)
        out_cls = self.d_cls(hidden)
        # print(hidden.size(), out_src.size(), out_cls.size())
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


if __name__ == '__main__':
    uagan = UAGAN(5, 30, 4, 5, deep_supervision=False, dropout_op=None, conv_op=nn.Conv3d, norm_op=nn.InstanceNorm3d,
                  convolutional_upsampling=True, atten_bitmap=[0, 0, 0, 0, 1], reduction=6)
    uagan = uagan.cuda()
    # d = Discriminator((28, 160, 256), repeat_num=5).cuda()
    # print(d)
    # x = torch.randn((1, 1, 28, 160, 256)).cuda()
    # d(x)
    # uagan.cuda()

    # import thop
    x = torch.randn((1, 1, 96, 128, 128)).cuda()
    a, b = uagan(x, output_tsl=True)
    print(a.size(), b.size())
    # c, d = d(b)
    # print(c.size(), d.size())
    # print(thop.profile(uagan, (x, )))
