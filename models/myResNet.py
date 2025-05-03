import sys
import os

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.split(dir_path)[:-1][0]
    dir_path = os.path.split(dir_path)[:-1][0]
    sys.path.append(dir_path)
import torch
import torch.nn as nn
import torchvision
# from earlyexitnet.tools import get_output_shape


def get_output_shape(module_list, img_dim):
    # returns output shape
    device = 'cpu'
    tmp_=0
    for module in module_list:
        module = module.to(device)
        if tmp_==0:
            dims = module(torch.rand(*(img_dim))).data.shape
        else:
            dims = module(torch.rand(*(dims))).data.shape
        tmp_+=1
    return dims


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

        self.residual_function = [self.conv1, self.bn1, self.relu, self.conv2, self.bn2]

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.downsample = nn.Sequential()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        y = self.downsample(x)
        for i in self.residual_function:
            x = i(x)
        x += y
        x = self.relu(x)
        return x


class BottleNeck(BasicBlock):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__(in_channels, out_channels, stride)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.residual_function = [self.conv1, self.bn1, self.relu, self.conv2, self.bn2, self.relu, self.conv3,
                                  self.bn3]


#####################################################
# TODO myResNet
# ResNet8 backbone for training experiments - weights should be transferable
class ResNet_backbone(nn.Module):
    def __init__(self, block=BasicBlock, num_block=[], num_classes=1000, init_weights=True):
        super(ResNet_backbone, self).__init__()
        self.exit_num = 1  # TODO -1 because of final exit

        self.block = block
        self.num_classes = num_classes
        self.init_weights = init_weights
        ## 이 부분을 수정해보자 (input size, layer 별로 맞춰두기)
        self.input_size = 32
        self.in_chans = 64

        self.num_block = num_block
        self.in_chan_sizes = [64, 128, 256, 512]
        self.strides = [1, 2, 2, 2]

        # init_conv Layer
        self.conv1 = nn.Conv2d(3, self.in_chans, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_chans)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.init_conv = [self.conv1, self.bn1, self.relu, self.maxpool]

        # backbone Layer
        # self.backbone = nn.ModuleList()
        self.layer1 = self._make_layer(self.block, self.in_chan_sizes[0], self.num_block[0], self.strides[0])
        self.layer2 = self._make_layer(self.block, self.in_chan_sizes[1], self.num_block[1], self.strides[1])
        self.layer3 = self._make_layer(self.block, self.in_chan_sizes[2], self.num_block[2], self.strides[2])
        self.layer4 = self._make_layer(self.block, self.in_chan_sizes[3], self.num_block[3], self.strides[3])

        # self._build_backbone()
        # print(self.backbone)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.end_layers = [self.avgpool, self.flatten, self.fc]

        # init weights+biases according to mlperf tiny
        if init_weights:
            self._weight_init()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_chans, out_channels, stride))
            self.in_chans = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _build_backbone(self):
        for i in range(len(self.num_block)):
            if i == 0:
                layer = self._make_layer(self.block, self.in_chan_sizes[i], self.num_block[i], self.strides[i])
                self.layer1.append(layer)
            elif i == 1:
                layer = self._make_layer(self.block, self.in_chan_sizes[i], self.num_block[i], self.strides[i])
                self.layer2.append(layer)
            elif i == 2:
                layer = self._make_layer(self.block, self.in_chan_sizes[i], self.num_block[i], self.strides[i])
                self.layer3.append(layer)
            elif i == 3:
                layer = self._make_layer(self.block, self.in_chan_sizes[i], self.num_block[i], self.strides[i])
                self.layer4.append(layer)

        return

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # init_conv Layer
        print('input shape:', x.shape)
        x = self.conv1(x)
        print('conv1 shape:', x.shape)
        x = self.bn1(x)
        print('bn1 shape:', x.shape)
        x = self.relu(x)
        print('relu shape:', x.shape)
        y = self.maxpool(x)
        print('maxpool shape:', y.shape)

        # y = self.init_conv(x)

        for b in self.layer1:
            y = b(y)
        for b in self.layer2:
            y = b(y)
        for b in self.layer3:
            y = b(y)
        for b in self.layer4:
            y = b(y)
        for b in self.end_layers:
            y = b(y)
        return [y]


## MSDNet Classifier
class ConvBasic(nn.Module):
    def __init__(self, chanIn, chanOut, k=3, s=1,
                 p=1, p_ceil_mode=False, bias=False):
        super(ConvBasic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(chanIn, chanOut, kernel_size=k, stride=s,
                      padding=p, bias=bias),
            nn.BatchNorm2d(chanOut),
            nn.ReLU(True)  # in place
        )

    def forward(self, x):
        return self.conv(x)

#
# class IntrClassif(nn.Module):
#     # intermediate classifer head to be attached along the backbone
#     # Inpsired by MSDNet classifiers (from HAPI):
#     # https://github.com/kalviny/MSDNet-PyTorch/blob/master/models/msdnet.py
#
#     def __init__(self, chanIn, input_shape, classes, bb_index, interChans=64, last_conv_chan=32):
#         super(IntrClassif, self).__init__()
#
#         # index for the position in the backbone layer
#         self.bb_index = bb_index
#         # input shape to automatically size linear layer
#         self.input_shape = input_shape
#
#         # intermediate conv channels
#         # interChans = 128 # TODO reduce size for smaller nets
#         self.interChans = interChans
#         self.last_conv_chan = last_conv_chan
#         # conv, bnorm, relu 1
#         layers = nn.ModuleList()
#         self.conv1 = ConvBasic(chanIn, interChans, k=5, s=1, p=2)
#         layers.append(self.conv1)
#         self.conv2 = ConvBasic(interChans, interChans, k=3, s=1, p=1)
#         layers.append(self.conv2)
#         self.conv3 = ConvBasic(interChans, last_conv_chan, k=3, s=1, p=1)
#         layers.append(self.conv3)
#         self.layers = layers
#
#         self.linear_dim = int(torch.prod(torch.tensor(self._get_linear_size(layers))))
#         # print(f"Classif @ {self.bb_index} linear dim: {self.linear_dim}") #check linear dim
#
#         # linear layer
#         self.linear = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(self.linear_dim, classes)
#         )
#
#     def _get_linear_size(self, layers):
#         for layer in layers:
#             self.input_shape = get_output_shape(layer, self.input_shape)
#         return self.input_shape
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return self.linear(x)


class ResNet_2EE(ResNet_backbone):
    # basic early exit network for resnet8
    def __init__(self, block=BasicBlock, num_block=[], num_classes=1000, data_shape=[1, 3, 32, 32],
                 init_weights=True):
        '''
    data_shape: batch size must be 1. ex) [1,3,32,32]
    '''
        super(ResNet_2EE, self).__init__(block=block, num_block=num_block,
                                         num_classes=num_classes, init_weights=init_weights)

        # NOTE structure:
        # init conv -> exit1
        # self.backbone
        # self.end_layer (avg pool, flatten, linear)

        self.exits = nn.ModuleList()
        # weighting for each exit when summing loss
        self.input_shape = data_shape  # input data shape /batch, channels, height, width

        self.exit_num = 2
        self.fast_inference_mode = False
        self.exit_loss_weights = [1.0, 1.0]  # for training need to match total exits_num
        self.exit_threshold = torch.tensor([0.8],
                                           dtype=torch.float32)  # for fast inference  #TODO: inference variable(not constant 0.8) need to make parameter
        # self._build_exits()

    # def _build_exits(self):  # adding early exits/branches
    #     # TODO generalise exit placement for multi exit
    #     # early exit 1
    #     previous_shape = get_output_shape(self.init_conv, self.input_shape)
    #     ee1 = IntrClassif(self.in_chan_sizes[0],
    #                       # TODO: branch before conv2 ->conv_x//why before conv2 not after? beacasue author did it wtf..
    #                       previous_shape, self.num_classes, 0)
    #     self.exits.append(ee1)
    #
    #     # final exit
    #     self.exits.append(self.end_layers)

    @torch.jit.unused  # decorator to skip jit comp
    def _forward_training(self, x):
        # TODO make jit compatible - not urgent
        # NOTE broken because returning list()
        # res = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        y = self.maxpool(x)
        # y = self.init_conv(x)
        # res.append(self.exits[0](y))
        # compute remaining backbone layers
        # for b in self.backbone:
        #     y = b(y)
        for b in self.layer1:
            y = b(y)
        for b in self.layer2:
            y = b(y)
        for b in self.layer3:
            y = b(y)
        for b in self.layer4:
            y = b(y)
        # final exit
        for b in self.end_layers:
            y = b(y)
        # y = self.end_layers(y)
        # res.append(y)

        return y

    def exit_criterion_top1(self, x):  # NOT for batch size > 1 (in inference mode)
        with torch.no_grad():
            pk = nn.functional.softmax(x, dim=-1)
            # top1 = torch.max(pk)          #originally x*log(x)#TODO np.sum(pk*log(pk))
            top1 = torch.log(pk) * pk
            return top1 < self.exit_threshold

    def forward(self, x):
        # std forward function
        if self.fast_inference_mode:
            # for bb, ee in zip(self.backbone, self.exits):
            #    x = bb(x)
            #    res = ee(x) #res not changed by exit criterion
            #    if self.exit_criterion_top1(res):
            #        return res
            # y = self.init_conv(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            y = self.maxpool(x)
            # res = self.exits[0](y)
            # if self.exit_criterion_top1(res):
            #     return res
            # compute remaining backbone layers
            # for b in self.backbone:
            #     y = b(y)
            for b in self.layer1:
                y = b(y)
            for b in self.layer2:
                y = b(y)
            for b in self.layer3:
                y = b(y)
            for b in self.layer4:
                y = b(y)
            # final exit
            res = self.exits[1](y)
            return res

        else:  # NOTE used for training
            # calculate all exits
            return self._forward_training(x)

    def set_fast_inf_mode(self, mode=True):
        if mode:
            self.eval()
        self.fast_inference_mode = mode


def resnet18():
    return ResNet_backbone(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet_backbone(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet_backbone(BottleNeck, [3, 4, 6, 3])


def resnet101():
    return ResNet_backbone(BottleNeck, [3, 4, 23, 3])


def resnet152():
    return ResNet_backbone(BottleNeck, [3, 8, 36, 3])


def resnet18_2EE():
    return ResNet_2EE(BasicBlock, [2, 2, 2, 2])


def resnet34_2EE():
    return ResNet_2EE(BasicBlock, [3, 4, 6, 3])


def resnet50_2EE():
    return ResNet_2EE(BottleNeck, [3, 4, 6, 3])


def resnet101_2EE():
    return ResNet_2EE(BottleNeck, [3, 4, 23, 3])


def resnet152_2EE():
    return ResNet_2EE(BottleNeck, [3, 8, 36, 3])


if __name__ == "__main__":
    # tracking the myResNet's parameter
    parameter_track_constatnt = 0

    script_dir = os.path.dirname(__file__)
    pth_rel_path = "pretrained_model"
    file_name = "pretrained_resnet.pth"
    abs_file_path = os.path.join(script_dir, pth_rel_path)
    try:
        if not os.path.exists(abs_file_path):
            os.makedirs(abs_file_path)

    except OSError:
        print('Error: Creating directory. ' + abs_file_path)
    abs_file_path = os.path.join(abs_file_path, file_name)
    resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
    torch.save(resnet.state_dict(), abs_file_path)
    model = resnet101().to('cuda')
    model.load_state_dict(torch.load(abs_file_path))

    if parameter_track_constatnt:
        log_rel_path = "parameter_log.txt"
        abs_file_path = os.path.join(script_dir, log_rel_path)
        #  파일 열기
        f = open(abs_file_path, 'w')
        #  파일에 텍스트 쓰기
        for param_tensor in model.state_dict():
            f.write(f'{param_tensor} \t {model.state_dict()[param_tensor].size()}\n')
            #  파일 닫기
        f.close()