import torch.nn as nn
import torch
import torchvision.transforms as transforms
from .binarized_modules import  BinarizeLinear,BinarizeConv2d
# from binarized_modules import  BinarizeLinear,BinarizeConv2d
__all__ = ['kws_bnn_binary2']

# onnx 对 BN1d 不支持，所以需要自定义BN1d
# 仅仅在转 onnx 时候使用
# class MyBatchNorm1d(nn.BatchNorm1d):
#     def __init__(self, num_features):
#         super(MyBatchNorm1d, self).__init__(num_features)
#         self.eps = 1e-5
#         self.affine = True

#     def forward(self, input):
#         self._check_input_dim(input)
#         # calculate running estimates
#         input = (input - self.running_mean) / (torch.sqrt(self.running_var + self.eps))
#         if self.affine:
#             input = input * self.weight + self.bias
#         return input

class kws_bnn2(nn.Module):

    def __init__(self, num_classes=2):
        super(kws_bnn2, self).__init__()
        self.ratioInfl=3
        self.features = nn.Sequential(
            # 1
            # BinarizeConv2d(3, int(64*self.ratioInfl), kernel_size=11, stride=4, padding=2),
            BinarizeConv2d(1, int(1*self.ratioInfl), kernel_size=(5,1), stride=(3,1), padding=(0,0),bias=False),
            # BinarizeConv2d(1, 3, kernel_size=(1,5), stride=(1,3), padding=(0,0), bias = False),

            # nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(int(1*self.ratioInfl)),
            nn.Hardtanh(inplace=True),
        )
        self.classifier = nn.Sequential(
            # BinarizeLinear(256 * 6 * 6, 4096),
            # nn.BatchNorm1d(4096),
            # nn.Hardtanh(inplace=True),
            # #nn.Dropout(0.5),
            # BinarizeLinear(4096, 4096),
            # nn.BatchNorm1d(4096),
            # nn.Hardtanh(inplace=True),
            # #nn.Dropout(0.5),


            # BinarizeLinear(4096, num_classes),
            # nn.BatchNorm1d(1000),
            # nn.LogSoftmax()
            BinarizeLinear(int(1*self.ratioInfl)*6*6, num_classes),
            nn.BatchNorm1d(num_classes),
            # MyBatchNorm1d(num_classes),
            nn.LogSoftmax()
        )

        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-2,
        #        'weight_decay': 5e-4, 'momentum': 0.9},
        #    10: {'lr': 5e-3},
        #    15: {'lr': 1e-3, 'weight_decay': 0},
        #    20: {'lr': 5e-4},
        #    25: {'lr': 1e-4}
        #}
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            20: {'lr': 1e-3},
            30: {'lr': 5e-4},
            35: {'lr': 1e-4},
            40: {'lr': 1e-5}
        }
        # 改为了 MNIST 的 std 和 mean
        # normalize = transforms.Normalize((0.1307,), (0.3081,))
        normalize = None
        # 改为了 MNIST transform 处理方式
        self.input_transform = {
            'train': transforms.Compose([
                # transforms.Resize(256),#Scale 新版本 torch 中 已经取消，换成 resize
                # transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        # x = x.view(-1, 256 * 6 * 6) # 256 * 6 * 6 = 9216
        x = x.view(-1, 3 * 6 * 6) # 256 * 6 * 6 = 9216
        x = self.classifier(x)
        return x


def kws_bnn_binary2(**kwargs):
    num_classes = kwargs.get( 'num_classes', 2)
    return kws_bnn2(num_classes)

if __name__ == '__main__':
    from torchsummary import summary
    from torchstat import stat
    import torch
    # device = torch.device("cuda")
    model = kws_bnn2()
    # .to(device)
    input=torch.Tensor(1,20,6)
    # summary(model,(224,224))
    # input = torch.Tensor(3,224,224)
    # input.cuda()
    # print(len(input))
    # print(type(input))
    # input.to(device)
    stat(model,(1,20,6))
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameters: %.2fM" % (total/1e6))
