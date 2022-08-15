import torch.nn as nn
import torch
import torchvision.transforms as transforms
from .binarized_modules import  BinarizeLinear,BinarizeConv2d
# from binarized_modules import  BinarizeLinear,BinarizeConv2d
__all__ = ['kws_bnn_binary3']


class kws_bnn3(nn.Module):

    def __init__(self, num_classes=2):
        super(kws_bnn3, self).__init__()
        self.ratioInfl=3
        self.features = nn.Sequential(
            # 1
            BinarizeConv2d(1, int(1*self.ratioInfl), kernel_size=(5,1), stride=(3,1), padding=(0,0),bias=False),
            nn.Hardtanh(inplace=True),
        )
        self.classifier = nn.Sequential(
            BinarizeLinear(int(1*self.ratioInfl)*6*6, num_classes),
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


def kws_bnn_binary3(**kwargs):
    num_classes = kwargs.get( 'num_classes', 2)
    return kws_bnn3(num_classes)

if __name__ == '__main__':
    from torchsummary import summary
    from torchstat import stat
    import torch
    # device = torch.device("cuda")
    model = kws_bnn3()
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
