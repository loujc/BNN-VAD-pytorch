import torch.nn as nn
import torchvision.transforms as transforms
from .binarized_modules import  BinarizeLinear,BinarizeConv2d
# from binarized_modules import  BinarizeLinear,BinarizeConv2d
__all__ = ['alexnet_binary']

class AlexNetOWT_BN(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetOWT_BN, self).__init__()
        self.ratioInfl=3
        self.features = nn.Sequential(
            # 1
            BinarizeConv2d(3, int(64*self.ratioInfl), kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(int(64*self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            # 2
            BinarizeConv2d(int(64*self.ratioInfl), int(192*self.ratioInfl), kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(int(192*self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            # 3
            BinarizeConv2d(int(192*self.ratioInfl), int(384*self.ratioInfl), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(384*self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            # 4
            BinarizeConv2d(int(384*self.ratioInfl), int(256*self.ratioInfl), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(256*self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            # 5
            BinarizeConv2d(int(256*self.ratioInfl), 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True)

        )
        self.classifier = nn.Sequential(
            BinarizeLinear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(4096, num_classes),
            nn.BatchNorm1d(1000),
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
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Resize(256),#Scale 新版本 torch 中 已经取消，换成 resize
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6) # 256 * 6 * 6 = 9216
        x = self.classifier(x)
        return x


def alexnet_binary(**kwargs):
    num_classes = kwargs.get( 'num_classes', 1000)
    return AlexNetOWT_BN(num_classes)

if __name__ == '__main__':
    from torchsummary import summary
    from torchstat import stat
    import torch
    # device = torch.device("cuda")
    model = AlexNetOWT_BN()
    # .to(device)
    input=torch.Tensor(3,224,224)
    # summary(model,(224,224))
    # input = torch.Tensor(3,224,224)
    # input.cuda()
    # print(len(input))
    # print(type(input))
    # input.to(device)
    stat(model,(3,224,224))
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameters: %.2fM" % (total/1e6))
