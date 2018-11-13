from Librarys import *


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape(3,220,128)
            nn.Conv2d(  # 84480
                in_channels=3,  # test  shape(3,28,28)
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(  # input shape(32,110,64)
            nn.Conv2d(32, 64, 5, 1, 2),  # output shape(64,55,32)
            nn.ReLU(),  # test shape(32,14,14)
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(  # input shape(64,110,64)
            nn.Conv2d(64, 128, 5, 1, 2),  # output shape(128,27,16)
            nn.ReLU(),  # test shape(64,7,7)
            nn.MaxPool2d(kernel_size=2),
        )  # test shape(128,3,3)

        self.out = nn.Linear(128 * 3 * 3, 2)
        # self.out = nn.Linear(64*7*7, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


class CNN1(nn.Module):

    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Sequential(  # input shape(1,220,128)
            nn.Conv2d(  # 84480
                in_channels=1,  # test shape(1,28,28)
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(  # input shape(32,110,64)
            nn.Conv2d(32, 64, 5, 1, 2),  # output shape(64,55,32)
            nn.ReLU(),  # test shape(32,14,14)
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(  # input shape(64,110,64)
            nn.Conv2d(64, 128, 5, 1, 2),  # output shape(128,27,16)
            nn.ReLU(),  # test shape(64,7,7)
            nn.MaxPool2d(kernel_size=2),
        )  # test shape(128,3,3)

        self.out = nn.Linear(128 * 3 * 3, 2)
        # self.out = nn.Linear(64*7*7, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x
