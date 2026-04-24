import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.rnn = nn.LSTM(128, 256, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)

        b, c, h, w = x.size()

        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)

        x, _ = self.rnn(x)
        x = self.fc(x)

        return x