import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
              nn.BatchNorm2d(out_channels),
              nn.ReLU()]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
    return nn.Sequential(*layers)


class ResNet14(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 32)
        self.conv2 = conv_block(32, 64, pool=True) # output: 16 * 16 * 64
        self.res_block1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))

        self.conv3 = conv_block(64, 128, pool=True) # output: 8 * 8 * 128
        self.res_block2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv4 = conv_block(128, 256, pool=True) # output: 4 * 4 * 256
        self.res_block3 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))

        self.conv5 = conv_block(256, 512, pool=True)  # output: 2 * 2 * 512
        self.res_block4 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2)), nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res_block1(out) + out

        out = self.conv3(out)
        out = self.res_block2(out) + out

        out = self.conv4(out)
        out = self.res_block3(out) + out

        out = self.conv5(out)
        out = self.res_block4(out) + out

        out = self.classifier(out)
        return out


class CIFARDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        super(CIFARDataset, self).__init__()
        self.data = torch.from_numpy(data).view([-1, 3, 32, 32])
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[index]


def run_model(data_train, data_test, trial=None, params=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    if device == 'cuda':
        pin_memory = True
    else:
        pin_memory = False
    epochs = 50

    transforms_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ])

    transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ])

    if params is None:
        if trial is None:
            raise AttributeError('trial must be called not NONE')
        params = dict()
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-10, 1e-2)
        params['weight_decay'] = trial.suggest_float('weight_decay', 1e-10, 1e-2)
        params['clip_grad'] = trial.suggest_float('clip_grad', 0.01, 0.9)
        params['gamma'] = trial.suggest_float('gamma', 0.9, 1.0)
        params['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        #params['optimizer'] = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])

    model = ResNet14(3, 10)

    model = model.to(device)
    print(f'params: {params}')

    #optimizer = getattr(optim, params['optimizer'])\
        #(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    lr_sheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=params['gamma'])
    criterion = nn.CrossEntropyLoss()

    dataset_train = CIFARDataset(data_train['data'], data_train['labels'], transform=transforms_train)
    loader_train = DataLoader(dataset=dataset_train, batch_size=params['batch_size'], num_workers=2,
                              shuffle=True, pin_memory=pin_memory)

    dataset_test = CIFARDataset(data_test['data'], data_test['labels'], transform=transforms_test)
    loader_test = DataLoader(dataset=dataset_test, batch_size=params['batch_size'], num_workers=2,
                             shuffle=True, pin_memory=pin_memory)

    best_accuracy = 0.0
    for epoch in range(epochs):
        # Train model
        for data, labels in loader_train:
            X = data.to(device)
            y = labels.to(device)

            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()

            nn.utils.clip_grad_value_(model.parameters(), clip_value=params['clip_grad'])

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        lr_sheduler.step()

        # Check accuracy on the testing dataset
        with torch.no_grad():
            model.eval()
            preds, y = torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)
            for data, labels in loader_test:
                X = data.to(device)
                y = torch.cat((y, labels), dim=0)
                preds = torch.cat((preds, torch.argmax(model(X).cpu(), dim=1)), dim=0)
            accuracy = accuracy_score(y, preds)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            print(f'Test accuracy: {accuracy} epoch: {epoch}')
            model.train()

    return best_accuracy
