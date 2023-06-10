import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


######################################
#            Generators              #
######################################

class Generator(nn.Module):
    def __init__(self, num_classes):
        super(Generator, self).__init__()
        self.embed = nn.Embedding(num_classes, 100)
        self.dense = nn.Linear(100, 7 * 7 * 256)
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh())

    def forward(self, label):
        embedded_label = self.embed(label)
        z = torch.randn(len(label), 100).to(device)
        x = embedded_label * z
        x = self.dense(x).view(-1, 256, 7, 7)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Generator_num(nn.Module):
    def __init__(self):
        super(Generator_num, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(100, 256 * 7 * 7)
            # nn.BatchNorm1d(256 * 7 * 7)
        )
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh())

    def forward(self, batch_size):
        x = torch.randn(batch_size, 100).to(device)
        x = self.linear(x)
        x = x.view(-1, 256, 7, 7)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Generator_acgan(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Embedding which outputs a vector of dimension z_dim
        self.embed = nn.Embedding(num_classes, 100)

        # Linear combination of the latent vector z
        self.dense = nn.Linear(100, 7 * 7 * 256)

        # The transposed convolutional layers are wrapped in nn.Sequential
        self.trans1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.trans2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.trans3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh())

    def forward(self, label):
        # Apply embedding to the input label
        embedded_label = self.embed(label)
        z = torch.randn(len(label), 100).to(device)
        # Element wise multiplication of latent vector and embedding
        x = embedded_label * z

        # Application of dense layer and transforming to 3d shape
        x = self.dense(x).view(-1, 256, 7, 7)
        x = self.trans1(x)
        x = self.trans2(x)
        x = self.trans3(x)

        return x


######################################
#          Discriminators            #
######################################

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.embed = nn.Embedding(num_classes, 28 * 28)
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 3 * 128, 1),
            nn.Sigmoid())

    def forward(self, x, label):
        embedded_label = self.embed(label).view_as(x)
        x = torch.cat([x, embedded_label], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dense(x)
        return x


class Discriminator_num(nn.Module):
    def __init__(self):
        super(Discriminator_num, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, 1, 28, 28))
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 3 * 128, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = torch.cat([x, self.weight.repeat(x.shape[0], 1, 1, 1).to(device)], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dense(x)
        return x


class Discriminator_acgan(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Embedding which outputs a vector of img_size
        self.embed = nn.Embedding(num_classes, 28 * 28)

        # It convenient to group conv layers with nn.Sequential
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1),
            nn.LeakyReLU())

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.LeakyReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        # The 3D feature map is flattened to perform a linear combination
        self.dense_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 3 * 128, 1),
            nn.Sigmoid())
        self.dense_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 3 * 128, 20),
            nn.Softmax())

    def forward(self, x, label):
        # Apply embedding and convert to same shape as x
        embedded_label = self.embed(label).view_as(x)

        x_1 = self.conv_1(x)
        x_1 = self.conv2(x_1)
        x_1 = self.conv3(x_1)
        x_2 = self.dense_2(x_1)
        # Concatenation of x and embedded label
        x = torch.cat([x, embedded_label], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_1 = self.dense_1(x)
        # x_2 = self.dense_2(x)

        return x_1, x_2


######################################
#            Classifiers             #
######################################

class Classifier(nn.Module):
    def __init__(self, num_class):
        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 3 * 128, num_class))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dense(x)
        return x
