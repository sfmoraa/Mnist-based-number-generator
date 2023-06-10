import os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm
from utils import sample_digits, tensor_to_img
from module import Classifier, Generator, Generator_num, Generator_acgan, Discriminator, Discriminator_num, Discriminator_acgan
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import DataLoader, RandomSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trans = Compose([ToTensor(), Lambda(lambda x: x * 2 - 1)])
train_set = MNIST('./data', train=True, transform=trans, download=True)
test_set = MNIST('./data', train=False, transform=trans, download=True)


######################################
#         Train Classifier           #
######################################

def classifier_train(batch_size=100):
    # gain train_data
    train_num = 60000
    train_sampler = RandomSampler(train_set, replacement=True, num_samples=train_num)
    train_loader = DataLoader(train_set, batch_size, sampler=train_sampler, drop_last=True)

    # train 
    train_epochs = trange(int(train_num / batch_size), ascii=True, leave=True, desc="Epoch", position=0)
    classifier = Classifier(11).to(device)
    classifier.load_state_dict(torch.load("weights/classifier_weights.pt"))
    print("TRAINING", '.' * 20)
    classifier.train()
    train_loss_sum = 0
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(params=classifier.parameters(), lr=0.001)

    for epoch in train_epochs:
        cur_datasets = next(iter(train_loader))
        imgs = cur_datasets[0].to(device)
        labels = cur_datasets[1].to(device)
        for _ in range(int(batch_size / 10)):
            random_num = random.randint(1, batch_size - 2)
            imgs[random_num] = (imgs[random_num] + imgs[random_num + 1] + imgs[random_num - 1]) / 3
            labels[random_num] = 10

        out = classifier(imgs).to(device)
        opt.zero_grad()
        loss = criterion(out, labels)
        loss.backward()
        train_loss_sum = train_loss_sum + loss.item()
        train_loss_score = train_loss_sum / ((epoch + 1) * batch_size)
        opt.step()
        train_epochs.set_description("Epoch (Loss=%g)" % round(train_loss_score, 5))

    torch.save(classifier.state_dict(), "weights/classifier_weights.pt")


######################################
#            Train GAN_1             #
######################################

def dis_gen_train_real(batch_size, iterations, sample_interval, generator, discriminator, criterion, g_optim, d_optim):
    sampler = RandomSampler(train_set, replacement=True, num_samples=batch_size * iterations)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, drop_last=True)
    y_real = torch.ones(batch_size, 1).to(device)
    y_fake = torch.zeros(batch_size, 1).to(device)

    losses = []
    iteration = 0
    d_loss = 0.0
    g_loss = 0.0

    for x, labels in tqdm(train_loader):
        x = x.to(device)
        labels = labels.to(device)

        # 1. Train discriminator on real images
        d_out = discriminator(x, labels)
        d_loss_real = criterion(d_out, y_real)
        d_optim.zero_grad()
        d_loss_real.backward()
        d_optim.step()

        # 2. Train discriminator on fake images

        x_gen = generator(labels)
        d_out = discriminator(x_gen, labels)
        d_loss_fake = criterion(d_out, y_fake)
        d_optim.zero_grad()
        d_loss_fake.backward()
        d_optim.step()

        # 3. Train generator to force discriminator making false predictions

        x_gen = generator(labels)
        d_out = discriminator(x_gen, labels)
        g_loss_real = criterion(d_out, y_real)
        g_optim.zero_grad()
        g_loss_real.backward()
        g_optim.step()

        # Calculate discriminator and generator loss for visualization
        d_loss += 0.5 * (d_loss_real + d_loss_fake).item()
        g_loss += g_loss_real.item()
        iteration += 1

        if iteration % sample_interval == 0:
            # Calculate the loss for the last sample_interval iterations
            d_loss = d_loss / sample_interval
            g_loss = g_loss / sample_interval

            print(f"D-Loss: {d_loss:.4f} G-Loss: {g_loss:.4f}")

            losses.append((d_loss, g_loss))
            d_loss = 0
            g_loss = 0

    return losses


def dis_gen_train(epoch_num=3):
    for epoch in range(epoch_num):
        generator = Generator(10).to(device)
        if os.path.exists("weights/generator_weights.pt"):
            print("using pretrained generator weights")
            generator.load_state_dict(torch.load("weights/generator_weights.pt"))

        discriminator = Discriminator(10).to(device)
        if os.path.exists("weights/discriminator_weights.pt"):
            print("using pretrained discriminator weights")
            discriminator.load_state_dict(torch.load("weights/discriminator_weights.pt"))

        g_optim = optim.Adam(generator.parameters(), lr=0.001)
        d_optim = optim.Adam(discriminator.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        losses = dis_gen_train_real(16, 2000, 100, generator, discriminator, criterion, g_optim, d_optim)

        sample_digits(generator, epoch=epoch)

        save = input("save or not?  y/n")
        if save == 'y':
            torch.save(generator.state_dict(), "weights/generator_weights.pt")
            torch.save(discriminator.state_dict(), "weights/discriminator_weights.pt")


######################################
#            Train GAN_2             #
######################################

def get_label_data(num=0, batch_size=10):
    x = train_set.data / 255 * 2 - 1
    x = x.unsqueeze(dim=1)
    label = train_set.targets
    data = list()
    part = list()
    part_num = 0
    for i in range(len(label)):
        if label[i] == num:
            part_num += 1
            part.append(x[i].numpy())
            if part_num % 10 == 0:
                data.append(torch.tensor(np.array(part)))
                part_num = 0
                part = list()
    return data


def number_train_real(num=0, batch_size=10):
    gen = Generator_num().to(device)
    dis = Discriminator_num().to(device)
    gen_path = "weights/GEN_{}.pt".format(num)
    dis_path = "weights/DIS_{}.pt".format(num)
    if os.path.exists(gen_path):
        gen.load_state_dict(torch.load(gen_path))
    if os.path.exists(dis_path):
        dis.load_state_dict(torch.load(dis_path))

    data = get_label_data(num)
    epochs = trange(len(data), ascii=True, leave=True, desc="Epoch", position=0)
    criterion = nn.BCELoss()
    d_optim = optim.Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optim = optim.Adam(gen.parameters(), lr=0.0001, betas=(0.5, 0.999))
    d_loss = 0
    g_loss = 0
    y_real = torch.ones(batch_size, 1).to(device)
    y_fake = torch.zeros(batch_size, 1).to(device)
    for epoch in epochs:
        x = data[epoch].to(device)
        # 1. Train discriminator on real images
        d_out = dis(x)
        d_loss_real = criterion(d_out, y_real)
        d_optim.zero_grad()
        d_loss_real.backward()
        d_optim.step()

        # 2. Train discriminator on fake images
        x_gen = gen(batch_size)
        d_out = dis(x_gen)
        d_loss_fake = criterion(d_out, y_fake)
        d_optim.zero_grad()
        d_loss_fake.backward()
        d_optim.step()

        # 3. Train generator to force discriminator making false predictionss
        x_gen = gen(batch_size)
        d_out = dis(x_gen)
        g_loss_real = criterion(d_out, y_real)
        g_optim.zero_grad()
        g_loss_real.backward()
        g_optim.step()

        d_loss += 0.5 * (d_loss_real + d_loss_fake).item()
        g_loss += g_loss_real.item()
        message = "DLoss={}, GLoss={}".format(round(d_loss / (epoch + 1), 5), round(g_loss / (epoch + 1), 5))
        epochs.set_description(message)

    # tensor_to_img(gen(batch_size=100))
    torch.save(gen.state_dict(), "weights/GEN_{}.pt".format(num))
    torch.save(dis.state_dict(), "weights/DIS_{}.pt".format(num))


def number_train():
    for num in range(10):
        for _ in range(5):
            number_train_real(num)


######################################
#            Train GAN_3             #
######################################

def acgan_train_real(batch_size, iterations, sample_interval, generator, discriminator, criterion_1, criterion_2,
                     g_optim, d_optim, _min):
    sampler = RandomSampler(train_set, replacement=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, drop_last=True)

    y_real = torch.ones(batch_size, 1).to(device)
    y_fake = torch.zeros(batch_size, 1).to(device)

    losses = []

    iteration = 0
    d_loss_all = 0.0
    g_loss = 0.0

    for x, labels in tqdm(train_loader):
        x = x.to(device)
        labels = labels.to(device)
        gen_labels = torch.tensor(np.random.randint(0, 10, batch_size)).to(device)

        # Train generator
        for _ in range(2):
            x_gen = generator(gen_labels)
            d_out_1, d_out_2 = discriminator(x_gen, gen_labels)

            g_loss_real = (criterion_1(d_out_1, y_real) + criterion_2(d_out_2, gen_labels.long())) * 0.5

            # Optimize according to the calculated loss
            g_optim.zero_grad()
            g_loss_real.backward()
            g_optim.step()

        # Train discriminator
        x_gen = generator(gen_labels)
        d_out_1_real, d_out_2_real = discriminator(x, labels)
        d_out_1, d_out_2 = discriminator(x_gen, gen_labels)

        d_loss_real = (criterion_1(d_out_1_real, y_real) + criterion_2(d_out_2_real, labels.long())) * 0.5
        d_loss_fake = (criterion_1(d_out_1, y_fake) + criterion_2(d_out_2, gen_labels.long() + 10)) * 0.5
        d_loss = (d_loss_fake + d_loss_real) * 0.5

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_loss_all += d_loss.item()
        g_loss = g_loss_real.item()

        iteration += 1

        if iteration % sample_interval == 0:
            d_loss_all = d_loss_all / sample_interval
            g_loss = g_loss / sample_interval

            print(f"D-Loss: {d_loss_all:.4f} G-Loss: {g_loss:.4f}")

            losses.append((d_loss_all, g_loss))
            d_loss = 0
            g_loss = 0

    return losses


def acgan_train(epochs=5):
    generator = Generator_acgan(10).to(device)
    if os.path.exists("weights/ac_generator_weights.pt"):
        print("using pretrained generator weights")
        generator.load_state_dict(torch.load("weights/ac_generator_weights.pt"))

    discriminator = Discriminator_acgan(20).to(device)
    if os.path.exists("weights/ac_discriminator_weights.pt"):
        print("using pretrained discriminator weights")
        discriminator.load_state_dict(torch.load("weights/ac_discriminator_weights.pt"))

    g_optim = optim.Adam(generator.parameters(), lr=0.001)
    d_optim = optim.Adam(discriminator.parameters(), lr=0.001)

    criterion_1 = nn.BCELoss()
    criterion_2 = nn.CrossEntropyLoss()
    _min = 0.018
    for epoch in range(epochs):
        print('epoch _ {}'.format(epoch))
        losses = acgan_train_real(64, 100, 100, generator, discriminator, criterion_1, criterion_2, g_optim, d_optim, _min)
        sample_digits(generator, epoch=epoch)
    torch.save(generator.state_dict(), "weights/ac_generator_weights.pt")
    torch.save(discriminator.state_dict(), "weights/ac_discriminator_weights.pt")


dis_gen_train()
