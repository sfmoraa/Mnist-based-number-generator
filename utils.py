import torch
import matplotlib.pyplot as plt
import math
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

""" 
函数功能:   输入（n，1，p，q）的tensor，在一张画布上输出对应的灰度图像并保存为图片
函数参数:    tensor_to_img(torch.Tensor)           : 输入的n*1*p*q的tensor
            save（Optional, Default is True）     : 是否保存为图片
            path（Optional, Default is "output"） : 保存的图片名称   注：图片保存在exmaple文件夹下
"""
def tensor_to_img(x: torch.Tensor, save=True, path="output"):
    x_shape = x.shape
    dim = len(x_shape)
    if dim == 3:
        x = x.squeeze(dim=0)
    if dim == 2:
        line = row = 1
    else:
        row = int(math.sqrt(x_shape[0]))
        line = math.ceil(x_shape[0] / row)
    x = x.cpu().detach().numpy()
    _, axs = plt.subplots(row, line, figsize=(line, row), sharey=True, sharex=True)

    if row == 1:
        axs.imshow(x.squeeze(), cmap='gray')
        axs.axis('off')
    else:
        cnt = 0
        for i in range(row):
            for j in range(line):
                axs[i, j].axis('off')
                if i * line + j >= x_shape[0]:
                    continue
                axs[i, j].imshow(x[cnt][0], cmap='gray')
                cnt += 1
    if not os.path.exists("example"):
        os.makedirs("example")
    if save:
        plt.savefig("example/{}.png".format(path))
    return axs



"""
函数功能:生成1-10的样例图像
函数参数:   generator 生成器
          image_grid_rows（Optional, Default is 6）       :  行数，默认输出6行1-10
          image_grid_columns（Optional, Default is 10）   :  列数，表示1-10十个数字
          epoch                                          :   输出的图片保存在example/output_epoch.png中

"""
def sample_digits(generator, image_grid_rows=6, image_grid_columns=10, epoch=0):
    with torch.no_grad():
        generator.eval()
        gen_imgs = generator(torch.tensor([i for _ in range(image_grid_rows) for i in range(0, 10)], device=device))
        gen_imgs = gen_imgs.cpu().detach().view(-1, 28, 28).numpy()
        generator.train()

    gen_imgs = 0.5 * gen_imgs + 0.5

    _, axs = plt.subplots(image_grid_rows,
                          image_grid_columns,
                          figsize=(10, 6),
                          sharey=True,
                          sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, :, :], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    if not os.path.exists("example"):
        os.makedirs("example")
    plt.savefig("example/output_{}.png".format(epoch))


# data1和data2分别是两个想要比对准确率的numpy类型数组，可以在aigcmn.py的generate函数中加入
#   accuracy1 = torch.sigmoid(self.cf(gen_img))[index, label]
#   accuracy1 = accuracy1.detach().numpy()
#   得到accuracy1即为data1
def plot_accuracy(data1, data2):
    index = range(len(data1))
    plt.figure(figsize=(15, 5))
    plt.plot(index, data1, label='before optim')
    plt.plot(index, data2, label='after optim')
    plt.ylabel("accuracy")
    plt.xlabel("index")
    plt.legend()
    plt.show()
