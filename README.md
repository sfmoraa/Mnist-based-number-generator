说明文档

**train_func.py** 此文件用于训练模型参数

**module.py** 此文件是我们的神经网络模型

**utils.py** 此文件是一些函数，主要用于获得一些输出图像

**aigcmn.py** 此文件中的接口类AiGcMn实现了输入n维tensor（n是batch的大小，每个整数在0~9范围内，代表需要生成的数字），输出`n*1*28*28`的tensor

**AiGcMn** 生成函数generate除整数型n维tensor参数label外可供选择的参数有三个：
* mode：设为‘all’时对全部生成结果进行优化，设为‘part’时对生成结果进行部分优化，默认为'all'
* show：设为True时将生成结果转为可视化图片存于example文件夹内，默认为False
* pretrain：可取的值有'1'、'2'、'3'，分别对应不同的训练模型，默认为‘1’

**注：**
optimize是优化函数，可以使输出的tensor更接近MNIST数据集

data文件夹 是MNIST原始数据集

weights文件夹 存放了预训练参数

example文件夹 存放了我们训练时获得的一些图片信息

test.py文件为使用样例，依次在不同优化模式下使用我们预训练的三种模型进行输出结果对比，因均选择将结果转换为图片，故运行时间相对稍长，实际调用时使用默认参数设置即可达到最优效果。


```python
from aigcmn import AiGcMn
import torch

ai = AiGcMn()
label = torch.randint(low=0, high=10, size=(1, 100))
for i in range((label.shape[1] + 9) // 10):
    print(label[0][i * 10:i * 10 + 10])

mode = ['all', 'part']
pretrain = ['1', '2', '3']

# 不同模型及优化方式输出图片对比
for i in range(3):
    for j in range(2):
        ai.generate(label, mode=mode[j], show=True, pretrain=pretrain[i])

# 常规调用
ai.generate(label)
```
在本实例中，模型1、3根据优化程度会分别输出三张图片，即不优化、部分优化、全部优化；而模型2在生成时已经进行了优化处理，故模式选择对其无影响。