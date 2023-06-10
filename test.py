from aigcmn import AiGcMn
import torch

ai = AiGcMn()
label = torch.randint(low=0, high=10, size=(1, 100))
for i in range((label.shape[1] + 9) // 10):
    print(label[0][i * 10:i * 10 + 10])

mode = ['all', 'part']
pretrain = ['1', '2', '3']

# 所有可行的模型及优化方式输出图片对比，共输出7张图片，总共需运行1分钟左右
for i in range(3):
    for j in range(2):
        ai.generate(label, mode=mode[j], show=True, pretrain=pretrain[i])

# 常规调用，速度较快
ai.generate(label)
