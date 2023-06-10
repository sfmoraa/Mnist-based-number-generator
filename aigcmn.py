import torch
import wget
import os
from module import Generator, Classifier, Generator_num, Generator_acgan
from utils import tensor_to_img

device = 'cpu'
MIN_SCORE = 0.85
MAX_ITER = 20

WEIGHTS_URL = "https://raw.githubusercontent.com/heatingma/MNIST_GENERATE/main/{}"


class AiGcMn():
    def __init__(self):
        self.gen = Generator(10).to(device)         # 第一种生成模型
        self.gen_num = Generator_num().to(device)   # 第二种生成模型
        self.acgen = Generator_acgan(10).to(device) # 第三种生成模型
        self.cf = Classifier(11).to(device)         # 分类器
        self.load_weights()                         # 加载预训练参数

    def load_weights(self):
        if not os.path.exists("weights"):
            os.makedirs("weights")

        gen_weights_path = "weights/generator_weights.pt"
        if not os.path.exists(gen_weights_path):
            print("downloading module1 weights......")
            wget.download(WEIGHTS_URL.format(gen_weights_path), gen_weights_path)
        self.gen.load_state_dict(torch.load(gen_weights_path))

        cf_weights_path = "weights/classifier_weights.pt"
        if not os.path.exists(cf_weights_path):
            print("downloading other weights......")
            wget.download(WEIGHTS_URL.format(cf_weights_path), cf_weights_path)
        self.cf.load_state_dict(torch.load(cf_weights_path))

    def generate(self, label: torch.Tensor, retrain=True, mode="all", show=False, pretrain="1"):
        if label.shape[1] > 1:
            label = label.squeeze()

        if pretrain == "1":
            return self.generate1(label, retrain, mode, show)
        elif pretrain == "2":
            return self.generate2(label, retrain, mode, show)
        elif pretrain == "3":
            return self.generate3(label, retrain, mode, show)

    def generate1(self, label: torch.Tensor, retrain=True, mode="all", show=False):
        label = label.int().to(device)
        gen_img = self.gen(label)

        if show:
            tensor_to_img(gen_img, path="1_before_optimize")
        if retrain:
            if mode == 'part':
                retrain_list = self.get_retrain(gen_img, label)
                gen_img = self.optimize(gen_img, retrain_list, self.gen)
                if show:
                    tensor_to_img(gen_img, path="1_after_part_optimize")
            elif mode == 'all':
                retrain_list = self.all_retrain(label)
                gen_img = self.optimize(gen_img, retrain_list, self.gen)
                if show:
                    tensor_to_img(gen_img, path="1_after_all_optimize")
            else:
                raise TypeError(mode + " is not supported!" + " available mode : all / part")
        return gen_img

    def generate2(self, label: torch.Tensor, retrain=True, mode="all", show=False):
        label = label.int().to(device)
        gen_img = self.gen_num(len(label))

        index = dict()
        for i in range(10):
            index[str(i)] = list()
        for i in range(len(label)):
            index[str(label[i].item())].append(i)

        for i in range(10):
            cur_index = index[str(i)]
            length = len(cur_index)
            if length == 0:
                continue
            gen_weights_path = "weights/GEN_{}.pt".format(i)
            if not os.path.exists(gen_weights_path):
                print("downloading module2 weights......")
                wget.download(WEIGHTS_URL.format(gen_weights_path), gen_weights_path)
            self.gen_num.load_state_dict(torch.load(gen_weights_path))
            for j in range(length):
                cur_iter = 0
                while (cur_iter < MAX_ITER):
                    img = self.gen_num(20)
                    score = self.cf(img)[:, i]
                    max_score_index = torch.argmax(score)
                    if score[max_score_index] > MIN_SCORE:
                        gen_img[cur_index[j]] = img[max_score_index]
                        break

        if show:
            tensor_to_img(gen_img, path="2_result")

        return gen_img

    def generate3(self, label: torch.Tensor, retrain=True, mode="all", show=False):

        gen_weights_path = "weights/ac_generator_weights.pt"
        if not os.path.exists(gen_weights_path):
            print("downloading other weights......")
            wget.download(WEIGHTS_URL.format(gen_weights_path), gen_weights_path)
        self.acgen.load_state_dict(torch.load(gen_weights_path))

        label = label.int().to(device)
        gen_img = self.acgen(label)

        if show:
            tensor_to_img(gen_img, path="3_before_optimize")
        if retrain:
            if mode == 'part':
                retrain_list = self.get_retrain(gen_img, label)
                gen_img = self.optimize(gen_img, retrain_list, self.acgen)
                if show:
                    tensor_to_img(gen_img, path="3_after_part_optimize")
            elif mode == 'all':
                retrain_list = self.all_retrain(label)
                gen_img = self.optimize(gen_img, retrain_list, self.acgen)
                if show:
                    tensor_to_img(gen_img, path="3_after_all_optimize")
            else:
                raise TypeError(mode + " is not supported!" + " available mode : all / part")
        return gen_img

    def get_retrain(self, gen_img, label):
        cf_score = torch.sigmoid(self.cf(gen_img))
        # cf_score = torch.softmax(self.cf(gen_img),dim=1)
        retrain = dict()
        for i in range(10):
            retrain[str(i)] = list()
        for i in range(len(label)):
            if cf_score[i][label[i]] < MIN_SCORE:
                retrain[str(label[i].item())].append(i)
        return retrain

    def all_retrain(self, label):
        retrain = dict()
        for i in range(10):
            retrain[str(i)] = list()
        for i in range(len(label)):
            retrain[str(label[i].item())].append(i)
        return retrain

    def optimize(self, gen_img, retrain, gen):
        for i in range(10):
            cur_retrain = retrain[str(i)]
            length = len(cur_retrain)
            if length == 0:
                continue
            for j in range(length):
                cur_iter = 0
                while (cur_iter < MAX_ITER):
                    cur_iter += 1
                    new_imgs = gen(torch.ones(20, dtype=int, device=device) * i)
                    score = self.cf(new_imgs)[:, i]
                    score_max_index = torch.argmax(score)
                    gen_img[cur_retrain[j]] = new_imgs[score_max_index]
                    if score[score_max_index] > MIN_SCORE:
                        break
        return gen_img
