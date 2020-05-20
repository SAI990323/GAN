import numpy as np
import torch
import torch.nn as nn
from torch import autograd

import load
import draw
import argparse
from tensorboardX import SummaryWriter

device = torch.device("cuda")
class gen(nn.Module):
    def __init__(self, input_size):
        super(gen,self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 2)
        )

    def forward(self, input):
        input = self.gen(input)
        return input

class dis(nn.Module):
    def __init__(self):
        super(dis, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(2,256),
            nn.ReLU(inplace=False),
            nn.Linear(256,64),
            nn.ReLU(inplace=False),
            nn.Linear(64,1),
        )

    def forward(self, input):
        input = self.dis(input)
        return input



train_set, test_set = load.get_data()



def train(epoch = 100, batch_size = 100, input_size = 2):
    GEN = gen(input_size).to(device).double()
    DIS = dis().to(device).double()
    optimizer_g = torch.optim.RMSprop(GEN.parameters(), lr=0.0001)
    optimizer_d = torch.optim.RMSprop(DIS.parameters(), lr=0.0001)
    for ep in range(epoch):
        total_gen_loss = 0
        total_dis_loss = 0
        for j in range(int(len(train_set) / batch_size)):
            func = torch.from_numpy(train_set[j * batch_size: (j + 1) * batch_size]).double().to(device)
            input = torch.randn(batch_size, input_size).to(device).double()
            noise = GEN(input)

            for p in DIS.parameters():
                p.data.clamp_(-0.1,0.1)

            GEN_loss = -torch.mean(DIS(noise)).to(device)
            optimizer_g.zero_grad()
            GEN.zero_grad()
            GEN_loss.backward()
            optimizer_g.step()

            DIS_loss = -torch.mean(DIS(func) - DIS(noise.detach())).to(device)
            optimizer_d.zero_grad()
            DIS.zero_grad()
            DIS_loss.backward()
            optimizer_d.step()

            total_gen_loss += GEN_loss.item() * batch_size
            total_dis_loss += DIS_loss.item() * batch_size
        print("epoch %d \t G_LOSS: %.4f \t D_LOSS: %.4f" % (ep + 1, total_gen_loss / len(train_set) , total_dis_loss / len(train_set)))

        if ep % 10 == 9 :
            with torch.no_grad():
                input = torch.randn(1000, input_size).to(device).double()
                output = GEN(input)
                data = np.array(output.cpu().data)
                x_min = np.min(test_set[:, 0])
                x_max = np.max(test_set[:, 0])
                y_min = np.min(test_set[:, 1])
                y_max = np.max(test_set[:, 1])
                x_min = np.min(np.append(data[:, 0], x_min))
                x_max = np.max(np.append(data[:, 0], x_max))
                y_min = np.min(np.append(data[:, 1], y_min))
                y_max = np.max(np.append(data[:, 1], y_max))
                i = x_min
                background = []
                while i <= x_max:
                    j = y_min
                    while j <= y_max:
                        background.append([i,j])
                        j = j + 0.01
                    background.append([i, y_max])
                    i = i + 0.01
                print(x_min,x_max,y_min,y_max)
                background = np.array(background)
                background_color = DIS(torch.Tensor(background).to(device).double())
                print(background_color.size())
                print(np.shape(background))
                draw.draw(ep, data,test_set, background , np.array(background_color.cpu().data), x_min, x_max, y_min, y_max, "wgan_rsp")
                GEN.zero_grad()
                DIS.zero_grad()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", dest="epoch", default=200, type=int)
    args = parser.parse_args()
    train(epoch=args.epoch)
    draw.merge(args.epoch / 10, "wgan_rsp")