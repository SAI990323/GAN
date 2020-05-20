import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio

device = torch.device("cuda")


def draw(epoch, gen_data, test_data, background, background_color, x_min, x_max, y_min, y_max, net_name):
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("Show")
    print(gen_data[0])
    bg = plt.scatter(background[:, 0], background[:, 1], c=np.squeeze(background_color), cmap=plt.cm.get_cmap('gray'))
    colorbar = plt.colorbar(bg)
    plt.scatter(gen_data[:,0], gen_data[:,1], c = 'r', s=10)
    plt.scatter(test_data[:,0], test_data[:,1], c = 'b' , s = 10)
    plt.savefig("./result//"+ net_name+ "//epoch" + str(epoch + 1))
    colorbar.remove()
    plt.cla()

def merge(len, net_name):
    imgs = []
    for i in range(1,int(len) + 1):
        imgs.append(imageio.imread("./result//"+net_name+ "//epoch" + str(i * 10) + ".png"))
    imageio.mimsave("./result//"+net_name+ "//" + net_name + ".gif", imgs, fps=3)
