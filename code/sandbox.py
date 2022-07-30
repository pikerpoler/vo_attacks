#this file is for experiments like checking tensor sizes and stuff
import torch
from torch.nn import ConvTranspose2d, Sequential


class PertGenerator(torch.nn.Module):
    def __init__(self, pert_shape, kernel_size=4):
        super(PertGenerator, self).__init__()
        self.pert_shape = pert_shape
        self.kernel_size = kernel_size
        self.kernel = torch.ones(1, 1, kernel_size, kernel_size)
        c1 = ConvTranspose2d(1, 1, kernel_size=(6, 6), stride=(4, 4), padding=2)
        c2 = ConvTranspose2d(1, 2, kernel_size=(13, 19), stride=(5, 5), padding=2)
        c3 = ConvTranspose2d(2, 3, kernel_size=(14, 12), stride=(6, 8), padding=2)

        self.cnn = Sequential(c1, c2, c3)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp



def experiment_conv_sizes():
    #### desired shape : [1, 3, 448, 640]

    # kernel = torch.ones(1, 1, 224, 320)
    # c1 = ConvTranspose2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=2)
    # c2 = ConvTranspose2d(1, 2, kernel_size=(4, 4), stride=(1, 1), padding=1)
    # c3 = ConvTranspose2d(2, 3, kernel_size=(6, 6), stride=(2, 2), padding=1)
    # model = Sequential(c1, c2, c3)


    kernel = torch.ones(1,1,1,1)
    # c1 = ConvTranspose2d(1, 2, kernel_size=(7, 10), stride=(1, 1), padding=(0, 1))
    # c2 = ConvTranspose2d(2, 3, kernel_size=(14, 25), stride=(2, 2), padding=(1, 2))
    # c3 = ConvTranspose2d(3, 4, kernel_size=(35, 50), stride=(3, 3), padding=(4, 7))
    # c4 = ConvTranspose2d(4, 3, kernel_size=(70, 100), stride=(4, 4), padding=(1, 4))
    # model = Sequential(c1, c2, c3, c4)
    c1 = ConvTranspose2d(1, 3, kernel_size=(448, 640), stride=(1, 1), padding=(0, 0))
    model = Sequential(c1)

    print(f'{get_n_params(model)} <? {3 * 448 * 640}')
    print(model(kernel).shape)

def experiment_num_params():
    kernel = torch.tensor([[[[1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 1, 1, 1]]]], dtype=torch.float32)
    print(kernel.shape)
    model = PertGenerator((1, 3, 448, 640))
    print(get_n_params(model))
    print(3 * 448 * 640)

def main():
    experiment_conv_sizes()

if __name__ == '__main__':
    main()