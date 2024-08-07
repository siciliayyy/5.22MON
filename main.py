import torch
import yaml

from dataset.prepare_dataset import prepare_dataloader
from evaluate import evaluate
from model.DenseUNet3d import DenseUNet3d
from model.vnet import VNet
from training.train import train


def main():
    with open("./config.yaml", "r") as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)

    if torch.cuda.is_available() and config["gpu"]["use_gpu"]:
        device = torch.device(config["gpu"]["gpu_name"])
    else:
        device = torch.device("cpu")

    trainloader, n_train = prepare_dataloader(config, train=0)
    valdataloader, n_val = prepare_dataloader(config, train=2)

    if config["model"] == 'DenseUNet3d':  # 输入0为VNET()
        model = DenseUNet3d()
    else:
        model = VNet()

    if config["pathing"]["pretrain"] != 0:
        model.to(device=device)
        argum = torch.load(config["pathing"]["pretrain"])
        model.load_state_dict(argum, True)


    train(config, model, device, trainloader, valdataloader, n_train, n_val)

    testloader = prepare_dataloader(config, train=0)
    scores = evaluate(model, device, testloader)
    print(scores)


if __name__ == "__main__":
    main()