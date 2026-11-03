from torch.utils.data import DataLoader
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import lightning as L
import argparse
import pathlib
import torchvision

from opensr_model.autoencoder.autoencoder import AutoencoderKLLightning
from opensr_model.data.DOP20Dataset import DOP20Dataset, MultiSpectralDataset, ImageDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("load", type=pathlib.Path)
    config = OmegaConf.load("opensr_model/configs/config_dop20.yaml")
    args = parser.parse_args()

    reconstruction_loss = torch.nn.MSELoss()
    ddconfig = config.first_stage_config
    embed_dim = 4
    training_config = config.training_config
    device = training_config.device
    model = AutoencoderKLLightning.load_from_checkpoint(
        args.load, config=ddconfig, embed_dim=embed_dim, loss=torch.nn.MSELoss(), learning_rate=1e-4
    )

    dataset_config = config.dataset_config
    s2_dataset_config = dataset_config.lr_dataset
    s2_dataset = MultiSpectralDataset(**s2_dataset_config)

    dop20_dataset_config = dataset_config.hr_dataset
    dop20_dataset = ImageDataset(**dop20_dataset_config)

    dataset = DOP20Dataset(
        s2_dataset=s2_dataset, dop20_dataset=dop20_dataset, harmonizer_config=config.get("harmonizer", None)
    )

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    lr, hr = train_dataset[0]
    hr = hr.to(model.device)
    output, _ = model(hr.unsqueeze(0))
    output = output.clamp(0, 1).detach().cpu()

    images = torch.cat([hr.unsqueeze(0)[:4].detach().cpu(), output[:4]])
    image_grid = torchvision.utils.make_grid(images, nrow=output.size()[0])
    image = torchvision.transforms.ToPILImage()(image_grid)
    image.save("foo.png")
