from torch.utils.data import DataLoader
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import lightning as L
from lightning.pytorch import loggers as pl_loggers
import argparse
import pathlib
import torchvision
import friendlywords
from opensr_model.autoencoder.autoencoder import AutoencoderKLLightning
from opensr_model.data.DOP20Dataset import DOP20Dataset, MultiSpectralDataset, ImageDataset
from omegaconf import DictConfig, ListConfig
from omegaconf.nodes import AnyNode
from omegaconf.base import ContainerMetadata
from typing import Any

torch.serialization.add_safe_globals([DictConfig, ListConfig, AnyNode, ContainerMetadata, Any, dict])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--load", type=pathlib.Path)
    parser.add_argument("-c", "--config", type=pathlib.Path, default="opensr_model/configs/config_dop20.yaml")
    args = parser.parse_args()
    log_name = pathlib.Path(str(args.config.name).split(".")[0] + "_" + friendlywords.generate(2, separator="-"))

    config = OmegaConf.load(args.config)
    ddconfig = config.first_stage_config
    embed_dim = 4
    training_config = config.training_config
    device = training_config.device
    seed = training_config.seed

    L.seed_everything(seed)

    reconstruction_loss = torch.nn.L1Loss()
    model = AutoencoderKLLightning(
        ddconfig, embed_dim, loss=reconstruction_loss, learning_rate=training_config.learning_rate
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

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=training_config.batch_size,
        shuffle=training_config.shuffle,
        pin_memory=training_config.pin_memory,
        num_workers=training_config.num_workers,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        pin_memory=training_config.pin_memory,
        num_workers=training_config.num_workers,
    )
    print(f"Logging data to: {log_name}")
    logger = pl_loggers.TensorBoardLogger(save_dir="logs/", name=log_name, version="")
    trainer = L.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=device,
        max_epochs=training_config.epochs,
        strategy="ddp_find_unused_parameters_true",
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.load,
        weights_only=False,
    )

    lr, hr = train_dataset[0]
    output, _ = model(hr.unsqueeze(0))
    output = output.clamp(0, 1).detach().cpu()
    print(output[0].shape, hr.shape)
    image = torchvision.transforms.ToPILImage()(output[0])
    image.save("foo.png")
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)

    for lr, hr in tqdm(dataloader):
        hr = hr.to(device)
        output, posterior = model(hr)
        rec_loss = reconstruction_loss(output, hr)
        kl_loss = posterior.kl().mean()
        loss = rec_loss + kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item()
    """
