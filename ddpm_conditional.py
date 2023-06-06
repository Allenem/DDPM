import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        '''
        Generate noise beta of each time step
        return: shape (noise_steps, )
        '''
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        '''
        Sample time steps for each image
        input:
            n: batch_size, int
        return:
            t: time_step, shape (n, ), values in [1, noise_step]
        '''
        return torch.randint(low=1, high=self.noise_steps, size=(n, ))

    def noise_images(self, x, t):
        '''
        Add noise process: x_0 -> x_{t}
        input:
            x: input_images, shape (batch_size, 1, img_size, img_size)
            t: time_step, int
        return:
            noise_images: shape (batch_size, 1, img_size, img_size)
            noise: shape (batch_size, 1, img_size, img_size)
            noise_images = sqrt(alpha_hat[t]) * x + sqrt(1 - alpha_hat[t]) * noise
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample(self, model, n, labels, cfg_scale=3):
        '''
        Denoise process: x_{t} -> x_{t-1} -> ... -> x_0
        input:
            model: nn.Module
            n: batch_size, int
            labels: shape (n, ), values in [0, 9]
            cfg_scale: float, 0.0 ~ 1.0, 0.0: unconditioned diffusion, 1.0: conditioned diffusion
        return:
            x_0: images in t0, shape (n, 1, img_size, img_size), values in [0, 255]
            sampled_images (x_{t-1}) = 1 / sqrt(alpha[t]) * (noisy_images (x_t) - (1 - alpha[t]) / sqrt(1 - alpha_hat[t]) * predicted_noise) + sqrt(beta[t]) * noise
        '''
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                # interpolate with unconditioned diffusion
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    # predicted_noise = predicted_noise * cfg_scale + uncond_predicted_noise * (1 - cfg_scale)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    '''
    Training process: 
    each epoch, each batch, rand_time_step, compare the predicted noise with the real noise
    '''
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}/{args.epochs}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            # 10% of the time, don't pass labels, unconditioned diffusion
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            # plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"), nrow=len(labels))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"), nrow=len(labels))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 300
    args.batch_size = 2
    args.image_size = 64
    args.num_classes = 10
    # args.dataset_path = "C:/Users/dome/datasets/cifar10/cifar10-64/train"
    args.dataset_path = "C:/Users/siat/Downloads/archive/cifar10-64/train"
    args.dataset_path = "D:/Download/Browser/archive/cifar10-64/train"
    args.train = True
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)

