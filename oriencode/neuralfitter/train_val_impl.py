import torch
import time
from typing import Union
import torch.nn as nn
from tqdm import tqdm
from collections import namedtuple

from .utils import log_train_val_progress
from ..evaluation.utils import MetricMeter


def train(baseline_model, generator, discriminator, optimizer_g, optimizer_d, loss, loss2, dataloader, grad_rescale, grad_mod, epoch, device, logger) -> float:

    """Some Setup things"""
    generator.train()
    discriminator.train()
    tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.)  # progress bar enumeration
    t0 = time.time()
    loss_epoch_g = MetricMeter()  # Generators loss meter
    loss_epoch_d = MetricMeter()  # Discriminators loss meter

    """Actual Training"""
    for batch_num, (x_em, x, y_tar, weight) in enumerate(tqdm_enum):  # model input (x), target (yt), weights (w)

        """Monitor time to get the data"""
        t_data = time.time() - t0

        """Ship the data to the correct device"""
        x_em, x, y_tar, weight = ship_device([x_em, x, y_tar, weight], device)
        
        with torch.no_grad():
                _, baseline_features = baseline_model(x_em)
                
        optimizer_d.zero_grad()
        real_labels = torch.ones(baseline_features.size(0), 1).to(device)
        d_loss_real = loss2(discriminator(baseline_features), real_labels)

        y_out, generator_features = generator(x)
        fake_labels = torch.zeros(generator_features.size(0), 1).to(device)
        d_loss_fake = loss2(discriminator(generator_features.detach()), fake_labels)
        #print(discriminator(generator_features.detach()).shape, fake_labels.shape)
        d_loss = 10 * (d_loss_real + d_loss_fake)
        d_loss.mean().backward()
        optimizer_d.step()
        #print(d_loss.shape)
        loss_epoch_d.update(d_loss.mean().item())
        
        #生成器
        optimizer_g.zero_grad()
        
        g_labels = torch.ones(generator_features.size(0), 1).to(device)
        g_loss_gan = 10 * loss2(discriminator(generator_features), g_labels)
        
        loss_od = loss(y_out, y_tar, weight)
        
        g_loss = loss_od.clone()  
        g_loss[:, 0] +=  g_loss_gan.squeeze()  
        print(d_loss.mean())
        print(g_loss_gan.mean())
        if grad_rescale:  
            weight, _, _ = generator.rescale_last_layer_grad(g_loss, optimizer_g)
            g_loss = g_loss * weight
            
        g_loss.mean().backward()
        if grad_mod:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.03, norm_type=2)
        optimizer_g.step()
        print(g_loss.shape)
        loss_epoch_g.update(g_loss.mean().item())
        
        t_batch = time.time() - t0

        loss_mean_gan, _ = loss.log(g_loss_gan) 
        loss_mean_g, loss_cmp_g = loss.log(g_loss)  
        loss_mean_d, loss_cmp_d = loss.log(d_loss)

        del g_loss, d_loss

        tqdm_enum.set_description(f"E: {epoch} - t: {t_batch:.2} - D_L: {loss_mean_d:.3} - G_gan: {loss_mean_gan:.3} - G_L: {loss_mean_g:.3}")

        t0 = time.time()

    log_train_val_progress.log_train(loss_p_batch=loss_epoch_g.vals, loss_mean=loss_epoch_g.mean, logger=logger, step=epoch)
    #log_train_val_progress.log_train(loss_p_batch=loss_epoch_d.vals, loss_mean=loss_epoch_d.mean, logger=logger, step=epoch, name='Discriminator')
    return loss_epoch_g.mean


_val_return = namedtuple("network_output", ["loss", "x", "y_out", "y_tar", "weight", "em_tar"])


def test(baseline_model, generator, discriminator, loss, loss2, dataloader, epoch, device):

    """Setup"""
    x_ep, y_out_ep, y_tar_ep, weight_ep, em_tar_ep = [], [], [], [], []  # store things epoche wise (_ep)
    loss_cmp_ep = []

    generator.eval()
    discriminator.eval()
    tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.)  # progress bar enumeration

    t0 = time.time()

    """Testing"""
    with torch.no_grad():
        for batch_num, (x_em, x, y_tar, weight) in enumerate(tqdm_enum):

            """Ship the data to the correct device"""
            x_em, x, y_tar, weight = ship_device([x_em, x, y_tar, weight], device)

            """
            Forward the data.
            """
            _, baseline_features = baseline_model(x_em)

            y_out, generator_features = generator(x)

            # Loss for discriminator
            real_labels = torch.ones(baseline_features.size(0), 1).to(device)
            fake_labels = torch.zeros(generator_features.size(0), 1).to(device)
            #print(discriminator(baseline_features).shape, real_labels.shape)
            d_loss_real = loss2(discriminator(baseline_features), real_labels)
            d_loss_fake = loss2(discriminator(generator_features), fake_labels)
            d_loss = d_loss_real + d_loss_fake

            # Loss for generator (GAN loss)
            g_labels = torch.ones(generator_features.size(0), 1).to(device)
            g_loss_gan = loss2(discriminator(generator_features), g_labels)

            # Main loss for the generator
            loss_od = loss(y_out, y_tar, weight)
            g_loss = loss_od + 0.5 * g_loss_gan

            t_batch = time.time() - t0

            """Logging and temporary save"""
            tqdm_enum.set_description(f"(Test) E: {epoch} - T: {t_batch:.2}")

            loss_cmp_ep.append(g_loss.detach().cpu())
            x_ep.append(x.cpu())
            y_out_ep.append(y_out.detach().cpu())

    """Epoch-Wise Merging"""
    loss_cmp_ep = torch.cat(loss_cmp_ep, 0)
    x_ep = torch.cat(x_ep, 0)
    y_out_ep = torch.cat(y_out_ep, 0)

    return loss_cmp_ep.mean(), _val_return(loss=loss_cmp_ep, x=x_ep, y_out=y_out_ep, y_tar=None, weight=None, em_tar=None)


def ship_device(x, device: Union[str, torch.device]):
    """
    Ships the input to a pytorch compatible device (e.g. CUDA)

    Args:
        x:
        device:

    Returns:
        x

    """
    if x is None:
        return x

    elif isinstance(x, torch.Tensor):
        return x.to(device)

    elif isinstance(x, (tuple, list)):
        x = [ship_device(x_el, device) for x_el in x]  # a nice little recursion that worked at the first try
        return x

    elif device != 'cpu':
        raise NotImplementedError(f"Unsupported data type for shipping from host to CUDA device.")

def check_data(x):
    if isinstance(x, torch.Tensor):
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input tensor contains NaN or infinite values")
    elif isinstance(x, (list, tuple)):
        for item in x:
            check_data(item)
    elif x is not None:
        raise ValueError("Unsupported data type for checking NaN or infinite values")


