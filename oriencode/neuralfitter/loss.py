from abc import ABC, abstractmethod  # abstract class
from typing import Union, Tuple

import torch
from deprecated import deprecated
from torch import distributions

# from . import MixtureSameFamily as mixture
from ..simulation import psf_kernel


class Loss(ABC):
    """Abstract class for my loss functions."""

    def __init__(self):
        super().__init__()

    def __call__(self, output, target, weight):
        """
        calls functional
        """
        return self.forward(output, target, weight)

    @abstractmethod
    def log(self, loss_val):
        """

        Args:
            loss_val:

        Returns:
            float: single scalar that is subject to the backprop algorithm
            dict:  dictionary with values being floats, describing additional information (e.g. loss components)
        """
        raise NotImplementedError

    def _forward_checks(self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
        """
        Some sanity checks for forward data

        Args:
            output:
            target:
            weight:

        """
        if not (output.size() == target.size() and target.size() == weight.size()):
            raise ValueError(f"Dimensions of output, target and weight do not match "
                             f"({output.size(), target.size(), weight.size()}.")

    @abstractmethod
    def forward(self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss term

        Args:
            output (torch.Tensor): output of the network
            target (torch.Tensor): target data
            weight (torch.Tensor): px-wise weight map

        Returns:
            torch.Tensor

        """
        raise NotImplementedError


class PPXYZBLoss(Loss):
    """
    Loss implementation for 6 channel output for SMLM data, where the channels are

        0: probabilities (without sigmoid)
        1: photon count
        2: x pointers
        3: y pointers
        4: z pointers
        5: background
    """

    def __init__(self, device: Union[str, torch.device], chweight_stat: Union[None, tuple, list, torch.Tensor] = None,
                 p_fg_weight: float = 1., forward_safety: bool = True):
        """

        Args:
            device: device in forward method (e.g. 'cuda', 'cpu', 'cuda:0')
            chweight_stat: static channel weight
            p_fg_weight: foreground weight
            forward_safety: check sanity of forward arguments
        """

        super().__init__()
        self.forward_safety = forward_safety

        if chweight_stat is not None:
            self._ch_weight = chweight_stat if isinstance(chweight_stat, torch.Tensor) else torch.Tensor(chweight_stat)
        else:
            self._ch_weight = torch.tensor([1., 1., 1., 1., 1., 1.])
        self._ch_weight = self._ch_weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)

        self._p_loss = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(p_fg_weight).to(device))
        self._phot_xyzbg_loss = torch.nn.MSELoss(reduction='none')

    def log(self, loss_val) -> (float, dict):
        loss_vec = loss_val.mean(-1).mean(-1).mean(0)
        return loss_vec.mean().item(), {
            'p': loss_vec[0].item(),
            'phot': loss_vec[1].item(),
            'x': loss_vec[2].item(),
            'y': loss_vec[3].item(),
            'z': loss_vec[4].item(),
            'bg': loss_vec[5].item()
        }

    def _forward_checks(self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
        super()._forward_checks(output, target, weight)

        if output.size(1) != 6:
            raise ValueError("Not supported number of channels for this loss function.")

    def forward(self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:

        if self.forward_safety:
            self._forward_checks(output, target, weight)

        ploss = self._p_loss(output[:, [0]], target[:, [0]])
        chloss = self._phot_xyzbg_loss(output[:, 1:], target[:, 1:])
        tot_loss = torch.cat((ploss, chloss), 1)

        tot_loss = tot_loss * weight * self._ch_weight

        return tot_loss


class GaussianMMLoss(Loss):
    """
    Model output is a mean and sigma value which forms a gaussian mixture model.
    """

    def __init__(self, *, xextent: tuple, yextent: tuple, img_shape: tuple, device: Union[str, torch.device],
                 chweight_stat: Union[None, tuple, list, torch.Tensor] = None, extra_weight = None,
                 forward_safety: bool = True):
        """

        Args:
            xextent: extent in x
            yextent: extent in y
            img_shape: image size
            device: device used in training (cuda / cpu)
            chweight_stat: static channel weight, mainly to disable background prediction
            forward_safety: check inputs to the forward method
        """
        super().__init__()

        if chweight_stat is not None:
            self._ch_weight = chweight_stat if isinstance(chweight_stat, torch.Tensor) else torch.Tensor(chweight_stat)
        else:
            self._ch_weight = torch.ones(2)
        self._ch_weight = self._ch_weight.reshape(1, 2).to(device)
        self.extra_weight = extra_weight if extra_weight is not None else 1.0

        self._bg_loss = torch.nn.MSELoss(reduction='none')
        self._offset2coord = psf_kernel.DeltaPSF(xextent=xextent, yextent=yextent, img_shape=img_shape)
        self.forward_safety = forward_safety
        self.weight1 = torch.nn.Parameter(torch.tensor(0.5))
        self.weight2 = torch.nn.Parameter(torch.tensor(0.4))
        self.weight3 = torch.nn.Parameter(torch.tensor(0.1))

    def log(self, loss_val):
        mean_loss = loss_val.mean().item()  # 计算所有值的平均数
        result = {'gmm': loss_val[:, 0].mean().item()}  # 始终计算 'gmm'

        # 根据列数判断是否需要添加 'bg'
        if loss_val.shape[1] == 2:
            result['bg'] = loss_val[:, 1].mean().item()  # 只在有第二列时计算 'bg'

        return mean_loss, result

    @staticmethod
    def _format_model_output(output: torch.Tensor) -> tuple:
        """
        Transforms solely channel based model output into more meaningful variables.

        Args:
            output: model output

        Returns:
            tuple containing
                p: N x H x W
                pxyz_mu: N x 4 x H x W
                pxyz_sig: N x 4 x H x W
                bg: N x H x W
        """
        p = output[:, 0]
        pxyz_mu = output[:, 1:5]
        pxyz_sig = output[:, 5:9]
        #crlb_sig = output[:, 9:-1]
        bg = output[:, -1]

        return p, pxyz_mu, pxyz_sig, bg #, crlb_sig

    def _compute_gmm_loss(self, p, pxyz_mu, pxyz_sig, pxyz_tar, mask) -> torch.Tensor:
        """
        Computes the Gaussian Mixture Loss.

        Args:
            p: the model's detection prediction (sigmoid already applied) size N x H x W
            pxyz_mu: prediction of parameters (phot, xyz) size N x C=4 x H x W
            pxyz_sig: prediction of uncertainties / sigma values (phot, xyz) size N x C=4 x H x W
            pxyz_tar: ground truth values (phot, xyz) size N x M x 4 (M being max number of tars)
            mask: activation mask of ground truth values (phot, xyz) size N x M

        Returns:
            torch.Tensor (size N x 1)

        """

        batch_size = pxyz_mu.size(0)
        log_prob = 0

        p_mean = p.sum(-1).sum(-1)
        p_var = (p - p ** 2).sum(-1).sum(-1)  # var estimate of bernoulli
        p_gauss = distributions.Normal(p_mean, torch.sqrt(p_var))

        log_prob = log_prob + p_gauss.log_prob(mask.sum(-1)) * mask.sum(-1)

        prob_normed = p / p.sum(-1).sum(-1).view(-1, 1, 1)

        """Hacky way to get all prob indices"""
        p_inds = tuple((p + 1).nonzero(as_tuple=False).transpose(1, 0))
        pxyz_mu = pxyz_mu[p_inds[0], :, p_inds[1], p_inds[2]]

        """Convert px shifts to absolute coordinates"""
        pxyz_mu[:, 1] += self._offset2coord.bin_ctr_x[p_inds[1]].to(pxyz_mu.device)
        pxyz_mu[:, 2] += self._offset2coord.bin_ctr_y[p_inds[2]].to(pxyz_mu.device)
        """Flatten img dimension --> N x (HxW) x 4"""
        pxyz_mu = pxyz_mu.reshape(batch_size, -1, 4)
        pxyz_sig = pxyz_sig[p_inds[0], :, p_inds[1], p_inds[2]].reshape(batch_size, -1, 4)

        """Set up mixture family"""
        mix = distributions.Categorical(prob_normed[p_inds].reshape(batch_size, -1))
        comp = distributions.Independent(distributions.Normal(pxyz_mu, pxyz_sig), 1)
        gmm = distributions.mixture_same_family.MixtureSameFamily(mix, comp)

        """Calc log probs if there is anything there"""
        if mask.sum():
            gmm_log = gmm.log_prob(pxyz_tar.transpose(0, 1)).transpose(0, 1)
            gmm_log = (gmm_log * mask).sum(-1)
            # print(f"LogProb: {log_prob.mean()}, GMM_log: {gmm_log.mean()}")
            log_prob = log_prob + gmm_log

        # log_prob = log_prob.reshape(batch_size, 1)  # need?

        loss = log_prob * (-1)

        return loss

    def _sigma11_loss(self, p, pxyz_mu, pxyz_sig, pxyz_tar, mask) -> torch.Tensor:
        """
        Computes the Gaussian Mixture Loss.

        Args:
            p: the model's detection prediction (sigmoid already applied) size N x H x W
            pxyz_mu: prediction of parameters (phot, xyz) size N x C=4 x H x W
            pxyz_sig: prediction of uncertainties / sigma values (phot, xyz) size N x C=4 x H x W
            pxyz_tar: ground truth values (phot, xyz) size N x M x 4 (M being max number of tars)
            mask: activation mask of ground truth values (phot, xyz) size N x M

        Returns:
            torch.Tensor (size N x 1)

        """

        batch_size = pxyz_mu.size(0)
        p_detached = p.detach()
        prob_normed = p_detached / p_detached.sum(-1).sum(-1).view(-1, 1, 1)

        """Hacky way to get all prob indices"""
        p_inds = tuple((p_detached + 1).nonzero(as_tuple=False).transpose(1, 0))
        pxyz_mu = pxyz_mu[p_inds[0], :, p_inds[1], p_inds[2]]

        """Convert px shifts to absolute coordinates"""
        #pxyz_mu[:, 1] += self._offset2coord.bin_ctr_x[p_inds[1]].to(pxyz_mu.device)
        #pxyz_mu[:, 2] += self._offset2coord.bin_ctr_y[p_inds[2]].to(pxyz_mu.device)

        """Flatten img dimension --> N x (HxW) x 4"""
        pxyz_mu = pxyz_mu.reshape(batch_size, -1, 2)
        pxyz_sig = pxyz_sig[p_inds[0], :, p_inds[1], p_inds[2]].reshape(batch_size, -1, 2)

        """Set up mixture family"""
        mix = distributions.Categorical(prob_normed[p_inds].reshape(batch_size, -1))
        comp = distributions.Independent(distributions.Normal(pxyz_mu, pxyz_sig), 1)
        gmm = distributions.mixture_same_family.MixtureSameFamily(mix, comp)

        """Calc log probs if there is anything there"""
        if mask.sum():
            gmm_log = gmm.log_prob(pxyz_tar.transpose(0, 1)).transpose(0, 1)
            gmm_log = (gmm_log * mask).sum(-1)
            # print(f"LogProb: {log_prob.mean()}, GMM_log: {gmm_log.mean()}")
            #log_prob = log_prob + gmm_log

        # log_prob = log_prob.reshape(batch_size, 1)  # need?

        loss = gmm_log * (-1)

        return  self.extra_weight * loss
    
    def _forward_checks(self, output: torch.Tensor, target: tuple, weight: None):

        if weight is not None:
            raise NotImplementedError(f"Weight must be None for this loss implementation.")

        if output.dim() != 4:
            raise ValueError(f"Output must have 4 dimensions (N,C,H,W).")

        if output.size(1) != 10:
            raise ValueError(f"Wrong number of channels.")

        if len(target) != 3:
            raise ValueError(f"Wrong length of target.")
        
    def _sig_loss(self, sigma_xyzi, crlb, locs): #[64, 4, 32, 32] , [64, 4, 32, 32], [64, 32, 32]
        sigma_xyzi = sigma_xyzi[:, 1:3, :, :]
        if locs.dim() == 3:
            locs = locs.unsqueeze(1)
        else:
            locs = locs

        sigma_xyzi = sigma_xyzi * locs  #[64, 2, 32, 32]
        crlb = crlb * locs
        #sig_loss = torch.nn.MSELoss(reduction='none')
        '''print(crlb[:,0].min())
        print(crlb[:,0].max())
        print(crlb[:,0].mean())
        print(crlb[:,1].min())
        print(crlb[:,1].max())
        print(crlb[:,1].mean())'''
        '''print(sigma_xyzi[:, 0].min())
        print(sigma_xyzi[:, 0].max())
        print(sigma_xyzi[:, 0].mean())
        print(sigma_xyzi[:, 1].min())
        print(sigma_xyzi[:, 1].max())
        print(sigma_xyzi[:, 1].mean())'''
        #cost = sig_loss(sigma_xyzi, crlb).sum(-1).sum(-1).sum(-1)
        cost = self.huber_loss(sigma_xyzi, crlb).sum(-1).sum(-1).sum(-1)
        
        return cost

    def huber_loss(self, input, target):
        abs_error = torch.abs(input - target)
        target_abs = torch.abs(target)
        use_mae = target_abs < 0.01
        use_mse = ~use_mae

        mae_loss = torch.where(use_mae, abs_error, torch.zeros_like(abs_error))
        mse_loss = torch.where(use_mse, abs_error ** 2, torch.zeros_like(abs_error))

        #mae_loss = mae_loss.mean()
        #mse_loss = mse_loss.mean()

        return mae_loss + mse_loss
        
    def eval_P_locs_loss(self, P, locs):
        if locs.dim() == 3:
            locs = locs.unsqueeze(1)
        else:
            locs = locs
        loss_cse = -(locs * torch.log(P) + (1 - locs) * torch.log(1 - P))
        loss_cse = loss_cse.sum(-1).sum(-1).sum(-1)
      
        return 0.1 * loss_cse

    def forward(self, output: torch.Tensor, target: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                weight: None) -> torch.Tensor:

        if self.forward_safety:
            self._forward_checks(output, target, weight)

        tar_param, tar_mask, tar_bg = target
        p, pxyz_mu, pxyz_sig, bg = self._format_model_output(output) #, crlb_sig
        bg_loss = self._bg_loss(bg, tar_bg).sum(-1).sum(-1)
        gmm_loss = self._compute_gmm_loss(p, pxyz_mu, pxyz_sig, tar_param[:,:,:4], tar_mask)
        #sigma_loss = self._sigma11_loss(p, pxyz_sig[:,1:3], crlb_sig, tar_param[:,:,4:], tar_mask)
        #p_locs_loss = self.eval_P_locs_loss(p, tar_locs)
        """Stack in 2 channels. 
        Factor 2 because original impl. adds the two terms, but this way it's better for logging."""
        loss = 2 * torch.stack((gmm_loss, bg_loss), 1) * self._ch_weight

        return loss
