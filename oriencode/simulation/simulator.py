import torch
from typing import Tuple, Union

from ..generic import EmitterSet
from . import psf_kernel

class Simulation:
    """
    A simulation class that holds the necessary modules, i.e. an emitter source (either a static EmitterSet or
    a function from which we can sample emitters), a psf background and noise. You may also specify the desired frame
    range, i.e. the indices of the frames you want to have as output. If they are not specified, they are automatically
    determined but may vary with new sampled emittersets.

    Attributes:
        em (EmitterSet): Static EmitterSet
        em_sampler: instance with 'sample()' method to sample EmitterSets from
        frame_range: frame indices between which to compute the frames. If None they will be
        auto-determined by the psf implementation.
        psf: psf model with forward method
        background (Background): background implementation
        noise (Noise): noise implementation
    """

    def __init__(self, psf: psf_kernel.PSF, em_sampler=None, em_sampler_unfocused=None, background=None, noise=None,
                 frame_range: Tuple[int, int] = None):
        """
        Init Simulation.

        Args:
            psf: point spread function instance
            em_sampler: callable that returns an EmitterSet upon call
            background: background instance
            noise: noise instance
            frame_range: limit frames to static range
        """

        self.em_sampler = em_sampler
        self.em_sampler_unfocused = em_sampler_unfocused
        self.frame_range = frame_range if frame_range is not None else (None, None)

        self.psf = psf
        self.background = background
        self.noise = noise

    def sample(self):
        """
        Sample a new set of emitters and forward them through the simulation pipeline.

        Returns:
            EmitterSet: sampled emitters
            torch.Tensor: simulated frames
            torch.Tensor: background frames
        """

        emitter = self.em_sampler()
        emitter_unfocused = self.em_sampler_unfocused.sample(emitter.xyz, emitter.frame_ix)
        frames_em, frames, bg = self.forward(emitter, emitter_unfocused)
        return emitter, frames_em, frames, bg
    
    def forward(self, em: EmitterSet, em_unfocused: EmitterSet = None, ix_low: Union[None, int] = None, ix_high: Union[None, int] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:   
        """
        Forward an EmitterSet through the simulation pipeline. 
        Setting ix_low or ix_high overwrites the frame range specified in the init.

        Args:
            em (EmitterSet): Emitter Set
            ix_low: lower frame index
            ix_high: upper frame index (inclusive)

        Returns:
            torch.Tensor: simulated frames
            torch.Tensor: background frames (e.g. to predict the bg seperately)
        """

        if ix_low is None:
            ix_low = self.frame_range[0]

        if ix_high is None:
            ix_high = self.frame_range[1]
        
        frames_em = self.psf.forward(em.xyz_px, em.phot, em.frame_ix,
                                  ix_low=ix_low, ix_high=ix_high)
         
        """
        Add background. This needs to happen here and not on a single frame, since background may be correlated.
        The difference between background and noise is, that background is assumed to be independent of the 
        emitter position / signal.
        """
        if self.background is not None:
            frames_em, bg_frames = self.background.forward(frames_em)
        else:
            bg_frames = None
        if self.noise is not None:
            frames_em = self.noise.forward(frames_em)
        
        '''crlb, _ = self.psf.crlb(em.xyz_px, em.phot, bg_frames)
        crlb_tar = torch.zeros_like(locs_tar.unsqueeze(1)).repeat_interleave(2, 1).cpu()
        s_inds = tuple(locs_tar.nonzero(as_tuple=False).transpose(1, 0))
        crlb_tar[s_inds[0], :, s_inds[1], s_inds[2]] = crlb[:, :2]'''
        if em_unfocused is None:
            frames = frames_em
        else:
            frames_unfocused = self.psf.forward(em_unfocused.xyz_px, em_unfocused.phot, em_unfocused.frame_ix,
                                    ix_low=ix_low, ix_high=ix_high)
            frames = frames_em + frames_unfocused.to(frames_em.device)
        
        return frames_em, frames, bg_frames
