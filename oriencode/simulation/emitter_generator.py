from abc import ABC, abstractmethod  # abstract class
from deprecated import deprecated

import numpy as np
import torch
from torch.distributions.exponential import Exponential

import decode.generic.emitter
from . import structure_prior


class EmitterSampler(ABC):
    """
    Abstract emitter sampler. All implementations / childs must implement a sample method.
    """

    def __init__(self, structure: structure_prior.StructurePrior, xy_unit: str, px_size: tuple):

        super().__init__()

        self.structure = structure
        self.px_size = px_size
        self.xy_unit = xy_unit

    def __call__(self) -> decode.generic.emitter.EmitterSet:
        return self.sample()

    @abstractmethod
    def sample(self) -> decode.generic.emitter.EmitterSet:
        raise NotImplementedError


class EmitterSamplerFrameIndependent(EmitterSampler):
    """
    Simple Emitter sampler. Samples emitters from a structure and puts them all on the same frame, i.e. their
    blinking model is not modelled.

    """

    def __init__(self, *, structure: structure_prior.StructurePrior, photon_range: tuple,
                 density: float = None, em_avg: float = None, xy_unit: str, px_size: tuple):
        """

        Args:
            structure: structure to sample from
            photon_range: range of photon value to sample from (uniformly)
            density: target emitter density (exactly only when em_avg is None)
            em_avg: target emitter average (exactly only when density is None)
            xy_unit: emitter xy unit
            px_size: emitter pixel size

        """

        super().__init__(structure=structure, xy_unit=xy_unit, px_size=px_size)

        self._density = density
        self.photon_range = photon_range

        """
        Sanity Checks.
        U shall not pa(rse)! (Emitter Average and Density at the same time!
        """
        if (density is None and em_avg is None) or (density is not None and em_avg is not None):
            raise ValueError("You must XOR parse either density or emitter average. Not both or none.")

        self.area = self.structure.area

        if em_avg is not None:
            self._em_avg = em_avg
        else:
            self._em_avg = self._density * self.area

    @property
    def em_avg(self) -> float:
        return self._em_avg

    def sample(self) -> decode.generic.emitter.EmitterSet:
        """
        Sample an EmitterSet.

        Returns:
            EmitterSet:

        """
        n = np.random.poisson(lam=self._em_avg)

        return self.sample_n(n=n)

    def sample_n(self, n: int) -> decode.generic.emitter.EmitterSet:
        """
        Sample 'n' emitters, i.e. the number of emitters is given and is not sampled from the Poisson dist.

        Args:
            n: number of emitters

        """

        if n < 0:
            raise ValueError("Negative number of samples is not well-defined.")

        xyz = self.structure.sample(n)
        phot = torch.randint(*self.photon_range, (n,))

        return decode.generic.emitter.EmitterSet(xyz=xyz, phot=phot,
                                                 frame_ix=torch.zeros_like(phot).long(),
                                                 id=torch.arange(n).long(),
                                                 xy_unit=self.xy_unit,
                                                 px_size=self.px_size)


class EmitterSamplerBlinking(EmitterSamplerFrameIndependent):
    def __init__(self, *, structure: structure_prior.StructurePrior, intensity_mu_sig: tuple, lifetime: float,
                 frame_range: tuple, xy_unit: str, px_size: tuple, density=None, em_avg=None, intensity_th=None):
        """

        Args:
            structure:
            intensity_mu_sig:
            lifetime:
            xy_unit:
            px_size:
            frame_range: specifies the frame range
            density:
            em_avg:
            intensity_th:

        """
        super().__init__(structure=structure,
                         photon_range=None,
                         xy_unit=xy_unit,
                         px_size=px_size,
                         density=density,
                         em_avg=em_avg)

        self.n_sampler = np.random.poisson
        self.frame_range = frame_range
        self.intensity_mu_sig = intensity_mu_sig
        self.intensity_dist = torch.distributions.normal.Normal(self.intensity_mu_sig[0],
                                                                self.intensity_mu_sig[1])
        #self.intensity_th = intensity_th if intensity_th is not None else 1e-8
        self.intensity_th = 100
        self.lifetime_avg = lifetime
        self.lifetime_dist = Exponential(1 / self.lifetime_avg)  # parse the rate not the scale ...

        self.t0_dist = torch.distributions.uniform.Uniform(*self._frame_range_plus)

        """
        Determine the total number of emitters. Depends on lifetime, frames and emitters.
        (lifetime + 1) because of binning effect.
        """
        self._emitter_av_total = self._em_avg * self._num_frames_plus / (self.lifetime_avg + 1)

    @property
    def _frame_range_plus(self):
        """
        Frame range including buffer in front and end to account for build up effects.

        """
        return self.frame_range[0] - 3 * self.lifetime_avg, self.frame_range[1] + 3 * self.lifetime_avg

    @property
    def num_frames(self):
        return self.frame_range[1] - self.frame_range[0] + 1

    @property
    def _num_frames_plus(self):
        return self._frame_range_plus[1] - self._frame_range_plus[0] + 1

    def sample(self):
        """
        Return sampled EmitterSet in the specified frame range.

        Returns:
            EmitterSet

        """

        n = self.n_sampler(self._emitter_av_total)
        

        loose_em = self.sample_loose_emitter(n=n)
        em = loose_em.return_emitterset()
        '''print(em.phot.min())
        print(em.phot.max())'''
        em = em.get_subset_frame(*self.frame_range)  # because the simulated frame range is larger
        
        return em

    def sample_n(self, *args, **kwargs):
        raise NotImplementedError

    def sample_loose_emitter(self, n) -> decode.generic.emitter.LooseEmitterSet:
        """
        Generate loose EmitterSet. Loose emitters are emitters that are not yet binned to frames.

        Args:
            n: number of 'loose' emitters

        Returns:
            LooseEmitterSet

        """

        xyz = self.structure.sample(n)    #（n, 3）

        """Draw from intensity distribution but clamp the value so as not to fall below 0."""
        intensity = torch.clamp(self.intensity_dist.sample((n,)), 500)

        """Distribute emitters in time. Increase the range a bit."""
        t0 = self.t0_dist.sample((n,))
        ontime = self.lifetime_dist.rsample((n,))
        
        
        return decode.generic.emitter.LooseEmitterSet(xyz, intensity, ontime, t0, id=torch.arange(n).long(),
                                                      xy_unit=self.xy_unit, px_size=self.px_size)

    @classmethod
    def parse(cls, param, structure, frames: tuple):
        return cls(structure=structure,
                   intensity_mu_sig=param.Simulation.intensity_mu_sig,
                   lifetime=param.Simulation.lifetime_avg,
                   xy_unit=param.Simulation.xy_unit,
                   px_size=param.Camera.px_size,
                   frame_range=frames,
                   density=param.Simulation.density,
                   em_avg=param.Simulation.emitter_av,
                   intensity_th=param.Simulation.intensity_th)

class EmitterSamplerBlinking_Ab2(EmitterSamplerFrameIndependent):
    def __init__(self, *, structure: structure_prior.StructurePrior, intensity_mu_sig: tuple, lifetime: float,
                 frame_range: tuple, xy_unit: str, px_size: tuple, density=None, em_avg=None, intensity_th=None):
        """

        Args:
            structure:
            intensity_mu_sig:
            lifetime:
            xy_unit:
            px_size:
            frame_range: specifies the frame range
            density:
            em_avg:
            intensity_th:

        """
        super().__init__(structure=structure,
                         photon_range=None,
                         xy_unit=xy_unit,
                         px_size=px_size,
                         density=None,
                         em_avg=em_avg if em_avg is not None else 1)

        self.n_sampler = np.random.poisson
        self.frame_range = frame_range
        self.intensity_mu_sig = intensity_mu_sig
        self.intensity_dist = torch.distributions.normal.Normal(self.intensity_mu_sig[0],
                                                                self.intensity_mu_sig[1])
        #self.intensity_th = intensity_th if intensity_th is not None else 1e-8
        self.intensity_th = float(intensity_th if intensity_th is not None else 1e-8) 
        self.lifetime_avg = float(lifetime)

    @property
    def _frame_range_plus(self):
        """
        Frame range including buffer in front and end to account for build up effects.

        """
        return self.frame_range[0] - 3 * self.lifetime_avg, self.frame_range[1] + 3 * self.lifetime_avg

    @property
    def num_frames(self):
        return self.frame_range[1] - self.frame_range[0] + 1

    @property
    def _num_frames_plus(self):
        return self._frame_range_plus[1] - self._frame_range_plus[0] + 1

    def sample(self):
    

        n = 1000
        frames_per_emitter = 10
        start_frame = self.frame_range[0]
        end_frame = self.frame_range[1]

        # 检查帧数是否匹配
        expected_frames = n * frames_per_emitter
        actual_frames = end_frame - start_frame + 1
        if expected_frames != actual_frames:
            raise ValueError(f"Expected {expected_frames} frames, but actual is {actual_frames}.")

        # 采样空间坐标
        xyz = self.structure.sample(n)  # shape: (n, 3)

        # 采样强度
        intensity = torch.clamp(self.intensity_dist.sample((n,)), min=500)  # shape: (n, )

        # 扩展坐标、强度、帧号、ID
        xyz_expanded = xyz.unsqueeze(1).repeat(1, frames_per_emitter, 1).view(-1, 3)  # (n*frames_per_emitter, 3)
        intensity_expanded = intensity.unsqueeze(1).repeat(1, frames_per_emitter).view(-1)  # (n*frames_per_emitter, )
        frame_ix = torch.arange(start_frame, start_frame + n * frames_per_emitter, dtype=torch.long)  # (n*frames_per_emitter, )
        id_ = torch.arange(n, dtype=torch.long).unsqueeze(1).repeat(1, frames_per_emitter).view(-1)  # (n*frames_per_emitter, )

        # 构造 EmitterSet
        return decode.generic.emitter.EmitterSet(
            xyz=xyz_expanded,
            phot=intensity_expanded,
            frame_ix=frame_ix,
            id=id_,
            xy_unit=self.xy_unit,
            px_size=self.px_size
        )


    def sample_n(self, *args, **kwargs):
        raise NotImplementedError

    def sample_loose_emitter(self) -> decode.generic.emitter.LooseEmitterSet:
        
        frames_per_emitter = max(1, int(round(self.lifetime_avg)))
        start = float(self.frame_range[0])
        total_frames = self.num_frames

        # 需要的发射体个数：铺满时间窗，最后一个允许截断
        n_full = total_frames // frames_per_emitter
        rem = total_frames % frames_per_emitter
        n = n_full + (1 if rem > 0 else 0)

        # 采样空间位置
        xyz = self.structure.sample(n)  # (n, 3)

        intensity = torch.clamp(self.intensity_dist.sample((n,)), 500)
        # 时间：从 frame_range[0] 开始，步长为 frames_per_emitter，确保任何时刻只有一个点在亮
        t0 = start + torch.arange(0, n * frames_per_emitter, step=frames_per_emitter, dtype=torch.float32)
        ontime = torch.full((n,), float(frames_per_emitter), dtype=torch.float32)
        if rem > 0:
            ontime[-1] = float(rem)
        return decode.generic.emitter.LooseEmitterSet(
            xyz, intensity, ontime, t0, id=torch.arange(n).long(),
            xy_unit=self.xy_unit, px_size=self.px_size)

    @classmethod
    def parse(cls, param, structure, frames: tuple):
        return cls(structure=structure,
                   intensity_mu_sig=param.Simulation.intensity_mu_sig,
                   lifetime=param.Simulation.lifetime_avg,
                   xy_unit=param.Simulation.xy_unit,
                   px_size=param.Camera.px_size,
                   frame_range=frames,
                   density=None,
                   em_avg=None,
                   intensity_th=param.Simulation.intensity_th)
        
class EmitterSamplerBlinking_unfocused(EmitterSamplerFrameIndependent):
    def __init__(self, *, structure: structure_prior.StructurePrior, intensity_mu_sig: tuple, lifetime: float,
                 frame_range: tuple, xy_unit: str, px_size: tuple, density=None, em_avg=None, intensity_th=None):
        """

        Args:
            structure:
            intensity_mu_sig:
            lifetime:
            xy_unit:
            px_size:
            frame_range: specifies the frame range
            density:
            em_avg:
            intensity_th:

        """
        super().__init__(structure=structure,
                         photon_range=None,
                         xy_unit=xy_unit,
                         px_size=px_size,
                         density=density,
                         em_avg=em_avg)

        self.n_sampler = np.random.poisson
        self.frame_range = frame_range
        self.intensity_mu_sig = intensity_mu_sig
        self.intensity_dist = torch.distributions.normal.Normal(self.intensity_mu_sig[0],
                                                                self.intensity_mu_sig[1])
        #self.intensity_th = intensity_th if intensity_th is not None else 1e-8
        self.intensity_th = 100
        self.lifetime_avg = lifetime
        self.lifetime_dist = Exponential(1 / self.lifetime_avg)  # parse the rate not the scale ...

        self.t0_dist = torch.distributions.uniform.Uniform(*self._frame_range_plus)

        """
        Determine the total number of emitters. Depends on lifetime, frames and emitters.
        (lifetime + 1) because of binning effect.
        """
        self._emitter_av_total = self._em_avg * self._num_frames_plus / (self.lifetime_avg + 1)

    @property
    def _frame_range_plus(self):
        """
        Frame range including buffer in front and end to account for build up effects.

        """
        return self.frame_range[0] - 3 * self.lifetime_avg, self.frame_range[1] + 3 * self.lifetime_avg

    @property
    def num_frames(self):
        return self.frame_range[1] - self.frame_range[0] + 1

    @property
    def _num_frames_plus(self):
        return self._frame_range_plus[1] - self._frame_range_plus[0] + 1

    def sample(self, em_xyz, em_frame_ix):
        """
        Return sampled EmitterSet in the specified frame range.

        Returns:
            EmitterSet

        """

        #n = self.n_sampler(self._emitter_av_total)
        

        loose_em = self.sample_loose_emitter(em_xyz, em_frame_ix)
        #em = loose_em.return_emitterset()
        '''print(em.phot.min())
        print(em.phot.max())'''
        em = loose_em.get_subset_frame(*self.frame_range)  # because the simulated frame range is larger
        
        return em

    def sample_n(self, *args, **kwargs):
        raise NotImplementedError

    def sample_loose_emitter(self, em_xyz, em_frame_ix):
        """
        Generate loose EmitterSet. Loose emitters are emitters that are not yet binned to frames.

        Args:
            n: number of 'loose' emitters

        Returns:
            LooseEmitterSet

        """

        xyz, frame_ix = self.structure.sample(em_xyz, em_frame_ix)    #（n, 3）
        n = xyz.shape[0]

        """Draw from intensity distribution but clamp the value so as not to fall below 0."""
        intensity = torch.clamp(self.intensity_dist.sample((n,)), 1000)

        """Distribute emitters in time. Increase the range a bit."""
        t0 = self.t0_dist.sample((n,))
        ontime = self.lifetime_dist.rsample((n,))
        
        return decode.generic.emitter.EmitterSet(xyz, intensity, frame_ix.long(), id=torch.arange(n).long(), xy_unit=self.xy_unit, px_size=self.px_size)
        '''return decode.generic.emitter.LooseEmitterSet(xyz, intensity, ontime, t0, id=torch.arange(n).long(),
                                                      xy_unit=self.xy_unit, px_size=self.px_size)'''
                                                      

    @classmethod
    def parse(cls, param, structure, frames: tuple):
        return cls(structure=structure,
                   intensity_mu_sig=param.Simulation.intensity_mu_sig_unfocused,
                   lifetime=param.Simulation.lifetime_avg,
                   xy_unit=param.Simulation.xy_unit,
                   px_size=param.Camera.px_size,
                   frame_range=frames,
                   density=param.Simulation.density,
                   em_avg=param.Simulation.emitter_av_unfocused,
                   intensity_th=param.Simulation.intensity_th)
        
@deprecated(reason="Deprecated in favour of EmitterSamplerFrameIndependent.", version="0.1.dev")
class EmitterPopperSingle:
    pass


@deprecated(reason="Deprecated in favour of EmitterSamplerBlinking.", version="0.1.dev")
class EmitterPopperMultiFrame:
    pass
