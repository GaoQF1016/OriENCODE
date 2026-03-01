import torch

from abc import ABC, abstractmethod
from typing import Tuple
import math

class StructurePrior(ABC):
    """
    Abstract structure which can be sampled from. All implementation / childs must define a 'pop' method and an area
    property that describes the area the structure occupies.

    """

    @property
    @abstractmethod
    def area(self) -> float:
        """
        Calculate the area which is occupied by the structure. This is useful to later calculate the density,
        and the effective number of emitters). This is the 2D projection. Not the volume.

        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """
        Sample n samples from structure.

        Args:
            n: number of samples

        """
        raise NotImplementedError


class RandomStructure(StructurePrior):
    """
    Random uniform 3D / 2D structure. As the name suggests, sampling from this structure gives samples from a 3D / 2D
    volume that origin from a uniform distribution.

    """

    def __init__(self, xextent: Tuple[float, float], yextent: Tuple[float, float], zextent: Tuple[float, float]):
        """

        Args:
            xextent: extent in x
            yextent: extent in y
            zextent: extent in z, set (0., 0.) for a 2D structure

        Example:
            The following initialises this class in a range of 32 x 32 px in x and y and +/- 750nm in z.
            >>> prior_struct = RandomStructure(xextent=(-0.5, 31.5), yextent=(-0.5, 31.5), zextent=(-750., 750.))

        """
        super().__init__()
        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent

        self.scale = torch.tensor([(self.xextent[1] - self.xextent[0]),
                                   (self.yextent[1] - self.yextent[0]),
                                   (self.zextent[1] - self.zextent[0])])

        self.shift = torch.tensor([self.xextent[0],
                                   self.yextent[0],
                                   self.zextent[0]])

    @property
    def area(self) -> float:
        return (self.xextent[1] - self.xextent[0]) * (self.yextent[1] - self.yextent[0])

    def sample(self, n: int) -> torch.Tensor:
        xyz = torch.rand((n, 3)) * self.scale + self.shift
        return xyz

    @classmethod
    def parse(cls, param):
        return cls(xextent=param.Simulation.emitter_extent[0],
                   yextent=param.Simulation.emitter_extent[1],
                   zextent=param.Simulation.emitter_extent[2])

class RandomStructure_unfocused(StructurePrior):
    
    def __init__(self, xextent: Tuple[float, float], yextent: Tuple[float, float], zextent: Tuple[Tuple[float, float], Tuple[float, float]]):
        """
        参数：
            xextent: x 方向的范围
            yextent: y 方向的范围
            zextent: 两个元组的元组表示 z 范围

        """
        super().__init__()
        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent

        self.scale = torch.tensor([(self.xextent[1] - self.xextent[0]),
                                   (self.yextent[1] - self.yextent[0])
                                   ])

        self.shift = torch.tensor([self.xextent[0],
                                   self.yextent[0]])  # Initialize shift for z with the start of the first range

    @property
    def area(self) -> float:
        return (self.xextent[1] - self.xextent[0]) * (self.yextent[1] - self.yextent[0])

    def sample(self, n: int) -> torch.Tensor:
        
        num_samples_z1 = n // 2
        z1 = torch.rand((num_samples_z1, 1)) * (self.zextent[0][1] - self.zextent[0][0]) + self.zextent[0][0]

        num_samples_z2 = n - num_samples_z1
        z2 = torch.rand((num_samples_z2, 1)) * (self.zextent[1][1] - self.zextent[1][0]) + self.zextent[1][0]

        z_samples = torch.vstack((z1, z2))

        xy = torch.rand((n, 2)) * self.scale + self.shift

        xyz = torch.cat((xy, z_samples), dim=1)
        return xyz

    @classmethod
    def parse(cls, param):
        return cls(xextent=param.Simulation.emitter_extent[0],
                   yextent=param.Simulation.emitter_extent[1],
                   zextent=(
            (param.Simulation.emitter_extent[3][0], param.Simulation.emitter_extent[3][1]),
            (param.Simulation.emitter_extent[3][2], param.Simulation.emitter_extent[3][3])
        ))

class FixedXPointsStructure(StructurePrior):
    def __init__(self, x_values: list, yextent: Tuple[float, float], zextent: Tuple[float, float]):
        """
        Args:
            x_values: x方向允许的离散值列表 (如 [31.7, 32.3])
            yextent: y、z方向的随机范围
            zextent: z方向的随机范围
        """
        super().__init__()
        self.x_values = x_values
        self.yextent = yextent
        self.zextent = zextent
        self.scale_y = yextent[1] - yextent[0]
        self.scale_z = zextent[1] - zextent[0]

    @property
    def area(self) -> float:
        
        x_min = min(self.x_values)
        x_max = max(self.x_values)
        y_span = self.yextent[1] - self.yextent[0]
        return (x_max - x_min) * y_span
    
    def sample(self, n: int) -> torch.Tensor:
        # 生成随机索引来选择x的值 (均匀概率)
        '''x_indices = torch.randint(0, len(self.x_values), (n,))
        x = torch.tensor([self.x_values[i] for i in x_indices])

        # 生成y和z的均匀随机值
        y = torch.rand(n) * self.scale_y + self.yextent[0]
        z = torch.rand(n) * self.scale_z + self.zextent[0]'''
        
        x_indices = torch.randint(0, len(self.x_values), (n,))
        x = torch.tensor([self.x_values[i] for i in x_indices])

        y_edges = torch.linspace(self.yextent[0], self.yextent[1], n + 1)
        y_start = y_edges[:-1]
        y_end = y_edges[1:]
        y = torch.rand(n) * (y_end - y_start) + y_start

        z = torch.rand(n) * self.scale_z + self.zextent[0]

        return torch.stack([x, y, z], dim=1)

class FixedYPointsCylinderStructure(StructurePrior):
    def __init__(self, x_values: list, yextent: Tuple[float, float], center_z: float):
        """
        Args:
            x_values: x方向允许的离散值列表 (如 [31.7, 32.3])
            yextent: y方向的随机范围
            center_z: 圆柱体在z方向上的圆心位置
        """
        super().__init__()
        self.x_values = x_values
        self.yextent = yextent
        self.center_z = center_z
        self.scale_y = yextent[1] - yextent[0]
        
        self.center_x_pixel = (min(x_values) + max(x_values)) / 2
        self.radius_pixel = (max(x_values) - min(x_values)) / 2

        # 单位转换系数：像素 -> 纳米
        self.pixel_to_nm = 160.0

    @property
    def area(self) -> float:
        x_min = min(self.x_values)
        x_max = max(self.x_values)
        y_span = self.yextent[1] - self.yextent[0]
        return (x_max - x_min) * y_span

    def sample(self, n: int) -> torch.Tensor:
        # Calculate center and radius of the cylinder base
        '''x_min = min(self.x_values)
        x_max = max(self.x_values)
        center_x = (x_max + x_min) / 2
        radius = (x_max - x_min) / 2

        # Generate random angles for points on the cylinder surface in the xy-plane
        angles = torch.rand(n) * 2 * math.pi

        # Calculate x and z using parametric equation of the circle
        x = center_x + radius * torch.cos(angles)
        #z = 160 * (self.center_z + radius * torch.sin(angles))
        z = self.center_z + (radius * 160.0) * torch.sin(angles)

        # Generate y values randomly within the specified range
        y = torch.rand(n) * self.scale_y + self.yextent[0]'''
        # Combine coordinates
        
        x_min = min(self.x_values)
        x_max = max(self.x_values)
        center_x = (x_max + x_min) / 2
        radius = (x_max - x_min) / 2

        angle_edges = torch.linspace(0, 2 * math.pi, n + 1)
        angles = torch.rand(n) * (angle_edges[1:] - angle_edges[:-1]) + angle_edges[:-1]


        x = center_x + radius * torch.cos(angles)
        #z = 160 * (self.center_z + radius * torch.sin(angles))
        z = self.center_z + (radius * 160.0) * torch.sin(angles)

        y = torch.rand(n) * self.scale_y + self.yextent[0]
        return torch.stack([x, y, z], dim=1)

class TiltedCylinderStructure(FixedYPointsCylinderStructure):
    def __init__(self, x_values: list, yextent: Tuple[float, float], center_z: float, tilt_angle: float):
        """
        Args:
            x_values: x方向允许的离散值列表
            yextent: y方向的随机范围
            center_z: 圆柱体在z方向上的圆心位置
            tilt_angle: 圆柱在xy平面的旋转角度
        """
        super().__init__(x_values, yextent, center_z)
        self.tilt_angle = math.radians(tilt_angle)  
        
    @property
    def area(self) -> float:
        x_min = min(self.x_values)
        x_max = max(self.x_values)
        y_span = self.yextent[1] - self.yextent[0]
        return (x_max - x_min) * y_span
    
    def rotate_points_in_xy(self, points: torch.Tensor) -> torch.Tensor:

        rotation_matrix = torch.tensor([
            [math.cos(self.tilt_angle), -math.sin(self.tilt_angle), 0],
            [math.sin(self.tilt_angle), math.cos(self.tilt_angle), 0],
            [0, 0, 1]  # 保持z不变
        ])

        return points.mm(rotation_matrix)

    def sample(self, n: int) -> torch.Tensor:
        
        cylinder_points = super().sample(n)
        
        tilted_points = self.rotate_points_in_xy(cylinder_points)
        return tilted_points


from typing import Tuple, List, Union

class generate_unfocused_spots_around(StructurePrior):
    
    def __init__(self, xextent: Tuple[float, float], yextent: Tuple[float, float],
                 zextent: Tuple[Tuple[float, float], Tuple[float, float]],
                 max_xy_offset: float = 2.0): 
        super().__init__()
        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent
        self.max_xy_offset = max_xy_offset

        self.xy_scale = torch.tensor([(self.xextent[1] - self.xextent[0]),
                                      (self.yextent[1] - self.yextent[0])])

        self.xy_shift = torch.tensor([self.xextent[0],
                                       self.yextent[0]])

    @property
    def area(self) -> float:
        return (self.xextent[1] - self.xextent[0]) * (self.yextent[1] - self.yextent[0])
    
    def sample(self, xyz_centers: torch.Tensor, n: int = 3) -> torch.Tensor:
        """

        参数：
            xyz_centers: 正常点的坐标，形状 (N, 3)
            n: 每个正常点周围生成的离焦点数量

        返回：
            所有离焦点的坐标，形状 (N * n, 3)
        """
        N = xyz_centers.shape[0]

        # 存储所有离焦点
        all_unfocused_points = []

        for i in range(N):
            x0, y0, z0 = xyz_centers[i]

            # 在XY方向上生成n个随机偏移
            xy_offsets = (torch.rand((n, 2)) * 2 - 1) * self.max_xy_offset
            xy_unfocused = torch.tensor([x0, y0]) + xy_offsets

            # 在Z方向上随机选择偏移区间
            if torch.rand(1) < 0.5:
                z_range = self.zextent[0]
            else:
                z_range = self.zextent[1]
            z_unfocused = torch.rand((n, 1)) * (z_range[1] - z_range[0]) + z_range[0]

            unfocused_points = torch.cat((xy_unfocused, z_unfocused), dim=1)

            unfocused_points[:, 0].clamp_(*self.xextent)
            unfocused_points[:, 1].clamp_(*self.yextent)

            all_unfocused_points.append(unfocused_points)

        # 合并所有离焦点
        return torch.vstack(all_unfocused_points)

class RandomStructure_unfocused_around(StructurePrior):
    
    def __init__(self, xextent: Tuple[float, float], yextent: Tuple[float, float],
                 zextent: Tuple[Tuple[float, float], Tuple[float, float]],
                 max_xy_offset: float = 3.0): 
        """
        参数：
            xextent: x 方向的范围（用于约束采样）
            yextent: y 方向的范围（用于约束采样）
            zextent: z 方向的两个离焦区间
            max_xy_offset: 每个正常点周围XY方向的最大偏移
        """
        super().__init__()
        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent
        self.max_xy_offset = max_xy_offset

        self.xy_scale = torch.tensor([(self.xextent[1] - self.xextent[0]),
                                      (self.yextent[1] - self.yextent[0])])

        self.xy_shift = torch.tensor([self.xextent[0],
                                       self.yextent[0]])

    @property
    def area(self) -> float:
        return (self.xextent[1] - self.xextent[0]) * (self.yextent[1] - self.yextent[0])
    
    def sample(self, xyz_centers: torch.Tensor, frame_ix: torch.Tensor, n: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        参数：
            xyz_centers: (N, 3)
            frame_ix: (N, 1)
            n: 每个点周围生成的离焦点数量

        返回：
            Tuple[unfocused_xyz: (M, 3), unfocused_frame_ix: (M, 1)]
        """
        device = xyz_centers.device
        N = xyz_centers.shape[0]
        num_selected = N #// 2

        if num_selected <= 0:
            return torch.empty((0, 3), device=device), torch.empty((0, 1), device=device, dtype=torch.int64)

        # 随机选点
        selected_indices = torch.randperm(N, device=device)[:num_selected]
        selected_xyz = xyz_centers[selected_indices]  # (num_selected, 3)
        selected_frame_ix = frame_ix[selected_indices]  # (num_selected, 1)

        # 生成 XY 偏移
        xy_offsets = (torch.rand(num_selected, n, 2, device=device) * 2 - 1) * self.max_xy_offset
        xy_unfocused = selected_xyz[:, :2].unsqueeze(1) + xy_offsets  # (num_selected, n, 2)

        # 生成 Z 坐标
        z_ranges = torch.tensor(self.zextent, dtype=torch.float32, device=device)  # (2, 2)
        z_choice = torch.rand(num_selected, device=device) < 0.5
        z_min = torch.where(z_choice, z_ranges[0, 0], z_ranges[1, 0])
        z_max = torch.where(z_choice, z_ranges[0, 1], z_ranges[1, 1])

        z_unfocused = torch.rand(num_selected, n, 1, device=device) * (z_max - z_min).view(-1, 1, 1) + z_min.view(-1, 1, 1)

        unfocused_xyz = torch.cat([
            xy_unfocused.view(-1, 2),
            z_unfocused.view(-1, 1)
        ], dim=1)

        unfocused_xyz[:, 0].clamp_(self.xextent[0], self.xextent[1])
        unfocused_xyz[:, 1].clamp_(self.yextent[0], self.yextent[1])

        unfocused_frame_ix = selected_frame_ix.repeat_interleave(n, dim=0)  # (num_selected * n, 1)
        
        unfocused_xyz = unfocused_xyz.detach()
        unfocused_frame_ix = unfocused_frame_ix.detach()

        return unfocused_xyz, unfocused_frame_ix


    @classmethod
    def parse(cls, param):
        return cls(xextent=param.Simulation.emitter_extent[0],
                   yextent=param.Simulation.emitter_extent[1],
                   zextent=(
            (param.Simulation.emitter_extent[3][0], param.Simulation.emitter_extent[3][1]),
            (param.Simulation.emitter_extent[3][2], param.Simulation.emitter_extent[3][3])
        ))