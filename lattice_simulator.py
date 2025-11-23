import numpy as np
from typing import List, Tuple, Dict, Any
import copy
from dataclasses import dataclass
from multiprocessing import Pool
from functools import partial
# 从scipy导入物理常数
from scipy.constants import physical_constants

@dataclass
class ElementParams:
    """存储晶格元件参数的类"""
    element_type: str  # 元件类型: 'drift', 'quadrupole', 'hcor', 'vcor', 'monitor'
    length: float      # 元件长度
    k1: float = 0.0    # 四极磁铁强度
    angle: float = 0.0 # 校正器角度
    tilt: float = 0.0  # 倾斜角度

@dataclass
class Particle:
    """粒子状态类"""
    x: float = 0.0   # 水平位置
    px: float = 0.0  # 水平角度
    y: float = 0.0   # 垂直位置
    py: float = 0.0  # 垂直角度
    tau: float = 0.0 # 纵向位置
    p: float = 0.0   # 纵向动量偏差
    E: float = 0.0   # 能量

def extract_lattice_parameters(lattice) -> List[ElementParams]:
    """
    从Ocelot晶格中提取参数
    
    Parameters:
        lattice: Ocelot MagneticLattice对象
    
    Returns:
        List[ElementParams]: 晶格元件参数列表
    """
    params_list = []
    
    for element in lattice.sequence:
        if element.__class__.__name__ == 'Drift':
            params_list.append(ElementParams(
                element_type='drift',
                length=element.l
            ))
        elif element.__class__.__name__ == 'Quadrupole':
            params_list.append(ElementParams(
                element_type='quadrupole',
                length=element.l,
                k1=element.k1,
                tilt=getattr(element, 'tilt', 0.0)
            ))
        elif element.__class__.__name__ == 'Hcor':
            params_list.append(ElementParams(
                element_type='hcor',
                length=element.l,
                angle=element.angle
            ))
        elif element.__class__.__name__ == 'Vcor':
            params_list.append(ElementParams(
                element_type='vcor',
                length=element.l,
                angle=element.angle
            ))
        elif element.__class__.__name__ == 'Monitor':
            params_list.append(ElementParams(
                element_type='monitor',
                length=element.l
            ))
    
    return params_list

def rot_mtx(angle: float) -> np.ndarray:
    """
    生成旋转矩阵
    
    Parameters:
        angle: 旋转角度
    
    Returns:
        np.ndarray: 6x6旋转矩阵
    """
    cs = np.cos(angle)
    sn = np.sin(angle)
    return np.array([[cs, 0., sn, 0., 0., 0.],
                     [0., cs, 0., sn, 0., 0.],
                     [-sn, 0., cs, 0., 0., 0.],
                     [0., -sn, 0., cs, 0., 0.],
                     [0., 0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 0., 1.]])

def uni_matrix(z: float, k1: float, hx: float, sum_tilts: float = 0., energy: float = 0.) -> np.ndarray:
    """
    通用转移矩阵计算
    
    Parameters:
        z: 元件长度
        k1: 四极磁铁强度
        hx: 曲率
        sum_tilts: 倾斜角度
        energy: 粒子能量
    
    Returns:
        np.ndarray: 6x6转移矩阵
    """
    # 使用scipy中的电子质量常数
    m_e_GeV = physical_constants['electron mass energy equivalent in MeV'][0] / 1000.0
    gamma = energy / m_e_GeV if m_e_GeV != 0 else 0
    
    kx2 = (k1 + hx * hx)
    ky2 = -k1
    kx = np.sqrt(kx2 + 0.j)
    ky = np.sqrt(ky2 + 0.j)
    cx = np.cos(z * kx).real
    cy = np.cos(z * ky).real
    sy = (np.sin(ky * z) / ky).real if ky != 0 else z

    igamma2 = 0.
    if gamma != 0:
        igamma2 = 1. / (gamma * gamma)

    beta = np.sqrt(1. - igamma2) if gamma != 0 else 1

    if kx != 0:
        sx = (np.sin(kx * z) / kx).real
        dx = hx / kx2 * (1. - cx)
        r56 = hx * hx * (z - sx) / kx2 / beta ** 2
    else:
        sx = z
        dx = z * z * hx / 2.
        r56 = hx * hx * z ** 3 / 6. / beta ** 2

    r56 -= z / (beta * beta) * igamma2

    u_matrix = np.array([[cx, sx, 0., 0., 0., dx / beta],
                         [-kx2 * sx, cx, 0., 0., 0., sx * hx / beta],
                         [0., 0., cy, sy, 0., 0.],
                         [0., 0., -ky2 * sy, cy, 0., 0.],
                         [hx * sx / beta, dx / beta, 0., 0., 1., r56],
                         [0., 0., 0., 0., 0., 1.]])
                         
    if sum_tilts != 0:
        u_matrix = np.dot(np.dot(rot_mtx(-sum_tilts), u_matrix), rot_mtx(sum_tilts))
    return u_matrix

def kick_b(z: float, l: float, angle: float, tilt: float) -> np.ndarray:
    """
    计算校正器的踢动向量
    
    Parameters:
        z: 位置
        l: 长度
        angle: 踢动角度
        tilt: 倾斜角度
    
    Returns:
        np.ndarray: 6x1踢动向量
    """
    angle_x = angle * np.cos(tilt)
    angle_y = angle * np.sin(tilt)
    if l == 0:
        hx = 0.
        hy = 0.
    else:
        hx = angle_x / l
        hy = angle_y / l

    dx = hx * z * z / 2.
    dy = hy * z * z / 2.
    dx1 = hx * z if l != 0 else angle_x
    dy1 = hy * z if l != 0 else angle_y
    b = np.array([[dx], [dx1], [dy], [dy1], [0.], [0.]])
    return b

def apply_element_transfer_map(particle: Particle, element_params: ElementParams, energy: float) -> Particle:
    """
    应用单个元件的转移矩阵到粒子
    
    Parameters:
        particle: 输入粒子
        element_params: 元件参数
        energy: 粒子能量
    
    Returns:
        Particle: 经过元件后的粒子
    """
    # 将粒子状态转换为数组形式
    X = np.array([[particle.x], [particle.px], [particle.y], [particle.py], [particle.tau], [particle.p]])
    
    R = np.eye(6)  # 默认单位矩阵
    B = np.zeros((6, 1))  # 默认零向量
    
    if element_params.element_type == 'drift':
        R = uni_matrix(element_params.length, 0, 0, element_params.tilt, energy)
    elif element_params.element_type == 'quadrupole':
        R = uni_matrix(element_params.length, element_params.k1, 0, element_params.tilt, energy)
    elif element_params.element_type == 'hcor':
        R = uni_matrix(element_params.length, 0, 0, element_params.tilt, energy)
        B = kick_b(element_params.length, element_params.length, element_params.angle, 0)
    elif element_params.element_type == 'vcor':
        R = uni_matrix(element_params.length, 0, 0, element_params.tilt, energy)
        B = kick_b(element_params.length, element_params.length, element_params.angle, np.pi/2)
    
    # 应用转移矩阵: X1 = R * X + B
    X_new = np.dot(R, X) + B
    
    # 更新粒子状态
    new_particle = copy.copy(particle)
    new_particle.x = X_new[0, 0]
    new_particle.px = X_new[1, 0]
    new_particle.y = X_new[2, 0]
    new_particle.py = X_new[3, 0]
    new_particle.tau = X_new[4, 0]
    new_particle.p = X_new[5, 0]
    
    return new_particle

def track_particle_through_lattice(particle: Particle, 
                                 lattice_params: List[ElementParams], 
                                 energy: float) -> Tuple[List[float], List[float], List[float]]:
    """
    粒子在晶格中的追踪
    
    Parameters:
        particle: 初始粒子
        lattice_params: 晶格参数列表
        energy: 粒子能量
    
    Returns:
        Tuple[List[float], List[float], List[float]]: BPM位置的(x, y)坐标和s位置
    """
    current_particle = copy.copy(particle)
    bpm_x_positions = []
    bpm_y_positions = []
    bpm_s_positions = []
    
    s = 0.0  # 当前位置
    
    for element_params in lattice_params:
        # 如果是监测器，记录当前粒子位置
        if element_params.element_type == 'monitor':
            bpm_x_positions.append(current_particle.x)
            bpm_y_positions.append(current_particle.y)
            bpm_s_positions.append(s)
        
        # 应用转移矩阵
        current_particle = apply_element_transfer_map(current_particle, element_params, energy)
        
        # 更新当前位置
        s += element_params.length
    
    return bpm_x_positions, bpm_y_positions, bpm_s_positions

def compute_orbit(lattice_params: List[ElementParams],
                 hcors_angles: List[float],
                 vcors_angles: List[float],
                 initial_particle: Particle,
                 energy: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算轨道的纯函数
    
    Parameters:
        lattice_params: 晶格参数列表
        hcors_angles: 水平校正器角度列表
        vcors_angles: 垂直校正器角度列表
        initial_particle: 初始粒子状态
        energy: 粒子能量
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (x轨道, y轨道)
    """
    # 复制晶格参数以避免修改原始数据
    params_copy = copy.deepcopy(lattice_params)
    
    # 更新校正器角度
    hcor_idx = 0
    vcor_idx = 0
    for param in params_copy:
        if param.element_type == 'hcor':
            param.angle = hcors_angles[hcor_idx]
            hcor_idx += 1
        elif param.element_type == 'vcor':
            param.angle = vcors_angles[vcor_idx]
            vcor_idx += 1
    
    # 追踪粒子
    x_positions, y_positions, _ = track_particle_through_lattice(initial_particle, params_copy, energy)
    
    return np.array(x_positions), np.array(y_positions)

def compute_orbit_parallel(args) -> Tuple[np.ndarray, np.ndarray]:
    """
    用于并行计算的包装函数
    
    Parameters:
        args: (lattice_params, hcors_angles, vcors_angles, initial_particle, energy)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (x轨道, y轨道)
    """
    return compute_orbit(*args)

def batch_compute_orbits(lattice_params: List[ElementParams],
                        hcors_angles_list: List[List[float]],
                        vcors_angles_list: List[List[float]],
                        initial_particles: List[Particle],
                        energy: float = 1.0,
                        num_processes: int = 4) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    批量并行计算多个轨道
    
    Parameters:
        lattice_params: 晶格参数列表
        hcors_angles_list: 多组水平校正器角度列表
        vcors_angles_list: 多组垂直校正器角度列表
        initial_particles: 多个初始粒子状态列表
        energy: 粒子能量
        num_processes: 并行进程数
    
    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: 轨道计算结果列表
    """
    # 准备参数
    args_list = [
        (lattice_params, hcors_angles_list[i], vcors_angles_list[i], initial_particles[i], energy)
        for i in range(len(hcors_angles_list))
    ]
    
    # 并行计算
    with Pool(processes=num_processes) as pool:
        results = pool.map(compute_orbit_parallel, args_list)
    
    return results

# 使用示例
if __name__ == "__main__":
    # 这里展示如何使用该模块
    print("Lattice Simulator模块已加载")
    print("请使用extract_lattice_parameters从Ocelot晶格提取参数")
    print("然后使用compute_orbit或batch_compute_orbits计算轨道")