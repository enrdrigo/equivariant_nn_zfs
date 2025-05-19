from mace import data, modules, tools
import torch
from e3nn.o3 import Irreps
import numpy as np

def cartesian_to_spherical_irreps(matrix):
    """
    Convert a 3x3 real symmetric matrix into spherical components
    ordered according to Irreps("0e + 1o + 2e").

    Returns:
        components: torch.Tensor of shape (9,) — spherical components
        irreps: Irreps object — "0e + 1o + 2e"
    """
    assert matrix.shape == (3, 3)
    irreps = Irreps("0e + 1o + 2e")
    components = np.zeros(9)

    # ℓ = 0 component (scalar: trace)
    components[0] = matrix.trace() / 3.0

    # ℓ = 1 components (antisymmetric part, imaginary in true tensors, zero here)
    components[1] = (matrix[1, 2] - matrix[2, 1])   # Y1-1
    components[2] = (matrix[0, 2] - matrix[2, 0])   # Y10
    components[3] = (matrix[0, 1] - matrix[1, 0])   # Y11

    # ℓ = 2 components (traceless symmetric part)
    components[4] = (matrix[0, 1] + matrix[1, 0])  # xy Y2-2
    components[5] = (matrix[1, 2] + matrix[2, 1])  # yz Y2-1
    components[6] = 2 * matrix[2, 2] - matrix[0, 0] - matrix[1, 1]  # zz - xx - yyY20
    components[7] = (matrix[0, 2] + matrix[2, 0])  # xz Y21
    components[8] = matrix[0, 0] - matrix[1, 1]  # xx - yy Y22

    return components

def spherical_components_torch_to_cartesian_tensor(T_sph: torch.Tensor) -> torch.Tensor:
    """
    Convert spherical tensor components to 3×3 real Cartesian tensors.
    Input: (batch_size, 9)
    Output: (batch_size, 3, 3)
    Safe for autograd (no in-place ops).
    """
    assert T_sph.shape[-1] == 9, "Last dimension must be 9"

    T00, T1_1, T10, T11, T2_2, T2_1, T20, T21, T22 = T_sph.unbind(dim=-1)

    sqrt3 = torch.sqrt(torch.tensor(3.0, dtype=T_sph.dtype, device=T_sph.device))
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=T_sph.dtype, device=T_sph.device))
    sqrt32 = torch.sqrt(torch.tensor(1.5, dtype=T_sph.dtype, device=T_sph.device))

    batch_size = T_sph.shape[0]
    eye = torch.eye(3, dtype=T_sph.dtype, device=T_sph.device).unsqueeze(0).repeat(batch_size, 1, 1)

    # ℓ = 0
    trace = T00 * sqrt3
    T0 = (trace / 3.0).unsqueeze(-1).unsqueeze(-1) * eye

    # ℓ = 1
    v = torch.stack([T11, T10, T1_1], dim=-1) / sqrt2
    zeros = torch.zeros(batch_size, dtype=T_sph.dtype, device=T_sph.device)

    row0 = torch.stack([zeros,    v[:, 0],  v[:, 1]], dim=-1)
    row1 = torch.stack([-v[:, 0], zeros,    v[:, 2]], dim=-1)
    row2 = torch.stack([-v[:, 1], -v[:, 2], zeros],   dim=-1)
    T1 = torch.stack([row0, row1, row2], dim=1)

    # ℓ = 2
    T2_xy = T2_2 / sqrt2
    T2_yz = T2_1 / sqrt2
    T2_xz = T21 / sqrt2
    T2_zz = T20 / sqrt32
    T2_xx =  T22 / sqrt2
    T2_yy = -T22 / sqrt2

    T2 = torch.stack([
        torch.stack([T2_xx, T2_xy, T2_xz], dim=-1),
        torch.stack([T2_xy, T2_yy, T2_yz], dim=-1),
        torch.stack([T2_xz, T2_yz, T2_zz], dim=-1)
    ], dim=1)
    return T0 + T1 + T2