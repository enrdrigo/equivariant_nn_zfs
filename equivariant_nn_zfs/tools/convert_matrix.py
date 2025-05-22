import torch

def cartesian_to_spherical_irreps(cartmat: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3x3 real symmetric matrix into spherical components
    ordered according to Irreps("0e + 1o + 2e").

    Returns:
        components: torch.Tensor of shape (9,) — spherical components
        irreps: Irreps object — "0e + 1o + 2e"
    """
    assert cartmat.shape == (3, 3)

    def t(data):
        return torch.tensor(data, dtype=cartmat.dtype, device=cartmat.device)

    #                        xx  xy  xz  yx  yy  yz  zx  zy  zz

    matrix = torch.stack([t([1/3,  0,  0,  0,  1/3,  0,  0,  0,  1/3]),
                          t([0,  0,  0,  0,  0,  1,  0, -1,  0]),
                          t([0,  0,  1,  0,  0,  0, -1,  0,  0]),
                          t([0,  1,  0, -1,  0,  0,  0,  0,  0]),
                          t([0,  1,  0,  1,  0,  0,  0,  0,  0]),
                          t([0,  0,  0,  0,  0,  1,  0,  1,  0]),
                          t([-1, 0,  0,  0, -1,  0,  0,  0,  2]),
                          t([0,  0,  1,  0,  0,  0,  1,  0,  0]),
                          t([1,  0,  0,  0, -1,  0,  0,  0,  0])
                          ],
                         dim=0
                         )

    return torch.matmul(matrix, cartmat.flatten())

def spherical_irreps_to_cartesian(sphmat: torch.Tensor) -> torch.Tensor:
    """
    Convert spherical tensor components to 3×3 real Cartesian tensors.
    Input: (batch_size, 9)
    Output: (batch_size, 3, 3)
    Safe for autograd (no in-place ops).
    """

    def t(data):
        return torch.tensor(data, dtype=sphmat.dtype, device=sphmat.device)

    #                        xx  xy  xz  yx  yy  yz  zx  zy  zz

    matrix = torch.stack([t([1/3,  0,  0,  0,  1/3,  0,  0,  0,  1/3]),
                          t([0,  0,  0,  0,  0,  1,  0, -1,  0]),
                          t([0,  0,  1,  0,  0,  0, -1,  0,  0]),
                          t([0,  1,  0, -1,  0,  0,  0,  0,  0]),
                          t([0,  1,  0,  1,  0,  0,  0,  0,  0]),
                          t([0,  0,  0,  0,  0,  1,  0,  1,  0]),
                          t([-1, 0,  0,  0, -1,  0,  0,  0,  2]),
                          t([0,  0,  1,  0,  0,  0,  1,  0,  0]),
                          t([1,  0,  0,  0, -1,  0,  0,  0,  0])
                          ],
                         dim=0
                         )

    inv_matrix = torch.inverse(matrix)

    return torch.matmul(inv_matrix, sphmat)
