import torch
from torch import nn
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from e3nn.o3 import wigner_3j


@compile_mode("script")
class ContractProduct3j(nn.Module):
    def __init__(self,
                 irreps_in1,
                 irreps_in2,
                 irreps_target
                 ):
        super().__init__()
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_target = irreps_target

    def extractl(self, x: torch.Tensor, irreps: Irreps, l_target: int) -> torch.Tensor:
        """
        Extract components from a tensor with given irreps corresponding to angular momentum l_target.

        Args:
            x: tensor of shape (..., irreps.dim)
            irreps: e3nn.o3.Irreps
            l_target: integer angular momentum l

        Returns:
            Tensor of shape (..., selected_dim), containing only components with l = l_target
        """
        out = []
        offset = 0
        for mul, ir in irreps:
            dim = mul * ir.dim
            if ir.l == l_target:
                out.append(x[..., offset:offset + dim])
            offset += dim
        if not out:
            raise ValueError(f"No irreps with l = {l_target} found in {irreps}")
        return torch.cat(out, dim=-1)  # nodes, nchannels, 2l + 1

    def forward(self,
                tensor_1,
                tensor_2
                ):
        # tensors are of shape [node, nchannels, irreps]

        device = tensor_1.device

        nodes, channels, _ = tensor_2.shape

        out = torch.zeros((nodes, channels, self.irreps_target.dim), device=device)

        dims = []

        for mul3_, (mul3, ir3) in enumerate(self.irreps_target):
            dims.append(ir3.dim)

        offset = [0]

        for idx, dim in enumerate(dims[:-1]):
            offset.append(offset[idx] + dim)

        for mul1_, (mul1, ir1) in enumerate(self.irreps_in1):
            for mul2_, (mul2, ir2) in enumerate(self.irreps_in2):
                for ir3 in ir1 * ir2:
                    if ir3 not in self.irreps_target:
                        continue
                    w123 = wigner_3j(ir1.l, ir2.l, ir3.l).to(device)  # with shape [2l_1+1, 2l_2+1, 2l_3+1]
                    tir1 = self.extractl(tensor_1, self.irreps_in1, ir1.l)
                    tir2 = self.extractl(tensor_2, self.irreps_in2, ir2.l)

                    out[..., offset[ir3.l]:int(2*ir3.l + 1 + offset[ir3.l])] += torch.einsum('ncl, ncm, lms -> ncs',
                                                                                             tir1,
                                                                                             tir2,
                                                                                             w123
                                                                                             )

        return out  # tensor is of shape [node, nchannels, irreps]
