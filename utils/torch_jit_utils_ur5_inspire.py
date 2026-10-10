# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch

def quat_mul(a, b):
    av, bv = a[..., :3], b[..., :3]
    return torch.cat((a[..., 3:]*bv + b[..., 3:]*av + torch.cross(av, bv, dim=-1),
                      a[..., 3:]*b[..., 3:] - (av*bv).sum(-1, keepdim=True)), dim=-1)

def quat_axis(q, axis):
    return quat_to_mat(q)[..., :, axis]

@torch.jit.script
def quat_to_mat(q):
    """
    Convert rotations given as quaternions to rotation matrices. From pytorch3d.

    Args:
        quaternions: quaternions with real part last (xyzw),
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(q, -1)
    two_s = 2.0 / (q * q).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))

@torch.jit.script
def mat_diff_rad(m1, m2):
    diff_mat = torch.matmul(m1.transpose(-1,-2), m2)
    diff_rad = torch.acos(torch.clamp((diff_mat[..., 0,0] + diff_mat[..., 1,1] + diff_mat[..., 2,2] - 1) / 2, -1, 1))
    return diff_rad

@torch.jit.script
def deambiguity_rotation(old_r):
    standard_r_mat = torch.eye(3).to(old_r.device)
    old_r_mat = quat_to_mat(old_r) # N, 3, 3
    ind = torch.tensor([[0,1],[0,2],[1,2],[1,0],[2,0],[2,1]], device=old_r.device) # 6, 2
    ind = torch.cat([ind, ind, ind, ind], dim=0) # 24, 2
    all_r_mat_12 = old_r_mat[:, :, ind].transpose(-2,-3)
    all_r_mat_12[:, :12, 0] = -all_r_mat_12[:, :12, 0]
    all_r_mat_12[:, 6:18, 1] = -all_r_mat_12[:, 6:18, 1]
    all_r_mat_3 = torch.cross(all_r_mat_12[...,0],all_r_mat_12[...,1], dim=-1).unsqueeze(-1)
    all_r_mat = torch.cat([all_r_mat_12, all_r_mat_3], dim=-1)  # N, 24, 3, 3 
    diff_rad = mat_diff_rad(all_r_mat, standard_r_mat[None,None])
    min_ind = diff_rad.argmin(dim=1)
    new_r = all_r_mat[torch.arange(all_r_mat.shape[0], device=old_r.device), min_ind]
    return new_r

