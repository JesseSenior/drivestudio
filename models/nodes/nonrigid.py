"""
NonRigidNodes: Non-rigid deformable nodes representation

Combines DeformableGaussians deformation with RigidNodes multi-instance management.
This module is designed for representing non-rigid objects (e.g., humans, deformable clothing)
with per-instance rigid poses and global deformable transformations.

Architecture:
- Inherits from DeformableGaussians for deformation network and temporal handling
- Adds instance management from RigidNodes (per-frame poses, visibility masks, point IDs)
- Supports multiple instances with shared deformation network but instance-specific rigid poses
"""

import logging
from typing import Dict, List, Tuple

import torch
from gsplat.cuda._wrapper import spherical_harmonics
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
from gsplat.cuda_legacy._wrapper import num_sh_bases
from pytorch3d.transforms import matrix_to_quaternion
from torch.nn import Parameter

from models.gaussians.basics import (
    RGB2SH,
    dataclass_camera,
    dup_in_optim,
    interpolate_quats,
    k_nearest_sklearn,
    quat_mult,
    random_quat_tensor,
    remove_from_optim,
)
from models.gaussians.deformgs import DeformableGaussians

logger = logging.getLogger()


class NonRigidNodes(DeformableGaussians):
    """
    Non-rigid deformable nodes with multi-instance support.

    This class combines:
    1. DeformableGaussians: Per-point deformation based on time
    2. RigidNodes: Multi-instance support with per-frame poses and visibility masks

    Each instance has:
    - Canonical point set (_means)
    - Per-frame rigid pose (rotation + translation)
    - Shared deformation network that produces per-point deformations
    - Per-frame visibility mask

    Rendering order:
    1. Apply deformation: canonical_means + deformation delta
    2. Apply instance rigid pose: rotate and translate deformed means
    3. Apply visibility mask: zero out opacities for invalid instances
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Instance management attributes will be initialized in create_from_pcd
        self.instances_size = None  # (num_instances, 3)
        self.instances_fv = None  # (num_frames, num_instances)
        self.instances_quats = None  # (num_frames, num_instances, 4)
        self.instances_trans = None  # (num_frames, num_instances, 3)
        self.point_ids = None  # (num_points, 1)

    @property
    def num_instances(self):
        return self.instances_fv.shape[1]

    @property
    def num_frames(self):
        return self.instances_fv.shape[0]

    def get_pts_valid_mask(self):
        """Get the mask for valid points in current frame."""
        return self.instances_fv[self.cur_frame][self.point_ids[..., 0]]

    def create_from_pcd(self, instance_pts_dict: Dict[str, torch.Tensor]) -> None:
        """
        Initialize from instance point cloud dictionary.

        Args:
            instance_pts_dict: {
                id_in_dataset: {
                    "class_name": str,
                    "pts": torch.Tensor, (N, 3)
                    "colors": torch.Tensor, (N, 3)
                    "poses": torch.Tensor, (num_frames, 4, 4)
                    "size": torch.Tensor, (3,)
                    "frame_info": torch.Tensor, (num_frames,)
                    "num_pts": int,
                },
            }
        """
        # Collect all instances
        init_means = []
        init_colors = []
        instances_pose = []
        instances_size = []
        instances_fv = []
        point_ids = []

        for id_in_model, (id_in_dataset, v) in enumerate(instance_pts_dict.items()):
            init_means.append(v["pts"])
            init_colors.append(v["colors"])
            instances_pose.append(v["poses"].unsqueeze(1))
            instances_size.append(v["size"])
            instances_fv.append(v["frame_info"].unsqueeze(1))
            point_ids.append(torch.full((v["num_pts"], 1), id_in_model, dtype=torch.long))

        init_means = torch.cat(init_means, dim=0).to(self.device)  # (N, 3)
        init_colors = torch.cat(init_colors, dim=0).to(self.device)  # (N, 3)
        instances_pose = torch.cat(instances_pose, dim=1).to(self.device)  # (num_frames, num_instances, 4, 4)
        self.instances_size = torch.stack(instances_size).to(self.device)  # (num_instances, 3)
        self.instances_fv = torch.cat(instances_fv, dim=1).to(self.device)  # (num_frames, num_instances)
        self.point_ids = torch.cat(point_ids, dim=0).to(self.device)

        # Convert poses to quaternions and translations
        instances_quats = self.get_instances_quats(instances_pose)
        instances_trans = instances_pose[..., :3, 3]

        # Initialize Gaussian parameters
        self._means = Parameter(init_means)
        distances, _ = k_nearest_sklearn(self._means.data, 3)
        distances = torch.from_numpy(distances)
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)
        avg_dist = avg_dist.clamp(0.002, 100)
        self._scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        self._quats = Parameter(random_quat_tensor(self.num_points).to(self.device))

        # Initialize instance poses for refinement
        self.instances_quats = Parameter(self.quat_act(instances_quats))  # (num_frames, num_instances, 4)
        self.instances_trans = Parameter(instances_trans)  # (num_frames, num_instances, 3)

        # Initialize spherical harmonics
        dim_sh = num_sh_bases(self.sh_degree)
        fused_color = RGB2SH(init_colors)
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        self._opacities = Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1, device=self.device)))

    def get_instances_quats(self, instances_pose: torch.Tensor) -> torch.Tensor:
        """Convert pose matrices to quaternions for all frames and instances."""
        num_frames = instances_pose.shape[0]
        num_instances = instances_pose.shape[1]
        quats = torch.zeros(num_frames * num_instances, 4, device=self.device)

        poses = instances_pose[..., :3, :3].view(-1, 3, 3)
        valid_mask = self.instances_fv.view(-1)
        _quats = matrix_to_quaternion(poses[valid_mask])
        _quats = self.quat_act(_quats)

        quats[valid_mask] = _quats
        quats[~valid_mask, 0] = 1.0
        return quats.reshape(num_frames, num_instances, 4)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Return parameter groups for optimization."""
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix + "deform_network"] = list(self.deform_network.parameters())
        param_groups[self.class_prefix + "ins_rotation"] = [self.instances_quats]
        param_groups[self.class_prefix + "ins_translation"] = [self.instances_trans]
        return param_groups

    def get_deformation(self, canonical_means: torch.Tensor) -> Tuple:
        """
        Get per-point deformations based on time.

        This uses the global deformation network to compute deformations
        that are applied before the instance-specific rigid poses.
        """
        if not self.defrom_gs:
            return None, None, None

        t = self.normalized_timestamps[self.cur_frame]
        t = t.unsqueeze(0).repeat(self.num_points, 1)
        normed_canonical_means = self.contract(canonical_means, self.bbox)

        ast_noise = (
            torch.randn(1, 1, device=self.device).expand(self.num_points, -1)
            * self.time_interval
            * self.smooth_term(self.step)
        )

        delta_xyz, delta_quat, delta_scale = self.deform_network(normed_canonical_means.data, t + ast_noise)
        return delta_xyz, delta_quat, delta_scale

    def contract(self, x: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
        """Contract coordinates to unit cube using piecewise projective function."""
        from models.gaussians.deformgs import contract

        return contract(x, aabb)

    def transform_means(self, means: torch.Tensor) -> torch.Tensor:
        """Transform means to world space applying instance rigid pose."""
        assert means.shape[0] == self.point_ids.shape[0]

        if self.in_test_set and (0 < self.cur_frame - 1 and self.cur_frame + 1 < self.num_frames):
            # Interpolate poses for smoother rendering in test set
            _quats_prev = self.instances_quats[self.cur_frame - 1]
            _quats_next = self.instances_quats[self.cur_frame + 1]
            _quats_cur = self.instances_quats[self.cur_frame]
            interpolated_quats = interpolate_quats(_quats_prev, _quats_next)

            inter_valid_mask = self.instances_fv[self.cur_frame - 1] & self.instances_fv[self.cur_frame + 1]
            quats_cur_frame = torch.where(inter_valid_mask[:, None], interpolated_quats, _quats_cur)
        else:
            quats_cur_frame = self.instances_quats[self.cur_frame]

        rot_cur_frame = quat_to_rotmat(self.quat_act(quats_cur_frame))
        rot_per_pts = rot_cur_frame[self.point_ids[..., 0]]

        if self.in_test_set and (0 < self.cur_frame - 1 and self.cur_frame + 1 < self.num_frames):
            _prev_trans = self.instances_trans[self.cur_frame - 1]
            _next_trans = self.instances_trans[self.cur_frame + 1]
            _cur_trans = self.instances_trans[self.cur_frame]
            interpolated_trans = (_prev_trans + _next_trans) * 0.5

            inter_valid_mask = self.instances_fv[self.cur_frame - 1] & self.instances_fv[self.cur_frame + 1]
            trans_cur_frame = torch.where(inter_valid_mask[:, None], interpolated_trans, _cur_trans)
        else:
            trans_cur_frame = self.instances_trans[self.cur_frame]

        trans_per_pts = trans_cur_frame[self.point_ids[..., 0]]

        # Apply rotation and translation
        means = torch.bmm(rot_per_pts, means.unsqueeze(-1)).squeeze(-1) + trans_per_pts
        return means

    def transform_quats(self, quats: torch.Tensor) -> torch.Tensor:
        """Transform quaternions to world space."""
        assert quats.shape[0] == self.point_ids.shape[0]
        global_quats_cur_frame = self.instances_quats[self.cur_frame]
        global_quats_per_pts = global_quats_cur_frame[self.point_ids[..., 0]]

        global_quats_per_pts = self.quat_act(global_quats_per_pts)
        _quats = self.quat_act(quats)
        return quat_mult(global_quats_per_pts, _quats)

    def get_gaussians(self, cam: dataclass_camera) -> Dict[str, torch.Tensor]:
        """Get Gaussian properties for rendering."""
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask

        # Apply deformations
        delta_xyz, delta_quat, delta_scale = None, None, None
        if self.defrom_gs:
            delta_xyz, delta_quat, delta_scale = self.get_deformation(self._means)
            if self.delta_xyz_rescale:
                delta_xyz = delta_xyz * self.scene_scale

        # Apply deformation to means
        if delta_xyz is not None:
            deformed_means = self._means + delta_xyz
        else:
            deformed_means = self._means

        # Transform to world space with instance poses
        world_means = self.transform_means(deformed_means)

        # Apply deformation to quaternions
        if delta_quat is not None:
            quats = self.get_quats + delta_quat
        else:
            quats = self.get_quats
        world_quats = self.transform_quats(quats)

        # Apply deformation to scales
        if delta_scale is not None:
            activated_scales = torch.exp(self._scales + delta_scale)
        else:
            activated_scales = torch.exp(self._scales)

        # Compute colors using spherical harmonics
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])

        # Apply instance validity mask
        valid_mask = self.get_pts_valid_mask()

        activated_opacities = self.get_opacity * valid_mask.float().unsqueeze(-1)
        activated_rotations = self.quat_act(world_quats)
        actovated_colors = rgbs

        # Collect gaussians
        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )

        # Check for NaN and Inf
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")

        self._gs_cache = {
            "_scales": activated_scales[filter_mask],
        }
        return gs_dict

    def split_gaussians(self, split_mask: torch.Tensor, samps: int) -> Tuple:
        """
        Override split_gaussians to include point_ids
        """
        # Get split results from parent class
        (
            split_means,
            split_feature_dc,
            split_feature_rest,
            split_opacities,
            split_scales,
            split_quats,
        ) = super().split_gaussians(split_mask, samps)

        # Compute split_ids: repeat the ids of split gaussians
        split_ids = self.point_ids[split_mask].repeat(samps, 1)

        return (
            split_means,
            split_feature_dc,
            split_feature_rest,
            split_opacities,
            split_scales,
            split_quats,
            split_ids,
        )

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        Override dup_gaussians to include point_ids
        """
        # Get dup results from parent class
        (
            dup_means,
            dup_feature_dc,
            dup_feature_rest,
            dup_opacities,
            dup_scales,
            dup_quats,
        ) = super().dup_gaussians(dup_mask)

        # Compute dup_ids: use the ids of duplicated gaussians
        dup_ids = self.point_ids[dup_mask]

        return (
            dup_means,
            dup_feature_dc,
            dup_feature_rest,
            dup_opacities,
            dup_scales,
            dup_quats,
            dup_ids,
        )

    def refinement_after(self, step: int, optimizer: torch.optim.Optimizer) -> None:
        """Densification and pruning after step."""
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return

        with torch.no_grad():
            reset_interval = self.ctrl_cfg.reset_alpha_interval
            do_densification = self.step < self.ctrl_cfg.stop_split_at and self.step % reset_interval > max(
                self.num_train_images, self.ctrl_cfg.refine_interval
            )

            print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")
            if do_densification:
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None

                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()

                splits = (
                    self.get_scaling.max(dim=-1).values > self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                if self.step < self.ctrl_cfg.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                splits &= high_grads

                nsamps = self.ctrl_cfg.n_split_samples
                (
                    split_means,
                    split_feature_dc,
                    split_feature_rest,
                    split_opacities,
                    split_scales,
                    split_quats,
                    split_ids,
                ) = self.split_gaussians(splits, nsamps)

                dups = (
                    self.get_scaling.max(dim=-1).values <= self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                dups &= high_grads
                (
                    dup_means,
                    dup_feature_dc,
                    dup_feature_rest,
                    dup_opacities,
                    dup_scales,
                    dup_quats,
                    dup_ids,
                ) = self.dup_gaussians(dups)

                self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0))
                self._features_dc = Parameter(
                    torch.cat([self._features_dc.detach(), split_feature_dc, dup_feature_dc], dim=0)
                )
                self._features_rest = Parameter(
                    torch.cat([self._features_rest.detach(), split_feature_rest, dup_feature_rest], dim=0)
                )
                self._opacities = Parameter(
                    torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0)
                )
                self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0))
                self.point_ids = torch.cat([self.point_ids, split_ids, dup_ids], dim=0)

                # Append zeros to max_2Dsize
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_scales[:, 0]),
                        torch.zeros_like(dup_scales[:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, split_idcs, param_groups, n=nsamps)

                dup_idcs = torch.where(dups)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, dup_idcs, param_groups, 1)

            # Culling
            if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                deleted_mask = self.cull_gaussians()
                param_groups = self.get_gaussian_param_groups()
                remove_from_optim(optimizer, deleted_mask, param_groups)

            print(f"Class {self.class_prefix} left points: {self.num_points}")

            # Reset opacity
            if self.step % reset_interval == self.ctrl_cfg.refine_interval:
                reset_value = torch.min(
                    self.get_opacity.data,
                    torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value,
                )
                self._opacities.data = torch.logit(reset_value)
                for group in optimizer.param_groups:
                    if group["name"] == self.class_prefix + "opacity":
                        old_params = group["params"][0]
                        param_state = optimizer.state[old_params]
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self):
        """Remove Gaussians with low opacity or out of bounds."""
        n_bef = self.num_points
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()

        if self.ctrl_cfg.get("cull_out_of_bound", False):
            culls = culls | self.get_out_of_bound_mask()

        if self.step > self.ctrl_cfg.reset_alpha_interval:
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values > self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()

        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        self.point_ids = self.point_ids[~culls]

        print(f"     Cull: {n_bef - self.num_points}")
        return culls

    def get_out_of_bound_mask(self):
        """Check if gaussians are out of instance bounding boxes."""
        per_pts_size = self.instances_size[self.point_ids[..., 0]]
        instance_pts = self._means

        mask = (instance_pts.abs() > per_pts_size / 2).any(dim=-1)
        return mask

    def state_dict(self) -> Dict:
        """Get state dictionary for saving."""
        state_dict = super().state_dict()
        state_dict.update(
            {
                "point_ids": self.point_ids,
                "instances_size": self.instances_size,
                "instances_fv": self.instances_fv,
            }
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        """Load state dictionary."""
        self.point_ids = state_dict.pop("point_ids")
        self.instances_size = state_dict.pop("instances_size")
        self.instances_fv = state_dict.pop("instances_fv")
        self.instances_trans = Parameter(torch.zeros(self.num_frames, self.num_instances, 3, device=self.device))
        self.instances_quats = Parameter(torch.zeros(self.num_frames, self.num_instances, 4, device=self.device))
        msg = super().load_state_dict(state_dict, **kwargs)
        return msg
