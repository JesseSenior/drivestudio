"""
NonRigidNodes: Non-rigid deformable nodes representation

Combines FreeTimeGS temporal Gaussian representation with RigidNodes multi-instance management.
This module is designed for representing non-rigid objects (e.g., humans, deformable clothing)
with per-instance rigid poses and global temporal dynamics.

Architecture:
- Inherits from VanillaGaussians for base Gaussian management
- Adds instance management from RigidNodes (per-frame poses, visibility masks, point IDs)
- Implements FreeTimeGS temporal representation (xyz_t, scaling_t, velocities)
- Supports multiple instances with instance-specific rigid poses and shared temporal parameters
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
from models.gaussians.vanilla import VanillaGaussians

logger = logging.getLogger()


class NonRigidNodes(VanillaGaussians):
    """
    Non-rigid deformable nodes with multi-instance support.

    This class combines:
    1. FreeTimeGS temporal representation: Per-point temporal filtering and motion compensation
    2. Multi-instance support: Per-frame rigid poses and visibility masks

    Each point has:
    - Canonical position in object space (_means)
    - Temporal center and scale (_xyz_t, _scaling_t) for time filtering
    - Velocity vector (_velocities) for motion compensation
    - Instance ID (point_ids) mapping to instance pose and visibility

    Rendering pipeline:
    1. Apply temporal filtering and motion compensation based on FreeTimeGS parameters
    2. Transform to world space using instance rigid poses (rotation + translation)
    3. Apply instance visibility mask and temporal opacity modulation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Instance management attributes will be initialized in create_from_pcd
        self.instances_size = None  # (num_instances, 3)
        self.instances_fv = None  # (num_frames, num_instances)
        self.instances_quats = None  # (num_frames, num_instances, 4)
        self.instances_trans = None  # (num_frames, num_instances, 3)
        self.point_ids = None  # (num_points, 1)

        # FreeTimeGS temporal parameters
        self._xyz_t = None  # (num_points, 1) - temporal center for each point
        self._scaling_t = None  # (num_points, 1) - temporal scale (log space) for each point
        self._velocities = None  # (num_points, 3) - 3D velocity vector for motion compensation
        self.marginal_th = 0.05  # threshold for temporal filtering
        self.duration = 1.0  # total video duration in normalized time

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

        # Initialize FreeTimeGS temporal parameters
        # Initialize temporal centers and scales based on each instance's trajectory time range
        self._xyz_t = torch.zeros(self.num_points, 1, device=self.device)
        self._scaling_t = torch.zeros(self.num_points, 1, device=self.device)

        # For each instance, compute its valid time range and initialize temporal parameters
        point_idx = 0
        num_frames_total = instance_pts_dict[list(instance_pts_dict.keys())[0]]["frame_info"].shape[0]

        for id_in_model, (id_in_dataset, v) in enumerate(instance_pts_dict.items()):
            num_pts = v["num_pts"]
            frame_info = v["frame_info"]

            # Get valid frames for this instance
            valid_frames = torch.where(frame_info)[0]
            if len(valid_frames) > 0:
                # Compute time range: normalize frame indices to [0, 1]
                t_start = valid_frames[0].float() / num_frames_total
                t_end = valid_frames[-1].float() / num_frames_total
                duration_instance = t_end - t_start

                # Randomly initialize temporal centers within the valid time range
                t_centers = torch.rand(num_pts, 1, device=self.device) * duration_instance + t_start
                self._xyz_t[point_idx : point_idx + num_pts] = t_centers

                # Set temporal scale: use average frame interval
                num_valid_frames = len(valid_frames)
                if num_valid_frames > 1:
                    # Average frame interval = duration / (num_frames - 1)
                    span = duration_instance / (num_valid_frames - 1)
                else:
                    # Fallback for single frame
                    span = 0.5 * self.duration
                scale_t_mult = 1.0
                sigma_t = torch.sqrt(
                    torch.tensor((span * scale_t_mult) ** 2 / (-2 * torch.log(torch.tensor(self.marginal_th))))
                )
                log_sigma_t = torch.log(torch.sqrt(sigma_t) * torch.ones(num_pts, 1, device=self.device))
                self._scaling_t[point_idx : point_idx + num_pts] = log_sigma_t
            else:
                # Fallback for instances with no valid frames
                self._xyz_t[point_idx : point_idx + num_pts] = 0
                span = 0.5 * self.duration
                scale_t_mult = 1.0
                sigma_t = torch.sqrt(
                    torch.tensor((span * scale_t_mult) ** 2 / (-2 * torch.log(torch.tensor(self.marginal_th))))
                )
                log_sigma_t = torch.log(torch.sqrt(sigma_t) * torch.ones(num_pts, 1, device=self.device))
                self._scaling_t[point_idx : point_idx + num_pts] = log_sigma_t

            point_idx += num_pts

        self._xyz_t = Parameter(self._xyz_t)
        self._scaling_t = Parameter(self._scaling_t)

        # Velocity vector: initialize to zero (will be learned during training)
        self._velocities = Parameter(torch.zeros(self.num_points, 3, device=self.device))

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
        param_groups[self.class_prefix + "ins_rotation"] = [self.instances_quats]
        param_groups[self.class_prefix + "ins_translation"] = [self.instances_trans]

        # Add FreeTimeGS temporal parameters to optimization
        param_groups[self.class_prefix + "xyz_t"] = [self._xyz_t]
        param_groups[self.class_prefix + "scaling_t"] = [self._scaling_t]
        param_groups[self.class_prefix + "velocities"] = [self._velocities]

        return param_groups

    def get_deformation(self) -> Tuple:
        """
        Get per-point deformations based on FreeTimeGS temporal representation.

        This implements FreeTimeGS temporal filtering and motion compensation:
        1. Temporal filtering: compute marginal probability based on temporal distance
        2. Motion compensation: apply velocity-based displacement
        3. Opacity modulation: return temporal weight for opacity

        Returns:
            (delta_xyz, delta_opacities) where:
            - delta_xyz: motion displacement (num_points, 3)
            - delta_opacities: temporal opacity weight, marginal_t (num_points,)
        """
        t_query = self.normalized_timestamps[self.cur_frame]

        # ============= Temporal Filtering =============
        # Compute temporal distance from each point's temporal center
        # marginal_t = exp(-0.5 * ((t_query - t_p) / sigma_t)^2)
        delta_t = t_query - self._xyz_t.squeeze(-1)  # (num_points,)
        sigma_t = torch.exp(self._scaling_t.squeeze(-1))  # (num_points,) - activate from log space

        # Compute marginal temporal probability (used for opacity modulation)
        marginal_t = torch.exp(-0.5 * (delta_t / (sigma_t + 1e-8)) ** 2)  # (num_points,)

        # Apply temporal threshold to filter out points with low temporal probability
        marginal_t = torch.where(marginal_t > self.marginal_th, marginal_t, torch.zeros_like(marginal_t))

        # ============= Motion Compensation =============
        # Apply velocity-based motion compensation: x_deformed = x + v * Δt
        # Δt = t_query - t_p (same as delta_t)
        delta_xyz = self._velocities * delta_t.unsqueeze(-1)  # (num_points, 3)

        # ============= Opacity Modulation =============
        # delta_opacities is the marginal_t weight that modulates opacity
        # Final opacity = sigmoid(raw_opacity) * delta_opacities (marginal_t)
        delta_opacities = marginal_t  # (num_points,)

        return delta_xyz, delta_opacities

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

    def set_cur_frame(self, frame_id: int):
        self.cur_frame = frame_id

    def register_normalized_timestamps(self, normalized_timestamps: int):
        self.normalized_timestamps = normalized_timestamps
        self.time_interval = 1 / len(normalized_timestamps)

    def get_gaussians(self, cam: dataclass_camera) -> Dict[str, torch.Tensor]:
        """Get Gaussian properties for rendering.

        Pipeline:
        1. Canonical means in object space
        2. Apply temporal filtering and motion compensation (FreeTimeGS)
        3. Transform to world space with instance rigid poses
        4. Apply instance visibility mask and opacity modulation
        """
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask

        # Apply FreeTimeGS temporal filtering and motion compensation
        delta_xyz, delta_opacities = None, None
        if hasattr(self, "normalized_timestamps"):
            delta_xyz, delta_opacities = self.get_deformation()

        # Apply motion compensation to means
        if delta_xyz is not None:
            deformed_means = self._means + delta_xyz
        else:
            deformed_means = self._means

        # Transform to world space with instance poses
        world_means = self.transform_means(deformed_means)

        # Get quaternions and scales (no deformation on these)
        world_quats = self.transform_quats(self.get_quats)
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

        # Apply instance validity mask and temporal opacity modulation
        instance_valid_mask = self.get_pts_valid_mask()

        # Apply both instance validity and temporal filtering to opacity
        # Final opacity: alpha_final = sigmoid(alpha_raw) * instance_mask * delta_opacities
        opacity_mask = instance_valid_mask.float().unsqueeze(-1)
        if delta_opacities is not None:
            opacity_mask = opacity_mask * delta_opacities.unsqueeze(-1)

        activated_opacities = self.get_opacity * opacity_mask
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
        Override split_gaussians to include point_ids and temporal parameters.
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

        # Split temporal parameters: repeat parent's temporal center and scale
        split_xyz_t = self._xyz_t[split_mask].repeat(samps, 1)
        split_scaling_t = self._scaling_t[split_mask].repeat(samps, 1)
        split_velocities = self._velocities[split_mask].repeat(samps, 1)

        return (
            split_means,
            split_feature_dc,
            split_feature_rest,
            split_opacities,
            split_scales,
            split_quats,
            split_ids,
            split_xyz_t,
            split_scaling_t,
            split_velocities,
        )

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        Override dup_gaussians to include point_ids and temporal parameters.
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

        # Duplicate temporal parameters
        dup_xyz_t = self._xyz_t[dup_mask]
        dup_scaling_t = self._scaling_t[dup_mask]
        dup_velocities = self._velocities[dup_mask]

        return (
            dup_means,
            dup_feature_dc,
            dup_feature_rest,
            dup_opacities,
            dup_scales,
            dup_quats,
            dup_ids,
            dup_xyz_t,
            dup_scaling_t,
            dup_velocities,
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
                    split_xyz_t,
                    split_scaling_t,
                    split_velocities,
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
                    dup_xyz_t,
                    dup_scaling_t,
                    dup_velocities,
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

                # Concatenate temporal parameters
                self._xyz_t = Parameter(torch.cat([self._xyz_t.detach(), split_xyz_t, dup_xyz_t], dim=0))
                self._scaling_t = Parameter(
                    torch.cat([self._scaling_t.detach(), split_scaling_t, dup_scaling_t], dim=0)
                )
                self._velocities = Parameter(
                    torch.cat([self._velocities.detach(), split_velocities, dup_velocities], dim=0)
                )

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

        # Remove temporal parameters for culled gaussians
        self._xyz_t = Parameter(self._xyz_t[~culls].detach())
        self._scaling_t = Parameter(self._scaling_t[~culls].detach())
        self._velocities = Parameter(self._velocities[~culls].detach())

        self.point_ids = self.point_ids[~culls]

        print(f"     Cull: {n_bef - self.num_points}")
        return culls

    def get_out_of_bound_mask(self):
        """Check if gaussians are out of instance bounding boxes."""
        per_pts_size = self.instances_size[self.point_ids[..., 0]]
        instance_pts = self._means

        mask = (instance_pts.abs() > per_pts_size / 2).any(dim=-1)
        return mask

    def compute_reg_loss(self):
        """Compute regularization losses including out-of-bound velocity constraint.

        Extends parent's compute_reg_loss by adding out_of_bound_loss to constrain
        gaussian points' velocities to not exceed the instance bounding boxes.
        """
        # Get parent's regularization losses
        loss_dict = super().compute_reg_loss()

        # Out-of-bound loss: constrain velocity to not exceed bbox
        out_of_bound_cfg = self.reg_cfg.get("out_of_bound", None)
        if out_of_bound_cfg is not None:
            w = out_of_bound_cfg.w
            step_interval = out_of_bound_cfg.get("step_interval", 1)
            velocity_scale = out_of_bound_cfg.get("velocity_scale", 1.0)
            visibility_threshold = out_of_bound_cfg.get("visibility_threshold", 0.2)

            if self.step % step_interval == 0:
                # Get per-point bbox size
                per_pts_size = self.instances_size[self.point_ids[..., 0]]  # (num_points, 3)
                bbox_half_size = per_pts_size / 2  # (num_points, 3)

                # Compute predicted positions after velocity motion
                # new_pos = current_pos + velocity * velocity_scale
                predicted_pos = self._means + self._velocities * velocity_scale  # (num_points, 3)

                # Get gaussian scales (point visibility radius)
                scales = torch.exp(self._scales)  # (num_points, 3)

                # Get temporal visibility (delta_opacity) to filter points
                if hasattr(self, "normalized_timestamps"):
                    _, delta_opacities = self.get_deformation()  # (num_points,)
                    # Only consider points with sufficient temporal visibility
                    visible_mask = delta_opacities >= visibility_threshold
                else:
                    visible_mask = torch.ones(self.num_points, dtype=torch.bool, device=self.device)

                # Compute gaussian visibility bounds after motion
                # The visible range is [predicted_pos - scales, predicted_pos + scales]
                lower_bound = predicted_pos - scales  # (num_points, 3)
                upper_bound = predicted_pos + scales  # (num_points, 3)

                # Check if gaussian visibility range exceeds bbox bounds
                # bbox bounds: [-size/2, size/2]
                lower_violation = torch.clamp(-lower_bound - bbox_half_size, min=0)  # (num_points, 3)
                upper_violation = torch.clamp(upper_bound - bbox_half_size, min=0)  # (num_points, 3)

                # Combined violation (sum across dimensions)
                violation = (lower_violation + upper_violation).sum(dim=-1)  # (num_points,)

                # Apply visibility mask: only penalize visible points
                violation = violation * visible_mask.float()

                # Compute loss as mean of violations
                if visible_mask.sum() > 0:
                    out_of_bound_loss = violation.sum() / visible_mask.sum().float() * w
                else:
                    out_of_bound_loss = torch.tensor(0.0, device=self.device)
                loss_dict["nonrigid_out_of_bound"] = out_of_bound_loss

        return loss_dict

    def state_dict(self) -> Dict:
        """Get state dictionary for saving."""
        state_dict = super().state_dict()
        state_dict.update(
            {
                "point_ids": self.point_ids,
                "instances_size": self.instances_size,
                "instances_fv": self.instances_fv,
                "_xyz_t": self._xyz_t,
                "_scaling_t": self._scaling_t,
                "_velocities": self._velocities,
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
        self._xyz_t = Parameter(torch.zeros_like(state_dict["_xyz_t"]))
        self._scaling_t = Parameter(torch.zeros_like(state_dict["_scaling_t"]))
        self._velocities = Parameter(torch.zeros_like(state_dict["_velocities"]))
        msg = super().load_state_dict(state_dict, **kwargs)
        return msg
