#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
	def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
				 image_name, uid,
				 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
				 train_test_exp = False, is_test_dataset = False, is_test_view = False
				 ):
		super(Camera, self).__init__()

		self.uid = uid
		self.colmap_id = colmap_id
		self.R = R
		self.T = T
		self.FoVx = FoVx
		self.FoVy = FoVy
		self.image_name = image_name

		try:
			self.data_device = torch.device(data_device)
		except Exception as e:
			print(e)
			print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
			self.data_device = torch.device("cuda")

		resized_image_rgb = PILtoTorch(image, resolution)
		gt_image = resized_image_rgb[:3, ...]
		self.alpha_mask = None
		if resized_image_rgb.shape[0] == 4:
			self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
		else: 
			self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

		if train_test_exp and is_test_view:
			if is_test_dataset:
				self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
			else:
				self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

		self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
		self.image_width = self.original_image.shape[2]
		self.image_height = self.original_image.shape[1]

		self.invdepthmap = None
		self.depth_reliable = False
		if invdepthmap is not None:
			self.depth_mask = torch.ones_like(self.alpha_mask)
			self.invdepthmap = cv2.resize(invdepthmap, resolution)
			self.invdepthmap[self.invdepthmap < 0] = 0
			self.depth_reliable = True

			if depth_params is not None:
				if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
					self.depth_reliable = False
					self.depth_mask *= 0
				
				if depth_params["scale"] > 0:
					self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

			if self.invdepthmap.ndim != 2:
				self.invdepthmap = self.invdepthmap[..., 0]
			self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

		self.zfar = 100.0
		self.znear = 0.01

		self.trans = trans
		self.scale = scale

		self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
		self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
		self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
		self.camera_center = self.world_view_transform.inverse()[3, :3]
		
class MiniCam:
	def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
		self.image_width = width
		self.image_height = height    
		self.FoVy = fovy
		self.FoVx = fovx
		self.znear = znear
		self.zfar = zfar
		self.world_view_transform = world_view_transform
		self.full_proj_transform = full_proj_transform
		view_inv = torch.inverse(self.world_view_transform)
		self.camera_center = view_inv[3][:3]

class InteractiveCam:
	def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform=None, projection_matrix = None):
		self.image_width = width
		self.image_height = height    
		self.FoVy = fovy
		self.FoVx = fovx
		self.znear = znear
		self.zfar = zfar
		
		if world_view_transform is None:
			world_view_transform = torch.eye(4).cuda()
		if projection_matrix is None:
			projection_matrix = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0,1).cuda()
		full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0) # komischer Code mit unsqueeze bmm und squeeze ...

		self.world_view_transform = world_view_transform
		self.projection_matrix = projection_matrix
		self.full_proj_transform = full_proj_transform
		view_inv = torch.inverse(self.world_view_transform)
		self.camera_center = view_inv[3][:3]
	
	def translate(self, dx=torch.zeros(1), dy=torch.zeros(1), dz=torch.zeros(1)):
		"""Translate camera in local coordinates"""
		t = torch.eye(4, device=self.world_view_transform.device)
		t[0,3] = dx
		t[1,3] = dy
		t[2,3] = dz
		self.world_view_transform = self.world_view_transform @ t.transpose(1,0)
		self.full_proj_transform = self.world_view_transform @ self.projection_matrix
		view_inv = torch.inverse(self.world_view_transform)
		self.camera_center = view_inv[3,:3].clone()
	
	def rotate(self, dyaw=torch.zeros(1), dpitch=torch.zeros(1), droll=torch.zeros(1)):
		"""Rotate camera around its local axes"""
		Rx = torch.tensor([
			[1,0,0,0],
			[0,torch.cos(dpitch), -torch.sin(dpitch), 0],
			[0,torch.sin(dpitch), torch.cos(dpitch),0],
			[0,0,0,1]
		], device=self.world_view_transform.device)
		Ry = torch.tensor([
			[torch.cos(dyaw),0,torch.sin(dyaw),0],
			[0,1,0,0],
			[-torch.sin(dyaw),0,torch.cos(dyaw),0],
			[0,0,0,1]
		], device=self.world_view_transform.device)
		Rz = torch.tensor([
			[torch.cos(droll), -torch.sin(droll),0,0],
			[torch.sin(droll), torch.cos(droll),0,0],
			[0,0,1,0],
			[0,0,0,1]
		], device=self.world_view_transform.device)
		self.world_view_transform = self.world_view_transform @ Rz @ Ry @ Rx
		self.full_proj_transform = self.world_view_transform @ self.projection_matrix
		
	def look_at(self, target, up=torch.FloatTensor([0.0, 0.0, 1.0]).cuda()):
		"""Set camera to look at target point"""
		# TODO: funktioniert noch nicht
		position = self.camera_center
		up = up/torch.norm(up)
		forward = (target - position)
		forward = forward / torch.norm(forward)
		right = torch.cross(up, forward)
		right = right / torch.norm(right)
		true_up = torch.cross(forward, right)
		R = torch.eye(4).cuda()
		R[0,:3] = right
		R[1,:3] = true_up
		R[2,:3] = forward
		R[3,:3] = position.clone()
		self.world_view_transform = torch.inverse(R)
		self.full_proj_transform = self.world_view_transform @ self.projection_matrix
		view_inv = torch.inverse(self.world_view_transform)
		self.camera_center = view_inv[3,:3].clone()
		 
	def set_position(self, position):
		"""Set camera position in world coordinates"""
		view_inv = torch.inverse(self.world_view_transform)
		view_inv[3,:3] = position
		self.world_view_transform = torch.inverse(view_inv)
		self.full_proj_transform = self.world_view_transform @ self.projection_matrix
		self.camera_center = position.clone()
	
	def get_pose(self):
		R = torch.inverse(self.world_view_transform)
		right = R[0,:3]
		up = R[1,:3]
		forward = R[2,:3]
		position = R[3,:3]
		print(f"position: {position} / right: {right} / up: {up} / forward: {forward}")
		return right,up,forward,position