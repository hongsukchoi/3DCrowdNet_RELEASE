import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module import Pose2Feat, PositionNet, RotationNet, Vposer
from nets.loss import CoordLoss, ParamLoss, NormalVectorLoss, EdgeLengthLoss
from utils.smpl import SMPL
from utils.mano import MANO
from config import cfg
from contextlib import nullcontext
import math

from utils.transforms import rot6d_to_axis_angle


class Model(nn.Module):
    def __init__(self, backbone, pose2feat, position_net, rotation_net, vposer):
        super(Model, self).__init__()
        self.backbone = backbone
        self.pose2feat = pose2feat
        self.position_net = position_net
        self.rotation_net = rotation_net
        self.vposer = vposer

        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.human_model = MANO()
            self.human_model_layer = self.human_model.layer.cuda()
        else:
            self.human_model = SMPL()
            self.human_model_layer = self.human_model.layer['neutral'].cuda()
        self.root_joint_idx = self.human_model.root_joint_idx
        self.mesh_face = self.human_model.face
        self.joint_regressor = self.human_model.joint_regressor

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()

    def get_camera_trans(self, cam_param, meta_info, is_render):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0]*cfg.focal[1]*cfg.camera_3d_size*cfg.camera_3d_size/(cfg.input_img_shape[0]*cfg.input_img_shape[1]))]).cuda().view(-1)
        if is_render:
            bbox = meta_info['bbox']
            k_value = k_value * math.sqrt(cfg.input_img_shape[0]*cfg.input_img_shape[1]) / (bbox[:, 2]*bbox[:, 3]).sqrt()
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        return cam_trans

    def make_2d_gaussian_heatmap(self, joint_coord_img):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        yy, xx = torch.meshgrid(y, x)
        xx = xx[None, None, :, :].cuda().float();
        yy = yy[None, None, :, :].cuda().float();

        x = joint_coord_img[:, :, 0, None, None];
        y = joint_coord_img[:, :, 1, None, None];
        heatmap = torch.exp(
            -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2)
        return heatmap

    def get_coord(self, smpl_pose, smpl_shape, smpl_trans):
        batch_size = smpl_pose.shape[0]
        mesh_cam, _ = self.human_model_layer(smpl_pose, smpl_shape, smpl_trans)
        # camera-centered 3D coordinate
        joint_cam = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)
        root_joint_idx = self.human_model.root_joint_idx

        # project 3D coordinates to 2D space
        x = joint_cam[:,:,0] / (joint_cam[:,:,2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]
        y = joint_cam[:,:,1] / (joint_cam[:,:,2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x,y),2)

        mesh_cam_render = mesh_cam.clone()
        # root-relative 3D coordinates
        root_cam = joint_cam[:,root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam - root_cam
        return joint_proj, joint_cam, mesh_cam, mesh_cam_render

    def forward(self, inputs, targets, meta_info, mode):
        early_img_feat = self.backbone(inputs['img'])  #pose_guided_img_feat

        # get pose gauided image feature
        joint_coord_img = inputs['joints']
        with torch.no_grad():
            joint_heatmap = self.make_2d_gaussian_heatmap(joint_coord_img.detach())
            # remove blob centered at (0,0) == invalid ones
            joint_heatmap = joint_heatmap * inputs['joints_mask'][:,:,:,None]
        pose_img_feat = self.pose2feat(early_img_feat, joint_heatmap)
        pose_guided_img_feat = self.backbone(pose_img_feat, skip_early=True)  # 2048 x 8 x 8

        joint_img, joint_score = self.position_net(pose_guided_img_feat)  # refined 2D pose or 3D pose

        # estimate model parameters
        root_pose_6d, z, shape_param, cam_param = self.rotation_net(pose_guided_img_feat, joint_img.detach(), joint_score.detach())
        # change root pose 6d + latent code -> axis angles
        root_pose = rot6d_to_axis_angle(root_pose_6d)
        pose_param = self.vposer(z)
        cam_trans = self.get_camera_trans(cam_param, meta_info, is_render=(cfg.render and (mode == 'test')))
        pose_param = pose_param.view(-1, self.human_model.orig_joint_num - 1, 3)
        pose_param = torch.cat((root_pose[:, None, :], pose_param), 1).view(-1, self.human_model.orig_joint_num * 3)
        joint_proj, joint_cam, mesh_cam, mesh_cam_render = self.get_coord(pose_param, shape_param, cam_trans)

        if mode == 'train':
            # loss functions
            loss = {}
            # joint_img: 0~8, joint_proj: 0~64, target: 0~64
            loss['body_joint_img'] = (1/8) * self.coord_loss(joint_img*8, self.human_model.reduce_joint_set(targets['orig_joint_img']), self.human_model.reduce_joint_set(meta_info['orig_joint_trunc']), meta_info['is_3D'])
            loss['smpl_joint_img'] = (1/8) * self.coord_loss(joint_img*8, self.human_model.reduce_joint_set(targets['fit_joint_img']),
                                                     self.human_model.reduce_joint_set(meta_info['fit_joint_trunc']) * meta_info['is_valid_fit'][:, None, None])
            loss['smpl_pose'] = self.param_loss(pose_param, targets['pose_param'], meta_info['fit_param_valid'] * meta_info['is_valid_fit'][:, None])
            loss['smpl_shape'] = self.param_loss(shape_param, targets['shape_param'], meta_info['is_valid_fit'][:, None])
            loss['body_joint_proj'] = (1/8) * self.coord_loss(joint_proj, targets['orig_joint_img'][:, :, :2], meta_info['orig_joint_trunc'])
            loss['body_joint_cam'] = self.coord_loss(joint_cam, targets['orig_joint_cam'], meta_info['orig_joint_valid'] * meta_info['is_3D'][:, None, None])
            loss['smpl_joint_cam'] = self.coord_loss(joint_cam, targets['fit_joint_cam'], meta_info['is_valid_fit'][:, None, None])

            return loss

        else:
            # test output
            out = {'cam_param': cam_param}
            # out['input_joints'] = joint_coord_img
            out['joint_img'] = joint_img * 8
            out['joint_proj'] = joint_proj
            out['joint_score'] = joint_score
            out['smpl_mesh_cam'] = mesh_cam
            out['smpl_pose'] = pose_param
            out['smpl_shape'] = shape_param

            out['mesh_cam_render'] = mesh_cam_render

            if 'smpl_mesh_cam' in targets:
                out['smpl_mesh_cam_target'] = targets['smpl_mesh_cam']
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if 'img2bb_trans' in meta_info:
                out['img2bb_trans'] = meta_info['img2bb_trans']
            if 'bbox' in meta_info:
                out['bbox'] = meta_info['bbox']
            if 'tight_bbox' in meta_info:
                out['tight_bbox'] = meta_info['tight_bbox']
            if 'aid' in meta_info:
                out['aid'] = meta_info['aid']

            return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(vertex_num, joint_num, mode):

    backbone = ResNetBackbone(cfg.resnet_type)
    pose2feat = Pose2Feat(joint_num)
    position_net = PositionNet()
    rotation_net = RotationNet()
    vposer = Vposer()

    if mode == 'train':
        backbone.init_weights()
        pose2feat.apply(init_weights)
        position_net.apply(init_weights)
        rotation_net.apply(init_weights)
   
    model = Model(backbone, pose2feat, position_net, rotation_net, vposer)
    return model

