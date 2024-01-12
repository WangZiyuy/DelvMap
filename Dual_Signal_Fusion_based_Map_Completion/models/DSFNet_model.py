from Dual_Signal_Fusion_based_Map_Completion.models.networks import define_translator, init_net
from Dual_Signal_Fusion_based_Map_Completion.models.base_model import BaseModel
from Dual_Signal_Fusion_based_Map_Completion.models.losses import SoftDiceLoss, BCEDiceLoss, BCE_Loss
from Dual_Signal_Fusion_based_Map_Completion.models.metrics import Metrics
import torch
import numpy as np
import torch.nn as nn


class DSFNetModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        print('using DSFNet Model')
        self.visual_names = ['img', 'label', 'pred_traj_img', 'src', 'building_label', 'pred_building_img', 'pred_src_traj_img']
        self.loss_names = ['all']
        self.model_names = ['DSFNet']
        self.image_paths = []

        input_nc = 5
        self.netDSFNet = define_translator(input_nc, opt.output_nc, opt.net_trans, gpu_ids=opt.gpu_ids)
        self.metrics = Metrics()
        self.metrics1 = Metrics()
        self.metrics2 = Metrics()

        self.criterion = BCEDiceLoss()
        self.criterion1 = SoftDiceLoss()
        self.criterion2 = nn.BCEWithLogitsLoss()
        self.criterion3 = nn.BCELoss()
        self.criterion4 = BCE_Loss()

        if opt.is_train:
            self.optimizer = torch.optim.Adam(self.netDSFNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

        self.label = None
        self.img = None
        self.src = None
        self.traj = None
        self.traj_path = None
        self.src_path = None
        self.label_path = None
        self.building_label = None
        self.building_label_path = None
        self.pred_traj = None
        self.pred_building = None
        self.pred_traj_img = None
        self.pred_building_img = None
        self.src_pred_traj = None
        self.pred_src_traj_img = None
        self.loss_Dice = None
        self.loss_Building = None
        self.loss_Traj = None
        self.loss_src_Traj = None
        self.loss_all = None

    def set_input(self, input):
        self.traj_path = input['traj_path']
        self.src_path = input['src_path']
        self.label_path = input['label_path']
        self.building_label_path = input['building_label_path']

        self.img = input['traj_data'].to(self.device)
        self.src = input['src_data'].to(self.device)
        self.label = input['label_data'].to(self.device)
        self.building_label = input['building_label_data'].to(self.device)

    def set_input_test(self, input):
        self.traj_path = input['traj_path']
        self.src_path = input['src_path']
        self.label_path = input['label_path']

        self.img = input['traj_data'].to(self.device)
        self.src = input['src_data'].to(self.device)
        self.label = input['label_data'].to(self.device)
        self.image_paths = [input['traj_path'][0]]

    def forward(self):
        self.pred_traj, self.pred_building, self.src_pred_traj = self.netDSFNet(self.img, self.src)
        self.pred_traj_img = DSFNetModel.pred2im(self.pred_traj)
        self.pred_building_img = DSFNetModel.pred2im(self.pred_building)
        self.pred_src_traj_img = DSFNetModel.pred2im(self.src_pred_traj)

    def _backward(self):
        self.loss_Traj = self.criterion(self.label, self.pred_traj)
        self.loss_Building = self.criterion3(self.pred_building, self.building_label)
        self.loss_src_Traj = self.criterion(self.label, self.src_pred_traj)

        self.loss_all = (self.loss_Traj + self.loss_Building + self.loss_src_Traj)
        self.loss_all.backward()

    def optimize_parameters(self):
        self.forward()
        metrics = self.metrics(self.pred_traj, self.label)
        metrics1 = self.metrics1(self.pred_building, self.building_label)
        metrics2 = self.metrics2(self.src_pred_traj, self.label)

        self.optimizer.zero_grad()
        self._backward()
        self.optimizer.step()
        return self.loss_all, metrics, metrics1, metrics2

    def test(self):
        BaseModel.test(self)
        metrics = self.metrics(self.pred_traj, self.label)
        metrics1 = self.metrics1(self.pred_building, self.building_label)
        metrics2 = self.metrics2(self.src_pred_traj, self.label)

        self.loss_Traj = self.criterion(self.label, self.pred_traj)
        self.loss_Building = self.criterion3(self.pred_building, self.building_label)
        self.loss_src_Traj = self.criterion(self.label, self.src_pred_traj)
        self.loss_all = self.loss_Traj + self.loss_Building + self.loss_src_Traj

        return self.loss_all, metrics, metrics1, metrics2, self.loss_Traj, self.loss_Building, self.loss_src_Traj

    def testtest(self):
        BaseModel.test(self)
        metrics = self.metrics(self.pred_traj, self.label)
        metrics2 = self.metrics2(self.src_pred_traj, self.label)

        self.loss_Traj = self.criterion(self.label, self.pred_traj)
        self.loss_src_Traj = self.criterion(self.label, self.src_pred_traj)
        self.loss_all = (self.loss_Traj + self.loss_src_Traj)

        self.pred_traj, self.pred_building, self.src_pred_traj = self.netDSFNet(self.img, self.src)
        self.pred_traj_img = DSFNetModel.pred2im(self.pred_traj)
        self.pred_building_img = DSFNetModel.pred2im(self.pred_building)
        self.pred_src_traj_img = DSFNetModel.pred2im(self.src_pred_traj)

        return self.loss_all, metrics, metrics2, self.src_pred_traj

    @staticmethod
    def pred2im(image_tensor):
        image_numpy = image_tensor[0].cpu().float().detach().numpy()  # convert it into a numpy array
        # handle sigmoid cases
        image_numpy[image_numpy > 0.5] = 1
        image_numpy[image_numpy <= 0.5] = 0
        image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255
        return image_numpy

