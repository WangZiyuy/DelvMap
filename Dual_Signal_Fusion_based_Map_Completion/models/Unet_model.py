from Dual_Signal_Fusion_based_Map_Completion.models.networks import define_translator, init_net
from Dual_Signal_Fusion_based_Map_Completion.models.base_model import BaseModel
from Dual_Signal_Fusion_based_Map_Completion.models.losses import SoftDiceLoss, BCEDiceLoss
from Dual_Signal_Fusion_based_Map_Completion.models.metrics import Metrics
import torch
import numpy as np


class UNetModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.visual_names = ['img', 'label', 'pred_img_img']
        self.loss_names = ['Dice']
        self.model_names = ['Unet']

        input_nc = 2
        self.netUnet = define_translator(input_nc, opt.output_nc, opt.net_trans, gpu_ids=opt.gpu_ids)
        self.metrics = Metrics()

        self.criterion = BCEDiceLoss()
        self.criterion1 = SoftDiceLoss()

        if opt.is_train:
            self.optimizer = torch.optim.Adam(self.netUnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        self.img = input['img_data'].to(self.device)
        # torch.Size([1, 1, 256, 256])
        self.label = input['label_data'].to(self.device)
        # torch.Size([1, 3, 256, 256])
        self.img_path = input['img_path']
        self.label_path = input['label_path']

    def forward(self):
        self.pred_img = self.netUnet(self.img)
        self.pred_img_img = UNetModel.pred2im(self.pred_img)
        # print('pred_img shape', self.pred_img.shape, self.pred_img_img.shape)
        # torch.Size([1, 256, 256]) (256, 256, 3)

    def _backward(self):
        self.loss_Dice = self.criterion(self.label, self.pred_img)
        self.loss_Dice.backward()

    def optimize_parameters(self):
        self.forward()
        metrics = self.metrics(self.pred_img, self.label)

        self.optimizer.zero_grad()
        self._backward()
        self.optimizer.step()
        return self.loss_Dice, metrics

    def test(self):
        BaseModel.test(self)
        metrics = self.metrics(self.pred_img, self.label)
        self.loss_Dice = self.criterion(self.pred_img, self.label)
        return self.loss_Dice, metrics,

    @staticmethod
    def pred2im(image_tensor):
        image_numpy = image_tensor[0].cpu().float().detach().numpy()  # convert it into a numpy array
        # handle sigmoid cases
        image_numpy[image_numpy > 0.5] = 1
        image_numpy[image_numpy <= 0.5] = 0
        image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255
        return image_numpy
