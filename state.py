import torch
from models import *
from utils.common import exist_value, to_cpu

class State:
    def __init__(self, device):
        self.device = device
        self.lr_image = None
        self.sr_image = None
        self.tensor = None
        self.move_range = 3

        dev = torch.device(device)
        self.HAN = HAN().to(device)
        model_path = "models/model_weights/HAN_BIX2.pt"
        self.HAN.load_state_dict(torch.load(model_path, dev))
        self.HAN.eval()

        self.HAT = HAT().to(device)
        model_path = 'models/model_weights/HAT_SRx2_ImageNet-pretrain.pth'
        self.HAT.load_state_dict(torch.load(model_path, dev)['params_ema'])
        self.HAT.eval()

        self.SwinIR = SwinIR().to(device)
        model_path = "models/model_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
        self.SwinIR.load_state_dict(torch.load(model_path, dev)['params'])
        self.SwinIR.eval()

        self.VDSR = SwinIR().to(device)
        model_path = "models/model_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
        self.VDSR.load_state_dict(torch.load(model_path, dev)['params'])
        self.VDSR.eval()

    def reset(self, lr, bicubic):
        self.lr_image = lr 
        self.sr_image = bicubic
        b, _, h, w = self.sr_image.shape
        previous_state = torch.zeros(size=(b, 64, h, w), dtype=self.lr_image.dtype)
        self.tensor = torch.concat([self.sr_image, previous_state], dim=1)

    def set(self, lr, bicubic):
        self.lr_image = lr
        self.sr_image = bicubic
        self.tensor[:,0:3,:,:] = self.sr_image

    def step(self, act, inner_state):
        act = to_cpu(act)
        inner_state = to_cpu(inner_state)
        han = self.sr_image.clone()
        hat = self.sr_image.clone()
        swinir = self.sr_image.clone()
        vdsr = self.sr_image.clone()

        neutral = (self.move_range - 1) / 2
        move = act.type(torch.float32)
        move = (move - neutral) / 255
        moved_image = self.sr_image.clone()
        for i in range(0, self.sr_image.shape[1]):
            moved_image[:,i] += move[0]

        self.lr_image = self.lr_image.to(self.device)
        self.sr_image = self.sr_image.to(self.device)

        with torch.no_grad():
            if exist_value(act, 3):
                han = to_cpu(self.HAN(self.lr_image))
            if exist_value(act, 4):
                hat = to_cpu(self.HAT(self.lr_image))
            if exist_value(act, 5):
                swinir = to_cpu(self.SwinIR(self.lr_image))
            if exist_value(act, 6):
                vdsr = to_cpu(self.VDSR(self.lr_image))

        self.lr_image = to_cpu(self.lr_image)
        self.sr_image = moved_image
        act = act.unsqueeze(1)
        act = torch.concat([act, act, act], 1)
        self.sr_image = torch.where(act==3, han,  self.sr_image)
        self.sr_image = torch.where(act==4, hat,  self.sr_image)
        self.sr_image = torch.where(act==5, swinir, self.sr_image)
        self.sr_image = torch.where(act==6, vdsr, self.sr_image)

        self.tensor[:,0:3,:,:] = self.sr_image
        self.tensor[:,-64:,:,:] = inner_state