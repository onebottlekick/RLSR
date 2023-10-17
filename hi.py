from models import *
import torch

swinir_pth = 'models/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth'
han_pth = 'models/han/HAN_BIX2.pt'
hat_pth = 'models/hat/HAT_SRx2_ImageNet-pretrain.pth'

pth = hat_pth
pth = torch.load(pth)
# print(pth)
model = HAT()
model.load_state_dict(pth['params_ema'])
# print(model)

img = torch.randn(1, 3, 256, 256)
y = model(img)
print(y.shape)


############################################################################################
# from PIL import Image
# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt

# img = Image.open('img_001_SRF_2_LR.png')
# img = ToTensor()(img)
# img = img.unsqueeze(0)
# sr = model(img)

# plt.imshow(sr.squeeze(0).permute(1, 2, 0).detach().numpy())
# plt.axis('off')
# plt.show()