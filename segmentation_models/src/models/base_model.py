# encoding:utf-8
import segmentation_models_pytorch as seg
import torch

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['disc', 'downy']
ACTIVATION = 'sigmoid'

unet = seg.Unet(encoder_name=ENCODER, encoder_depth=5, encoder_weights=ENCODER_WEIGHTS,
                activation=ACTIVATION, classes=len(CLASSES))

unetpp = seg.UnetPlusPlus(encoder_name=ENCODER, encoder_depth=5,
                          encoder_weights=ENCODER_WEIGHTS, activation=ACTIVATION,
                          classes=len(CLASSES))

fpn = seg.FPN(encoder_name=ENCODER, encoder_depth=5, encoder_weights=ENCODER_WEIGHTS,
              activation=ACTIVATION, classes=len(CLASSES))

deeplab3 = seg.DeepLabV3(encoder_name=ENCODER, encoder_depth=5, encoder_weights=ENCODER_WEIGHTS,
                         activation=ACTIVATION, classes=len(CLASSES))

deeplab3plus = seg.DeepLabV3Plus(encoder_name=ENCODER, encoder_depth=5,
                                 encoder_weights=ENCODER_WEIGHTS,
                                 activation=ACTIVATION, classes=len(CLASSES))

pan = seg.PAN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
              activation=ACTIVATION, classes=len(CLASSES))

manet = seg.MAnet(encoder_name=ENCODER, encoder_depth=5, encoder_weights=ENCODER_WEIGHTS,
                  activation=ACTIVATION, classes=len(CLASSES))

linknet = seg.Linknet(encoder_name=ENCODER, encoder_depth=5, encoder_weights=ENCODER_WEIGHTS,
                      activation=ACTIVATION, classes=len(CLASSES))

# renext_50_unet = seg.Unet(encoder_name='resnext50_32x4d',encoder_depth=5, encoder_weights=ENCODER_WEIGHTS,
#                           activation=ACTIVATION, classes=len(CLASSES))

# timmres2next_unet = seg.Unet(encoder_name='timm-res2net50_26w_4s',encoder_depth=5, encoder_weights=ENCODER_WEIGHTS,
#                           activation=ACTIVATION, classes=len(CLASSES))

# regnet_unet = seg.Unet(encoder_name='timm-regnetx_040', encoder_depth=5, encoder_weights=ENCODER_WEIGHTS,
#                        activation=ACTIVATION, classes=len(CLASSES))

# gernet_unet = seg.Unet(encoder_name='timm-gernet_m', encoder_depth=5,
#                        activation=ACTIVATION, classes=len(CLASSES))
# gernet_unet.load_state_dict(torch.load('/home/qtian/.cache/torch/hub/checkpoints/gernet_m-0873c53a.pth'))
#
#
# senet5032 = seg.Unet(encoder_name='se_resnext50_32x4d', encoder_depth=5,
#                      activation=ACTIVATION, classes=len(CLASSES))
# senet5032.load_state_dict(torch.load('/home/qtian/.cache/torch/hub/checkpoints/se_resnext50_32x4d-a260b3a4.pth'))
#
#
# skresnet34 = seg.Unet(encoder_name='timm-skresnet34', encoder_depth=5,
#                       activation=ACTIVATION, classes=len(CLASSES))
# skresnet34.load_state_dict(torch.load('/home/qtian/.cache/torch/hub/checkpoints/skresnet34_ra-bdc0ccde.pth'))
#
#
# inceptionv4_unet = seg.Unet(encoder_name='inceptionv4', encoder_depth=5,
#                             activation=ACTIVATION, classes=len(CLASSES))
# inceptionv4_unet.load_state_dict(torch.load('/home/qtian/.cache/torch/hub/checkpoints/inceptionv4-8e4777a0.pth'))
#
# dpn = seg.Unet(encoder_name='dpn68b', encoder_depth=5,
#                activation=ACTIVATION, classes=len(CLASSES))
# dpn.load_state_dict(torch.load('/home/qtian/.cache/torch/hub/checkpoints/dpn68b_extra-363ab9c19.pth'))

mvt = seg.Unet(encoder_name='mit_b2', encoder_depth=5, encoder_weights=ENCODER_WEIGHTS,
               activation=ACTIVATION, classes=len(CLASSES))

model_list = {"unet": unet, "unetpp": unetpp, "deeplab3": deeplab3,
              "deeplab3+": deeplab3plus, "pan": pan, "maenet": manet,
              "linknet": linknet, "fpn": fpn,
              }
