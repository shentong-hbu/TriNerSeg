import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
from tqdm import tqdm
import SimpleITK as sitk
import torchio
import time

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

DATABASE = 'TRINE/'
#
args = {
    'root'     : '/',
    'test_path': './dataset/' + DATABASE + 'test/',
    'pred_path': 'assets/' + 'SegResults/',
}

if not os.path.exists(args['pred_path']):
    os.makedirs(args['pred_path'])


def load_3dV2():
    test_images = sorted(glob.glob(os.path.join(args['test_path'], "images", "*.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(args['test_path'], "labels", "*.nii.gz")))
    return test_images, test_labels


def load_net():
    net = torch.load('./checkpoint/FinerRes2CSNet.pth')
    print(net)
    return net


def save_prediction(pred, filename='', spacing=None, origin=None, direction=None):
    pred = torch.argmax(pred, dim=1)
    save_path = args['pred_path'] + 'pred/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Make dirs success!")
    mask = pred.data.cpu().numpy()
    mask = mask / np.max(mask)
    mask = (mask * 255).astype(np.uint8)

    mask = mask.squeeze(0)  # for CE Loss
    mask = np.transpose(mask, axes=(2, 1, 0))
    mask = sitk.GetImageFromArray(mask)
    if spacing is not None:
        mask.SetSpacing(spacing)
        mask.SetOrigin(origin)
        mask.SetDirection(direction)
    sitk.WriteImage(mask, os.path.join(save_path + filename + ".mha"))


def predict():
    net = load_net()
    images, labels = load_3dV2()
    print(len(images))
    with torch.no_grad():
        net.eval()
        fps = []
        for i in tqdm(range(len(images))):
            name_list = images[i].split('/')
            index = name_list[-1][:-7]
            transform = torchio.RescaleIntensity()
            subject = torchio.Subject(
                image=torchio.ScalarImage(images[i]),
                label=torchio.LabelMap(labels[i]),
            )
            transformed = transform(subject)
            image = transformed['image'][torchio.DATA].numpy().astype(np.float32)
            label = transformed['label'][torchio.DATA].numpy().astype(np.int64).squeeze(0)

            # select a reference volume to obtain the spacing, origin, and direction
            config = sitk.ReadImage(images[i])
            spacing = config.GetSpacing()
            origin = config.GetOrigin()
            direction = config.GetDirection()

            # if cuda
            image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
            # image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).unsqueeze(0)
            image = image.cuda()
            coarse, output = net(image)

            # save_prediction(output, affine=affine, filename=index + '_pred')
            save_prediction(output, filename=index + '_pred', spacing=spacing, origin=origin, direction=direction)


if __name__ == '__main__':
    predict()
