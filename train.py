import os
from time import time
import torch
import SimpleITK as sitk
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as dataset
from Dice_Bce_Loss import Dice
from Dice_Bce_Loss import DiceBCELoss
from paznet import generate_model
from transforms import RandomRotate, RandomFlip, Brightness, ContrastAdjustment, Compose

class Dataset(dataset):
    def __init__(self, tem_dir, mri_dir, seg_dir):

        self.mri_list = os.listdir(mri_dir)
        self.tem_list = list(map(lambda x: x.replace('pre_norm', 'tem_norm'), self.mri_list))
        self.seg_list = list(map(lambda x: x.replace('pre_norm', 'mask').replace('.nii', '.nii.gz'), self.mri_list))

        self.tem_list = list(map(lambda x: os.path.join(tem_dir, x), self.tem_list))
        self.mri_list = list(map(lambda x: os.path.join(mri_dir, x), self.mri_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))

        self.transforms = Compose([RandomFlip(), RandomRotate(), Brightness(), ContrastAdjustment()])

    def __getitem__(self, index):
        tem_path = self.tem_list[index]
        mri_path = self.mri_list[index]
        seg_path = self.seg_list[index]

        tem = sitk.ReadImage(tem_path, sitk.sitkFloat32)
        mri = sitk.ReadImage(mri_path, sitk.sitkFloat32)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        tem_array = sitk.GetArrayFromImage(tem)
        mri_array = sitk.GetArrayFromImage(mri)
        seg_array = sitk.GetArrayFromImage(seg)

        tem_array = torch.FloatTensor(tem_array).unsqueeze(0)
        mri_array = torch.FloatTensor(mri_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)

        if self.transforms:
            tem_array, mri_array, seg_array = self.transforms(tem_array, mri_array, seg_array)

        return tem_array, mri_array, seg_array.squeeze(0)

    def __len__(self):

        return len(self.mri_list)

# 训练网络
if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    para = {
        "training_set_path": "./traindata_80/",
        "batch_size": 8,
        "num_workers": 4,
        "pin_memory": True,
        "cudnn_benchmark": True,
        "epoch": 201,
        "pretrain_path": "./pretrain/resnet_10.pth"
    }

    train_ds = Dataset(os.path.join(para['training_set_path'], 'tem_norm'), os.path.join(para['training_set_path'], 'pre_norm'), os.path.join(para['training_set_path'], 'mask'))
    train_dl = DataLoader(train_ds, para['batch_size'], True, num_workers=para['num_workers'], pin_memory=para['pin_memory'])
    loss_func = DiceBCELoss()
    dice_tr = Dice()

    # 定义网络
    model, parameters = generate_model(
        training=True,
        no_cuda=False,
        gpu_id=[0],
        phase='train',
        pretrain_path=para["pretrain_path"]
    )
    params = [
        {'params': parameters['base_parameters'], 'lr': 0.001},
        {'params': parameters['new_parameters'], 'lr': 0.001 * 100}
    ]

    opt = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    model.train()

    start = time()

    for epoch in range(para['epoch']):

        mean_loss = []

        for step, (tem, mri, seg) in enumerate(train_dl):

            tem = tem.cuda()
            mri = mri.cuda()
            seg = seg.cuda()

            outputs = model(mri, tem)
            loss1 = loss_func(outputs[0], seg)
            loss2 = loss_func(outputs[1], seg)
            loss3 = loss_func(outputs[2], seg)
            loss = loss1 + 0.5 * loss2 + 0.3 * loss3
            dice = dice_tr(outputs[0], seg)

            mean_loss.append(loss.item())

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
            opt.step()

            if step % 10 == 0:
                print('epoch:{}, step:{}, dice:{:.3f}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, time:{:.3f} min'
                      .format(epoch, step, dice.item(), loss1.item(), loss2.item(), loss3.item(), (time() - start) / 60))

        mean_loss = sum(mean_loss) / len(mean_loss)
        scheduler.step(mean_loss)
        if epoch % 20 == 0 and epoch != 0:
            torch.save(model.state_dict(), './module/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss))

        print(f'Current learning rate: {scheduler.optimizer.param_groups[0]["lr"]}')


