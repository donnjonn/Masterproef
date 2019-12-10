import sys
import torch

import PIL
from tqdm import tqdm
import numpy as np
learning_rate = 0.03
from scripts.main import *
from scripts.default_config import *
from torchreid.engine import *
import matplotlib.pyplot as plt
sys.path.append('/home/jonas/deep-person-reid/adversarial_yolo')
from adversarial_yolo.train_patch import PatchTrainer
from torchvision import transforms
import torch.nn.functional as F
pt = PatchTrainer("paper_obj")
cfg = get_default_config()
adv_patch = pt.generate_patch("gray")
adv_patch.requires_grad_(True)
dm = build_datamanager()
cfg.use_gpu = True
model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=dm.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
optimizer = torch.optim.Adam([adv_patch], lr=learning_rate, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
print("scheduler ok")
img_size = 416
batch_size = 8
n_epochs = 100
max_lab = 14
trainloader, testloader = dm.return_dataloaders()
epoch_length = len(trainloader)
counter = 0
engine = build_engine(cfg, dm, model, optimizer, scheduler)
for epoch in range(n_epochs):
            print(counter)
            counter+=1
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            for batch_idx, data in enumerate(trainloader):
                imgs, pids = engine._parse_data_for_train(data)
                img_batch = imgs.cuda()
                #img = img_batch[1, :, :,:]
                #print(img)
                #imgplt = transforms.ToPILImage()(img.detach().cpu())
                #imgpltmat = np.array(imgplt)
                #print(imgpltmat)
                #print(pids[0])
                #plt.imshow(imgplt)
                #plt.show()
                pid_batch = pids.cuda()
                advpatch_cuda = adv_patch.cuda()
                advbatch_t = pt.patch_transformer(advpatch_cuda, pid_batch, img_size, do_rotate=True, rand_loc=False)
                advbatch_tc = advbatch_t.cuda()
                #plt.imshow(advbatch_t[0])
                #print(img_batch.size())
                #print(advbatch_tc.size())
                p_img_batch = pt.patch_applier(img_batch, advbatch_tc)
                p_img_batch = F.interpolate(p_img_batch, (256, 128))
                img = p_img_batch[1, :, :,]
                imgplt = transforms.ToPILImage()(img.detach().cpu())
                imgplt.show()
                #img.permute(1,2,0)
                #imgplt = transforms.ToPILImage()(img)
                #plt.imshow(imgplt)
                #plt.show()
                #img.show()
                continue

