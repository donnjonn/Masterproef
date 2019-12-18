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
import pandas as pd


pt = PatchTrainer("paper_obj")
cfg = get_default_config()
adv_patch = pt.generate_patch("gray")
adv_patch.requires_grad_(True)
dm = build_datamanager()
cfg.use_gpu = True
cfg.test.evaluate = True
model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=dm.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
centers_df = pd.read_csv('featuretensors/centers.csv')
centers_df.drop(centers_df.columns[[0]], axis=1, inplace=True)
target = centers_df.values[0]
optimizer = torch.optim.Adam([adv_patch], lr=learning_rate, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
img_size = 416
batch_size = 8
n_epochs = 100
max_lab = 14
trainloader, testloader = dm.return_dataloaders()
galleryloader = testloader['market1501']['gallery']
targets = list(testloader.keys())
epoch_length = len(trainloader)
counter = 0
engine = build_engine(cfg, dm, model, optimizer, scheduler)

def tester(qf, q_pids, q_camids, batch_idx, data):
    imgs, pids, camids = engine._parse_data_for_eval(data)
    #imgs = imgs.cuda()
    features = engine._extract_features(imgs)
    features = features.data.cpu()
    #print("features lijst: ", features)
    #print("lengte:", len(features))
    qf.append(features)
    q_pids.extend(pids)
    q_camids.extend(camids)
    return features

for epoch in range(n_epochs):
            print(counter)
            counter+=1
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            qf, q_pids, q_camids = [], [], []
            for batch_idx, data in enumerate(galleryloader):
                imgs, pids, camids = engine._parse_data_for_eval(data)
                img_batch = imgs.cuda()
                img = img_batch[0, :, :,:]
                print(img)
                #transform = transforms.Compose([transforms.Normalize((-1,-1,-1),(2,2,2))])
                #img = transform(img)
                #print(img)
                imgplt = transforms.ToPILImage()(img.detach().cpu())
                #imgpltmat = np.array(imgplt)
                #print(imgpltmat)
                print(pids[0])
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
		
                plt.imshow(imgplt)
                plt.show()
                features = tester(qf, q_pids, q_camids, batch_idx, data)
                #print(features.size())
                afst_array = []
                for row in features:
                    afst_array.append(torch.dist(torch.from_numpy(target), row.double()))
                    #print("afst: ",afst)
                afst_array = torch.stack(afst_array)
                #torch.cat(afst_array, out=b)
                afst = torch.mean(afst_array)
                afst.requires_grad_(True)
                #batch_size = img_batch.size()
                #afst = afst.item()/batch_size[0]
                print(afst)
                #print(qf)
                afst.backward()
                optimizer.step()
                optimizer.zero_grad()
                #img.permute(1,2,0)
                #imgplt = transforms.ToPILImage()(img)
                #plt.imshow(imgplt)
                #plt.show()
                #img.show()
                continue




