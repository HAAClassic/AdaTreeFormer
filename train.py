import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime
import torch.nn.functional as F
import random

from datasets.crowd import Crowd_TC, Crowd_UL_TC
from torch.nn.modules.loss import CrossEntropyLoss 

from network.Attention import QKV
from network.discriminator import FCDiscriminator

from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils
import argparse
from losses.rank_loss import RankLoss

from losses import ramps, losses
from losses.ot_loss import OT_Loss
from losses.consistency_loss import *

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--data-dir', default='', help='data path')
parser.add_argument('--data-dir-ul', default='', help='data path')  
   
parser.add_argument('--dataset', default='TC')
parser.add_argument('--lr', type=float, default=1e-5, help='the initial learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='the weight decay')
parser.add_argument('--ema-decay', type=float, default=0.99, help='the ema decay')
parser.add_argument('--resume', default='', type=str, help='the path of resume training model')
parser.add_argument('--max-epoch', type=int, default=4000, help='max training epoch')
parser.add_argument('--val-epoch', type=int, default=1, help='the num of steps to log training information')
parser.add_argument('--val-start', type=int, default=0, help='the epoch start to val')
parser.add_argument('--batch-size', type=int, default=5, help='train batch size')
parser.add_argument('--batch-size-ul', type=int, default=5, help='train batch size')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--num-workers', type=int, default=0, help='the num of training process')
parser.add_argument('--ot', type=float, default=0.1, help='entropy regularization in sinkhorn')
parser.add_argument('--tv', type=float, default=0.01, help='entropy regularization in sinkhorn')
parser.add_argument('--crop-size', type=int, default= 256, help='the crop size of the train image')
parser.add_argument('--reg', type=float, default=1, help='entropy regularization in sinkhorn')
parser.add_argument('--num-of-iter-in-ot', type=int, default=100, help='sinkhorn iterations')
parser.add_argument('--norm-cood', type=int, default=0, help='whether to norm cood when computing distance') 
parser.add_argument('--run-name', default='Y2L_CutMix_Att_Dis_new', help='run name for wandb interface/logging')
parser.add_argument('--consistency', type=int, default=1, help='whether to norm cood when computing distance')
parser.add_argument('--consistency-ramp', type=int, default=200, help='whether to norm cood when computing distance')
parser.add_argument('--scale-factor', type=int, default=4, help='whether to norm cood when computing distance')

args = parser.parse_args() 

def rand_bbox(size, lamb):
    """ Generate random bounding box 
    Args:
        - size: [width, breadth] of the bounding box
        - lamb: (lambda) cut ratio parameter
    Returns:
        - Bounding box
    """
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lamb)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def generate_cutmix_image(image_batch, gauss_batch, gt_discrete_batch, beta, scale_factor):
    """ Generate a CutMix augmented image from a batch 
    Args:
        - image_batch: a batch of input images
        - image_batch_labels: labels corresponding to the image batch
        - beta: a parameter of Beta distribution.
    Returns:
        - CutMix image batch, updated labels
    """
    # generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = np.random.permutation(len(image_batch))
    bbx1, bby1, bbx2, bby2 = rand_bbox(image_batch[0][0].shape, lam)
    
    image_batch_updated = image_batch.clone()
    image_batch_updated[:,:, bbx1:bbx2, bby1:bby2] = image_batch[rand_index,:, bbx1:bbx2, bby1:bby2]
    
    gauss_batch_updated = gauss_batch.clone()
    gauss_batch_updated[:, bbx1:bbx2, bby1:bby2] = gauss_batch[rand_index, bbx1:bbx2, bby1:bby2]
    
    # import pdb;pdb.set_trace()
    gt_discrete_batch_updated = gt_discrete_batch.clone()
    gt_discrete_batch_updated[:, :, bbx1//scale_factor:bbx2//scale_factor, bby1//scale_factor:bby2//scale_factor] = \
            gt_discrete_batch[rand_index, :, bbx1//scale_factor:bbx2//scale_factor, bby1//scale_factor:bby2//scale_factor]
    
    return image_batch_updated, gauss_batch_updated, gt_discrete_batch_updated
    
def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    gauss = torch.stack(transposed_batch[1], 0)
    points = transposed_batch[2]
    gt_discretes = torch.stack(transposed_batch[3], 0)
    return images, gauss, points, gt_discretes

def train_collate_ul(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    gauss = torch.stack(transposed_batch[1], 0)
    points = transposed_batch[2]
    gt_discretes = torch.stack(transposed_batch[3], 0)
    return images, gauss, points, gt_discretes


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_ramp)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        sub_dir = (
            "UDA/{}_12-1-input-{}_reg-{}_nIter-{}_normCood-{}".format(
                args.run_name,args.crop_size,args.reg,
                args.num_of_iter_in_ot,args.norm_cood))

        self.save_dir = os.path.join("/scratch/users/k2254235","ckpts", sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        time_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
        self.logger = log_utils.get_logger(
            os.path.join(self.save_dir, "train-{:s}.log".format(time_str)))
            
        log_utils.print_config(vars(args), self.logger)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            self.logger.info("using {} gpus".format(self.device_count))
        else:
            raise Exception("gpu is not available")
        
        
        downsample_ratio = args.scale_factor
        self.datasets = {"train": Crowd_TC(os.path.join(args.data_dir, "train_data_100"), args.crop_size,
                downsample_ratio, "train"), "val": Crowd_TC(os.path.join(args.data_dir, "test_data"),
                args.crop_size, downsample_ratio, "val")}
        
        self.datasets_shot = { "train": Crowd_TC(os.path.join(args.data_dir_ul, "train_data_01"), 
                args.crop_size, downsample_ratio, "train"),"val": Crowd_TC(os.path.join(args.data_dir_ul, "test_data"),
                args.crop_size, downsample_ratio, "val")}
        
       
        self.dataloaders = {
            x: DataLoader(self.datasets[x],
                collate_fn=(train_collate if x == "train" else default_collate),
                batch_size=(args.batch_size if x == "train" else 1),
                shuffle=(True if x == "train" else False),
                num_workers=args.num_workers * self.device_count,
                pin_memory=(True if x == "train" else False))
            for x in ["train", "val"]}
        
        self.dataloaders_shot = {
            x: DataLoader(self.datasets_shot[x],
                collate_fn=(train_collate_ul if x == "train" else default_collate),
                batch_size=(args.batch_size_ul if x == "train" else 1),
                shuffle=(True if x == "train" else False),
                num_workers=args.num_workers * self.device_count,
                pin_memory=(True if x == "train" else False))
            for x in ["train", "val"]}
   

        self.model = QKV('swin','swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth', False)
        self.model.to(self.device)
        
        self.DAN = FCDiscriminator(num_classes=1)
        self.DAN.to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.DAN_optimizer = optim.Adam(self.DAN.parameters(), lr=0.0001, betas=(0.9, 0.99))
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = losses.DiceLoss(1) 
        
        self.start_epoch = 0
        
        if args.resume:

            self.logger.info("loading pretrained model from " + args.resume)
            suf = args.resume.rsplit(".", 1)[-1]
            if suf == "tar":
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"])
                self.start_epoch = checkpoint["epoch"] + 1
            elif suf == "pth":
                self.model.load_state_dict(
                    torch.load(args.resume, self.device))
        else:
            self.logger.info("random initialization")
            
        self.ot_loss = OT_Loss(args.crop_size, downsample_ratio, args.norm_cood, 
              self.device, args.num_of_iter_in_ot, args.reg)
              
        self.tvloss = nn.L1Loss(reduction="none").to(self.device)
        self.att_dis = nn.MSELoss(reduction = "sum").to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.logger.info("-" * 5 + "Epoch {}/{}".format(epoch, args.max_epoch) + "-" * 5)
            self.epoch = epoch
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_epoch(self):
        epoch_loss = AverageMeter()
        epoch_ot_s2s = AverageMeter()
        epoch_tv_s2s = AverageMeter()
        epoch_count_s2s = AverageMeter()
        epoch_ot_t2s = AverageMeter()
        epoch_tv_t2s = AverageMeter()
        epoch_count_t2s = AverageMeter()
        epoch_ot_s2t = AverageMeter()
        epoch_tv_s2t = AverageMeter()
        epoch_count_s2t = AverageMeter()
        epoch_ot_t2t = AverageMeter()
        epoch_tv_t2t = AverageMeter()
        epoch_count_t2t = AverageMeter()
        
        epoch_att = AverageMeter()
        epoch_consistency = AverageMeter()
        
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        
        
        self.model.train()  # Set model to training mode
        self.DAN.eval()
        
        for step, (inputs, gausss, points, gt_discrete) in enumerate(self.dataloaders["train"]):
            inputs = inputs.to(self.device)
            gausss = gausss.to(self.device).unsqueeze(1)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            N_tr = inputs.size(0)
            
            for st, (inputs_shot, gausss_shot, points_shot, gt_discrete_shot) in enumerate(self.dataloaders_shot["train"]):
                if random.random() > 0.1:
                    inputs_shot, gausss_shot, gt_discrete_shot = generate_cutmix_image(inputs_shot, gausss_shot, gt_discrete_shot, 1, args.scale_factor)
                    inputs_shot = inputs_shot.to(self.device)
                    gausss_shot = gausss_shot.to(self.device).unsqueeze(1)
                    gd_count_shot = np.array([len(p) for p in points_shot], dtype=np.float32)
                    gt_discrete_shot = gt_discrete_shot.to(self.device)
                    N_shot = inputs_shot.size(0)
                    points_shot = []
                    for ibs in range(0,N_shot):
                        point_ = []
                        sel_gt = gt_discrete_shot[ibs][0]
                        if int(sel_gt.max())>1:
                            for nbs in (1,int(sel_gt.max())):
                                if nbs != 0:
                                    pointid = torch.fliplr((sel_gt==nbs).nonzero())
                                    if nbs>1:
                                        for sbs in (1,int(sel_gt.max())):
                                            point_.append(pointid)
                                    else:
                                        point_.append(pointid)
                            pointvec = torch.cat(point_, dim=0).float()
                        else:
                            pointvec = torch.fliplr((sel_gt==1).nonzero()).float()
                        points_shot.append(pointvec*args.scale_factor)
                else:
                    inputs_shot = inputs_shot.to(self.device)
                    gausss_shot = gausss_shot.to(self.device).unsqueeze(1)
                    gd_count_shot = np.array([len(p) for p in points_shot], dtype=np.float32)
                    points_shot = [p.to(self.device) for p in points_shot]
                    gt_discrete_shot = gt_discrete_shot.to(self.device)
                    N_shot = inputs_shot.size(0)
                break
            self.model.train()
            self.DAN.eval()
 
            with torch.set_grad_enabled(True): 
                [s2s_den,att_s2s], [t2s_den,att_t2s], [s2t_den,att_s2t], [t2t_den,att_t2t] = self.model(inputs, inputs_shot)
                
                B,C,H,W = s2s_den.size()
                label_sum = s2s_den.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                s2s_normed = s2s_den / (label_sum + 1e-6)
                
                B,C,H,W = t2s_den.size()
                label_sum = t2s_den.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                t2s_normed = t2s_den / (label_sum + 1e-6)
        
                B,C,H,W = s2t_den.size()
                label_sum = s2t_den.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                s2t_normed = s2t_den / (label_sum + 1e-6)
                
                B,C,H,W = t2t_den.size()
                label_sum = t2t_den.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                t2t_normed = t2t_den / (label_sum + 1e-6)
                
                # Compute counting loss.
                s2s_count_loss = self.mae(s2s_den.sum(1).sum(1).sum(1),torch.from_numpy(gd_count).float().to(self.device))*self.args.reg
                epoch_count_s2s.update(s2s_count_loss.item(), N_tr)
                
                t2s_count_loss = self.mae(t2s_den.sum(1).sum(1).sum(1),torch.from_numpy(gd_count).float().to(self.device))*self.args.reg
                epoch_count_t2s.update(t2s_count_loss.item(), N_tr)
                
                s2t_count_loss = self.mae(s2t_den.sum(1).sum(1).sum(1),torch.from_numpy(gd_count_shot).float().to(self.device))*self.args.reg
                epoch_count_s2t.update(s2t_count_loss.item(), N_shot)
                
                t2t_count_loss = self.mae(t2t_den.sum(1).sum(1).sum(1),torch.from_numpy(gd_count_shot).float().to(self.device))*self.args.reg
                epoch_count_t2t.update(t2t_count_loss.item(), N_shot)
                
     
                # Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(s2s_normed, s2s_den, points)    
                ot_s2s_loss = ot_loss * self.args.ot
                epoch_ot_s2s.update(ot_s2s_loss.item(), N_tr)
                
                ot_loss, wd, ot_obj_value = self.ot_loss(t2s_normed, t2s_den, points)    
                ot_t2s_loss = ot_loss * self.args.ot
                epoch_ot_t2s.update(ot_t2s_loss.item(), N_tr)
                
                ot_loss, wd, ot_obj_value = self.ot_loss(s2t_normed, s2t_den, points_shot)    
                ot_s2t_loss = ot_loss * self.args.ot
                epoch_ot_s2t.update(ot_s2t_loss.item(), N_shot)
                                
                ot_loss, wd, ot_obj_value = self.ot_loss(t2t_normed, t2t_den, points_shot)    
                ot_t2t_loss = ot_loss * self.args.ot
                epoch_ot_t2t.update(ot_t2t_loss.item(), N_shot)
                
                # Compute TV loss.
                gd_count_tensor = (torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3))
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                tv_s2s_loss = (self.tvloss(s2s_normed, gt_discrete_normed).sum(1).sum(1).sum(1)* torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.args.tv
                epoch_tv_s2s.update(tv_s2s_loss.item(), N_tr)
                
                gd_count_tensor = (torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3))
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                tv_t2s_loss = (self.tvloss(t2s_normed, gt_discrete_normed).sum(1).sum(1).sum(1)* torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.args.tv
                epoch_tv_t2s.update(tv_t2s_loss.item(), N_tr)
                
                gd_count_shot_tensor = (torch.from_numpy(gd_count_shot).float().to(self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3))
                gt_discrete_normed = gt_discrete_shot / (gd_count_shot_tensor + 1e-6)
                tv_s2t_loss = (self.tvloss(s2t_normed, gt_discrete_normed).sum(1).sum(1).sum(1)* torch.from_numpy(gd_count_shot).float().to(self.device)).mean(0) * self.args.tv
                epoch_tv_s2t.update(tv_s2t_loss.item(), N_shot)
                                
                gd_count_shot_tensor = (torch.from_numpy(gd_count_shot).float().to(self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3))
                gt_discrete_normed = gt_discrete_shot / (gd_count_shot_tensor + 1e-6)
                tv_t2t_loss = (self.tvloss(t2t_normed, gt_discrete_normed).sum(1).sum(1).sum(1)* torch.from_numpy(gd_count_shot).float().to(self.device)).mean(0) * self.args.tv
                epoch_tv_t2t.update(tv_t2t_loss.item(), N_shot)
                
                # import pdb;pdb.set_trace()
                # Attention Loss
                Att_loss_s = self.att_dis(att_t2s[0], att_s2s[0]) + self.att_dis(att_t2s[1], att_s2s[1]) + self.att_dis(att_t2s[2], att_s2s[2])
                Att_loss_t = self.att_dis(att_s2t[0], att_t2t[0]) + self.att_dis(att_s2t[1], att_t2t[1]) + self.att_dis(att_s2t[2], att_t2t[2])
                Att_loss = Att_loss_s + Att_loss_t
                epoch_att.update(Att_loss.item(), N_tr)
                
                ###################################################### Adversial ##############################
                vol_input = torch.cat((inputs, inputs_shot),0)
                vol_output = torch.cat((s2s_den, t2t_den),0)
                DAN_target = torch.tensor([0] * (args.batch_size + args.batch_size_ul)).cuda()
                DAN_target[:args.batch_size] = 1
                
                DAN_outputs = self.DAN(vol_output[args.batch_size,:], vol_input[args.batch_size:])
                consistency_loss = F.cross_entropy(DAN_outputs, DAN_target[:args.batch_size].long())
                consistency_weight = get_current_consistency_weight(self.epoch)
                epoch_consistency.update(consistency_loss.item(), N_tr)
                
                # import pdb;pdb.set_trace()
                loss = (s2s_count_loss + ot_s2s_loss + tv_s2s_loss) + (t2s_count_loss + ot_t2s_loss + tv_t2s_loss) + (s2t_count_loss + ot_s2t_loss + tv_s2t_loss ) + \
                            (t2t_count_loss + ot_t2t_loss + tv_t2t_loss )  +  consistency_weight*( consistency_loss + Att_loss_s + Att_loss_t)#
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_count = (torch.sum(t2s_den.view(N_tr, -1),dim=1).detach().cpu().numpy())
                              
                pred_err = pred_count - gd_count
                epoch_loss.update(loss.item(), N_tr)
                epoch_mse.update(np.mean(pred_err * pred_err), N_tr)
                epoch_mae.update(np.mean(abs(pred_err)), N_tr)
    
            self.model.eval()  # Set model to training mode
            self.DAN.train()
        
            with torch.no_grad():
                [s2s_den,_], _,_, [t2t_den,_] = self.model(inputs, inputs_shot)
            vol_output = torch.cat((s2s_den, t2t_den),0)
            DAN_outputs = self.DAN(vol_output, vol_input)
            DAN_loss = F.cross_entropy(DAN_outputs, DAN_target.long())      
            self.DAN_optimizer.zero_grad()
            DAN_loss.backward()
            self.DAN_optimizer.step()    
        
        self.logger.info(
            "Epoch {} Train, Loss: {:.2f}, Count s2s Loss: {:.2f}, t2s Loss: {:.2f}, Count s2t Loss: {:.2f}, Count t2t Loss: {:.2f}, Attention Loss: {:.2f},"
                 "Consistency Loss: {:.2f}, MSE: {:.2f}, MAE: {:.2f}, Cost {:.1f} sec".format(
                self.epoch, epoch_loss.get_avg(), epoch_count_s2s.get_avg(), epoch_count_t2s.get_avg(), epoch_count_s2t.get_avg(), epoch_count_t2t.get_avg(), 
                epoch_att.get_avg(), epoch_consistency.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(), time.time() - epoch_start))
                  
     


    def val_epoch(self):
        args = self.args 
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        for inputs, count, name, gauss_im in self.dataloaders["val"]:
            for inputs_shot, count_shot, name_shot, gausss_shot in self.dataloaders["val"]:
                break
            with torch.no_grad():
                inputs_shot = inputs_shot.to(self.device)
                inputs = inputs.to(self.device)
                    
                crop_imgs_shot, crop_masks_shot = [], []
                crop_imgs, crop_masks = [], []
                b, c, h, w = inputs.size()
                bsh, csh, hsh, wsh = inputs.size()
                rh, rw = args.crop_size, args.crop_size
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        randloc = abs(int(np.random.rand(1)*inputs_shot.shape[2])-rw)
                        crop_imgs_shot.append(inputs_shot[:, :, randloc:randloc+rh, randloc:randloc+rw])
                        crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                        mask = torch.zeros([b, 1, h, w]).to(self.device)
                        mask_shot = torch.zeros([bsh, 1, hsh, wsh]).to(self.device)
                        mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                        crop_masks_shot.append(mask_shot)
                        crop_masks.append(mask)
                
                crop_imgs_shot, crop_masks_shot = map(lambda x: torch.cat(x, dim=0), (crop_imgs_shot, crop_masks_shot))
                crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))
                       
                crop_preds = []
                nz, bz = crop_imgs.size(0), args.batch_size
                for i in range(0, nz, bz):
                    
                    gs, gt = i, min(nz, i + bz)
                    _,_,_,crop_pred = self.model(crop_imgs[gs:gt],crop_imgs[gs:gt])  
                    crop_pred = crop_pred[0]   
                    _, _, h1, w1 = crop_pred.size()                
                    crop_pred = (F.interpolate(crop_pred, size=(h1 * args.scale_factor, w1 * args.scale_factor),
                            mode="bilinear", align_corners=True) / args.scale_factor**2 )
                    crop_preds.append(crop_pred)
                crop_preds = torch.cat(crop_preds, dim=0)
                
                # splice them to the original size
                idx = 0
                pred_map = torch.zeros([b, 1, h, w]).to(self.device)
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                        idx += 1   
                
                mask = crop_masks.sum(dim=0).unsqueeze(0)
                outputs = pred_map / mask

                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)
        epoch_res = np.array(epoch_res)
        mse_val = np.sqrt(np.mean(np.square(epoch_res)))
        mae_val = np.mean(np.abs(epoch_res))
        
        epoch_res = []
        for inputs, count, name, gauss_im in self.dataloaders_shot["val"]:
            for inputs_shot, count_shot, name_shot, gausss_shot in self.dataloaders_shot["val"]:
                break
            with torch.no_grad():
                inputs_shot = inputs_shot.to(self.device)
                inputs = inputs.to(self.device)
                    
                crop_imgs_shot, crop_masks_shot = [], []
                crop_imgs, crop_masks = [], []
                b, c, h, w = inputs.size()
                bsh, csh, hsh, wsh = inputs.size()
                rh, rw = args.crop_size, args.crop_size
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        randloc = abs(int(np.random.rand(1)*inputs_shot.shape[2])-rw)
                        crop_imgs_shot.append(inputs_shot[:, :, randloc:randloc+rh, randloc:randloc+rw])
                        crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                        mask = torch.zeros([b, 1, h, w]).to(self.device)
                        mask_shot = torch.zeros([bsh, 1, hsh, wsh]).to(self.device)
                        mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                        crop_masks_shot.append(mask_shot)
                        crop_masks.append(mask)
                
                crop_imgs_shot, crop_masks_shot = map(lambda x: torch.cat(x, dim=0), (crop_imgs_shot, crop_masks_shot))
                crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))
                       
                crop_preds = []
                nz, bz = crop_imgs.size(0), args.batch_size
                for i in range(0, nz, bz):
                    
                    gs, gt = i, min(nz, i + bz)
                    crop_pred,_,_,_ = self.model(crop_imgs[gs:gt],crop_imgs[gs:gt])  
                    crop_pred = crop_pred[0]   
                    _, _, h1, w1 = crop_pred.size()                
                    crop_pred = (F.interpolate(crop_pred, size=(h1 * args.scale_factor, w1 * args.scale_factor),
                            mode="bilinear", align_corners=True) / args.scale_factor**2 )
                    crop_preds.append(crop_pred)
                crop_preds = torch.cat(crop_preds, dim=0)
                
                # splice them to the original size
                idx = 0
                pred_map = torch.zeros([b, 1, h, w]).to(self.device)
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                        idx += 1   
                
                mask = crop_masks.sum(dim=0).unsqueeze(0)
                outputs = pred_map / mask

                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)
        epoch_res = np.array(epoch_res)
        mse_shot = np.sqrt(np.mean(np.square(epoch_res)))
        mae_shot = np.mean(np.abs(epoch_res))
        
        mae = mae_val + mae_shot
        mse = mse_val + mse_shot
        

        self.logger.info("Epoch {} Val, MSE: {:.2f}, MAE: {:.2f}, Cost {:.1f} sec".format(
                self.epoch, mse, mae, time.time() - epoch_start ))

        model_state_dic = self.model.state_dict()
        print("Comaprison", mae,  self.best_mae)
        if mae < self.best_mae:
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info(
                "save best mse {:.2f} mae {:.2f} model epoch {}".format(
                    self.best_mse, self.best_mae, self.epoch))
                    
            print("Saving best model at {} epoch".format(self.epoch))
            model_path = os.path.join(
                self.save_dir, "best_model_mae-{:.2f}_epoch-{}.pth".format(
                    self.best_mae, self.epoch))
                    
            torch.save(model_state_dic, model_path)


if __name__ == "__main__":
    import torch
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()

    





