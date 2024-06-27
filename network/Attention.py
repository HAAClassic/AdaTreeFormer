r""" Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

from network.swin_transformer_v2 import SwinTransformerV2 as SwinTransformer
from network.transformer_attention import MultiHeadedAttention, PositionalEncoding
import numpy as np
from scipy.io import savemat 


class QKV(nn.Module):

    def __init__(self, backbone, pretrained_path, use_original_imgsize):
        super(QKV, self).__init__()

        self.backbone = backbone
        self.use_original_imgsize = use_original_imgsize

        # feature extractor initialization
        if backbone == 'resnet50':
            self.feature_extractor = resnet.resnet50()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))
        elif backbone == 'resnet101':
            self.feature_extractor = resnet.resnet101()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 23, 3]
            self.feat_ids = list(range(0, 34))
        elif backbone == 'swin':
            self.feature_extractor = SwinTransformer(img_size=256, patch_size=4, window_size=16, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            self.feature_extractor.load_state_dict(torch.load(pretrained_path)['model'])
            
            self.feat_channels = [128, 256, 512, 1024]
            self.nlayers = [2, 2, 18, 2]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.feature_extractor.eval()

        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)
        self.model = QKV_model(in_channels=self.feat_channels, stack_ids=self.stack_ids)


    def forward(self, query_img, support_img):
        
        # with torch.no_grad():
        query_feats = self.extract_feats(query_img)
        support_feats = self.extract_feats(support_img)
        
        
        s2s_mask, att_s2s = self.model(query_feats, query_feats) 
        t2t_mask, att_t2t = self.model(support_feats, support_feats)
        
        vol_qs = torch.cat((query_img,support_img),0)
        vol_sq = torch.cat((support_img,query_img),0)
        vol_qs_feats = self.extract_feats(vol_qs)
        vol_sq_feats = self.extract_feats(vol_sq)

        volume_mask, volume_att = self.model(vol_qs_feats,vol_sq_feats)
        dim = int(volume_mask.size()[0]/2)            
        
        s2t_mask = volume_mask[:dim]
        att_s2t = [volume_att[0][:dim],volume_att[1][:dim],volume_att[2][:dim]]
        t2s_mask = volume_mask[dim:]
        att_t2s = [volume_att[0][dim:],volume_att[1][dim:],volume_att[2][dim:]]
        
       
        return [s2s_mask,att_s2s], [t2s_mask,att_s2t], [s2t_mask,att_t2s], [t2t_mask,att_t2t]

    def extract_feats(self, img):
        r""" Extract input image features """
        feats = []
                
        if self.backbone == 'swin':
            _ = self.feature_extractor.forward_features(img)
            for feat in self.feature_extractor.feat_maps:
                bsz, hw, c = feat.size()
                h = int(hw ** 0.5)
                feat = feat.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
                feats.append(feat)
                
                
        elif self.backbone == 'resnet50' or self.backbone == 'resnet101':
            bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
            # Layer 0
            feat = self.feature_extractor.conv1.forward(img)
            feat = self.feature_extractor.bn1.forward(feat)
            feat = self.feature_extractor.relu.forward(feat)
            feat = self.feature_extractor.maxpool.forward(feat)

            # Layer 1-4
            for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
                res = feat
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

                if bid == 0:
                    res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

                feat += res

                if hid + 1 in self.feat_ids:
                    feats.append(feat.clone())

                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats


    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.feature_extractor.eval()


class QKV_model(nn.Module):
    def __init__(self, in_channels, stack_ids):
        super(QKV_model, self).__init__()

        self.stack_ids = stack_ids

        # QKV blocks
        self.QKV_blocks = nn.ModuleList()
        self.pe = nn.ModuleList()
        for inch in in_channels[1:]:
            self.QKV_blocks.append(MultiHeadedAttention(h=8, d_model=inch, dropout=0.5))
            self.pe.append(PositionalEncoding(d_model=inch, dropout=0.5))

        outch1, outch2, outch3 = 16, 64, 128

        # conv blocks
        self.conv1 = self.build_conv_block(128, [256, 512, 1024], [3, 3, 3], [1, 1, 1]) # 1/32
        self.conv2 = self.build_conv_block(4608, [2048, 1024, 512], [5, 3, 3], [1, 1, 1]) # 1/16
        self.conv3 = self.build_conv_block(2048, [1024, 512, 256], [5, 5, 3], [1, 1, 1]) # 1/8
        
        for inch in (1024,512,256):    
            self.QKV_blocks.append(MultiHeadedAttention(h=8, d_model=inch, dropout=0.5))
            self.pe.append(PositionalEncoding(d_model=inch, dropout=0.5))
            
        self.conv4 = self.build_conv_block(64, [128, 256, 512], [3, 3, 3], [1, 1, 1]) 

        # mixer blocks
        self.mixer1 = nn.Sequential(nn.Conv2d(1792, 1024, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 512, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer2 = nn.Sequential(nn.Conv2d(512, 256, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 128, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer3 = nn.Sequential(nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

    def forward(self, query_feats, support_feats):
        coarse_masks = []
        for idx, query_feat in enumerate(query_feats):
            # 1/4 scale feature only used in skip connect
            if idx < self.stack_ids[0]: continue
        
            bsz, ch, ha, wa = query_feat.size()
            # reshape the input feature and mask
            query = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous()
            support_feat = support_feats[idx]
            support_feat = support_feat.view(support_feat.size()[0], support_feat.size()[1], -1).permute(0, 2, 1).contiguous()

            # QKV blocks forward
            if idx < self.stack_ids[1]:
                coarse_mask = self.QKV_blocks[0](self.pe[0](query), self.pe[0](support_feat))
            elif idx < self.stack_ids[2]:
                coarse_mask = self.QKV_blocks[1](self.pe[1](query), self.pe[1](support_feat))
            else:
                coarse_mask = self.QKV_blocks[2](self.pe[2](query), self.pe[2](support_feat))
            coarse_masks.append(coarse_mask.permute(0, 2, 1).contiguous().view(bsz, ha*wa, ha, wa))

        # multi-scale conv blocks forward
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[3]-1-self.stack_ids[0]].size()
        Attention_1 = torch.stack(coarse_masks[self.stack_ids[2]-self.stack_ids[0]:self.stack_ids[3]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[2]-1-self.stack_ids[0]].size()
        Attention_2 = torch.stack(coarse_masks[self.stack_ids[1]-self.stack_ids[0]:self.stack_ids[2]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[1]-1-self.stack_ids[0]].size()
        Attention_3 = torch.stack(coarse_masks[0:self.stack_ids[1]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
        
        coarse_masks1 = self.conv1(Attention_1)
        coarse_masks2 = self.conv2(Attention_2)
        coarse_masks3 = self.conv3(Attention_3)
        
        v_decoder_1 = query_feats[23].view(bsz, 1024, -1).permute(0, 2, 1).contiguous()# + support_feats[23].view(bsz, 1024, -1).permute(0, 2, 1).contiguous()
        qk_encoder_1 = coarse_masks1.view(bsz, 1024, -1).permute(0, 2, 1).contiguous()
        qkv_ed_1 = self.QKV_blocks[3](self.pe[3](qk_encoder_1), self.pe[3](v_decoder_1)).transpose(0, 1).contiguous().view(bsz, -1, 8, 8)
        
        v_decoder_2 = F.interpolate(self.conv4(qkv_ed_1), scale_factor = 2, mode='bilinear', align_corners=True)
        
        v_decoder_2 = v_decoder_2.view(bsz, 512, -1).permute(0, 2, 1).contiguous()
        qk_encoder_2 = coarse_masks2.view(bsz, 512, -1).permute(0, 2, 1).contiguous()
        qkv_ed_2 = self.QKV_blocks[4](self.pe[4](qk_encoder_2), self.pe[4](v_decoder_2)).transpose(0, 1).contiguous().view(bsz, -1, 16, 16)
        
        v_decoder_3 = F.interpolate(qkv_ed_2, scale_factor = 2, mode='bilinear', align_corners=True)
        v_decoder_3 = v_decoder_3.view(bsz, 256, -1).permute(0, 2, 1).contiguous()
        qk_encoder_3 = coarse_masks3.view(bsz, 256, -1).permute(0, 2, 1).contiguous()
        qkv_ed_3 = self.QKV_blocks[5](self.pe[5](qk_encoder_3), self.pe[5](v_decoder_3)).transpose(0, 1).contiguous().view(bsz, -1, 32, 32)
        
        support_skip_1 = support_feats[self.stack_ids[1] - 1]
        query_skip_1 = query_feats[self.stack_ids[1] - 1]
        mix = torch.cat((qkv_ed_3, support_skip_1, query_skip_1), 1)
        
        mix = F.interpolate(mix, scale_factor = 2, mode='bilinear', align_corners=True) 
        
        support_skip_2 = support_feats[self.stack_ids[0] - 1]
        query_skip_2 = query_feats[self.stack_ids[0] - 1]
        mix = torch.cat((mix, support_skip_2, query_skip_2), 1)
       
        out = self.mixer3(self.mixer2(self.mixer1(mix)))

       
        return out, [Attention_1,Attention_2,Attention_3]

    def build_conv_block(self, in_channel, out_channels, kernel_sizes, spt_strides, group=4):
        r""" bulid conv blocks """
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            pad = ksz // 2

            building_block_layers.append(nn.Conv2d(in_channels=inch, out_channels=outch,
                                                   kernel_size=ksz, stride=stride, padding=pad))
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*building_block_layers)
