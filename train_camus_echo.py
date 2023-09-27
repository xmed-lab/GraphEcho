# MIT License
# Copyright (c) 2023 Jiewen Yang
# Please refer to the paper : https://arxiv.org/abs/2309.11145.

import os
import sys
import argparse
import time
import math
import random
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.data import DataLoader

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorboardX import SummaryWriter

from datasets.camus import DataLoaderCamus
from datasets.echo import Echo
from utils.tools import get_world_size, get_global_rank, get_local_rank, get_master_ip
from utils.metrics import DiceScore
from utils.lr_scheduler import WarmupMultiStepLR
from utils.sinkhorn_distance import SinkhornDistance
from utils.losses import BinaryDiceLoss, DiceLoss
from models.fpnseg import FPN, Discriminator
from models.graph_matching import GModule
from models.TGCN import TGCN


torch.autograd.set_detect_anomaly(True)
matplotlib.use('Agg')
os.environ['CUDA_ENABLE_DEVICES'] = '0,1,2,3,4,5,6,7'
parts_num = {'1':2, '2':1, '3':2, '4':4} # for camus dataset we only have to chamber for segmentation


class Trainer():
    def __init__(self, config, debug=False):
        self.config = config
        self.logger = self.logger_config(log_path='log_camus_echo.txt', logging_name='experiment')
        torch.backends.cudnn.benchmark = config['train']['cudnn']
        self.distributed = config['train']['distributed']
        self.device = config['train']['device']
        self.local_rank = config['train']['local_rank']
        self.seg_parts = config['train']['seg_parts']
        self.view_num = config['train']['view_num']
        self.cyc_loss = config['train']['cyc_loss']
        self.temporal_graph = config['train']['temporal_graph']
        self.graph_matching = config['train']['graph_matching']
        self.discriminator = config['train']['discriminator']

        self.out_channels = parts_num[self.view_num[0]] if self.seg_parts else 1
        self.network = FPN([2,4,23,3], num_classes=self.out_channels, in_channel=1, back_bone="resnet")

        self.optimizer_dict, self.scheduler_dict = dict(), dict()

        self.optimizer_dict['Net'] = self.set_optimizer(self.network, config['net']['opt'])
        self.scheduler_dict['Net'] = self.set_scheduler(self.optimizer_dict['Net'], config['net']['sch'])
        self.network = self.network.to(self.device)

        if self.graph_matching:
            self.graph_model = GModule(in_channels=256, num_classes=self.out_channels, device=self.device)
            self.optimizer_dict['Graph'] = self.set_optimizer(self.graph_model, config['gmn']['opt'])
            self.scheduler_dict['Graph'] = self.set_scheduler(self.optimizer_dict['Graph'], config['gmn']['sch'])
            self.graph_model = self.graph_model.to(self.device)
        
        if self.discriminator and self.graph_matching:
            self.dis_dict = dict()
            self.dis_dict['dis_p2'] = Discriminator(grad_reverse_lambda=0.02)
            self.dis_dict['dis_p3'] = Discriminator(grad_reverse_lambda=0.02)
            self.dis_dict['dis_p4'] = Discriminator(grad_reverse_lambda=0.02)
            self.dis_dict['dis_p5'] = Discriminator(grad_reverse_lambda=0.02)

            self.optimizer_dict['Dis_P2'] = self.set_optimizer(self.dis_dict['dis_p2'], config['dis']['opt'])
            self.optimizer_dict['Dis_P3'] = self.set_optimizer(self.dis_dict['dis_p3'], config['dis']['opt'])
            self.optimizer_dict['Dis_P4'] = self.set_optimizer(self.dis_dict['dis_p4'], config['dis']['opt'])
            self.optimizer_dict['Dis_P5'] = self.set_optimizer(self.dis_dict['dis_p5'], config['dis']['opt'])

            self.scheduler_dict['Dis_P2'] = self.set_scheduler(self.optimizer_dict['Dis_P2'], config['dis']['sch'])
            self.scheduler_dict['Dis_P3'] = self.set_scheduler(self.optimizer_dict['Dis_P3'], config['dis']['sch'])
            self.scheduler_dict['Dis_P4'] = self.set_scheduler(self.optimizer_dict['Dis_P4'], config['dis']['sch'])
            self.scheduler_dict['Dis_P5'] = self.set_scheduler(self.optimizer_dict['Dis_P5'], config['dis']['sch'])

            for key in self.dis_dict.keys():
                self.dis_dict[key].to(self.device)

        if self.temporal_graph:
            self.train_dataset_source_temp = DataLoaderCamus(dataset_path='please indicate your CAMUS dataset path',input_name="4CH_ED",target_name="4CH_ED",condition_name="4CH_ED_gt",stage="train", img_res=(124, 124), img_crop=(112, 112))
            self.train_dataset_target_temp = Echo(root='please indicate your Echonet dataset path', validation=False, split='train')

            self.train_loader_source_temp = DataLoader(self.train_dataset_source_temp, batch_size=4, shuffle=True, num_workers=config['train']['num_workers'])
            self.train_loader_target_temp = DataLoader(self.train_dataset_target_temp, batch_size=4, shuffle=True, num_workers=config['train']['num_workers'])

            self.source_data_num, self.target_data_num = self.train_dataset_source_temp.num_data, self.train_dataset_target_temp.num_data

            self.tgcn_dict, self.g_optimizer_dict, self.g_scheduler_dict = dict(), dict(), dict()
            # self.tgcn_dict['tgcn_p2'] = TGCN(input_dim=256, hidden_dim=256, soucre_class=self.source_data_num, target_class=self.target_data_num)
            # self.tgcn_dict['tgcn_p3'] = TGCN(input_dim=256, hidden_dim=256, soucre_class=self.source_data_num, target_class=self.target_data_num)
            # self.tgcn_dict['tgcn_p4'] = TGCN(input_dim=256, hidden_dim=256, soucre_class=self.source_data_num, target_class=self.target_data_num)
            self.tgcn_dict['tgcn_p5'] = TGCN(input_dim=256, hidden_dim=256, clip_shape=(8, 8, 8), soucre_class=self.source_data_num, target_class=self.target_data_num)

            # self.g_optimizer_dict['tgcn_p2'] = self.set_optimizer(self.tgcn_dict['tgcn_p2'], config['tgcn']['opt'])
            # self.g_optimizer_dict['tgcn_p3'] = self.set_optimizer(self.tgcn_dict['tgcn_p3'], config['tgcn']['opt'])
            # self.g_optimizer_dict['tgcn_p4'] = self.set_optimizer(self.tgcn_dict['tgcn_p4'], config['tgcn']['opt'])
            self.g_optimizer_dict['tgcn_p5'] = self.set_optimizer(self.tgcn_dict['tgcn_p5'], config['tgcn']['opt'])

            # self.g_scheduler_dict['tgcn_p2'] = self.set_scheduler(self.g_optimizer_dict['tgcn_p2'], config['tgcn']['sch'])
            # self.g_scheduler_dict['tgcn_p3'] = self.set_scheduler(self.g_optimizer_dict['tgcn_p3'], config['tgcn']['sch'])
            # self.g_scheduler_dict['tgcn_p4'] = self.set_scheduler(self.g_optimizer_dict['tgcn_p4'], config['tgcn']['sch'])
            self.g_scheduler_dict['tgcn_p5'] = self.set_scheduler(self.g_optimizer_dict['tgcn_p5'], config['tgcn']['sch'])

            for key in self.tgcn_dict.keys():
                self.tgcn_dict[key].to(self.device)

        #self.load()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean").to(self.device)
        self.ce_loss  = nn.CrossEntropyLoss().to(self.device)
        self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=5, reduction='mean').to(self.device)
        self.dice_loss = DiceLoss().to(self.device)

        if self.distributed:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.network = torch.nn.parallel.DistributedDataParallel(
                self.network,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=True,)

            for key in self.dis_dict.keys():
                self.dis_dict[key] = torch.nn.parallel.DistributedDataParallel(self.dis_dict[key], device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=True, find_unused_parameters=True)

            for key in self.tgcn_dict.keys():
                self.tgcn_dict[key] = torch.nn.parallel.DistributedDataParallel(self.tgcn_dict[key], device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False, find_unused_parameters=True)

        elif len(config['train']['enable_GPUs_id']) > 1:
            self.network = nn.DataParallel(self.network, device_ids=config['train']['enable_GPUs_id'], output_device=config['train']['enable_GPUs_id'][0])

            for key in self.dis_dict.keys():
                self.dis_dict[key]  = nn.DataParallel(self.dis_dict[key],  device_ids=config['train']['enable_GPUs_id'])

            for key in self.tgcn_dict.keys():
                self.tgcn_dict[key] = nn.DataParallel(self.tgcn_dict[key], device_ids=config['train']['enable_GPUs_id'])

        self.print_allow = True if self.local_rank == config['train']['enable_GPUs_id'][0] else False
        
        self.train_dataset_source = DataLoaderCamus(dataset_path='please indicate your CAMUS dataset path',input_name="4CH_ED",target_name="4CH_ED",condition_name="4CH_ED_gt",stage="train", img_res=(124, 124), img_crop=(112, 112))
        self.train_dataset_target = Echo(root='please indicate your Echonet dataset path', validation=False, split='train')

        self.sampler = None
        if self.distributed:
           self.sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, num_replicas=self.config['train']['world_size'], rank=self.local_rank, shuffle=True)
        
        self.train_loader_source = DataLoader(self.train_dataset_source, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['train']['num_workers'], drop_last=True)
        
        if self.graph_matching:
            self.train_loader_target = DataLoader(self.train_dataset_target, batch_size=config['train']['batch_size'] * 21, shuffle=True, num_workers=config['train']['num_workers'])

        self.valid_echo_dataset  = Echo(root='please indicate your Echonet dataset path', validation=True, split='VAL', target_type='LargeTrace')
        self.valid_echo_loader  = DataLoader(self.valid_echo_dataset, batch_size=1, shuffle=False, num_workers=config['train']['num_workers'])

        self.test_echo_dataset  = Echo(root='please indicate your Echonet dataset path', validation=True, split='TEST', target_type='LargeTrace')
        self.test_echo_loader  = DataLoader(self.test_echo_dataset, batch_size=1, shuffle=False, num_workers=config['train']['num_workers'])
        
        self.valid_camus_dataset  = DataLoaderCamus(dataset_path='please indicate your CAMUS dataset path',input_name="4CH_ED",target_name="4CH_ED",condition_name="4CH_ED_gt",stage="valid", img_res=(124, 124), img_crop=(112, 112))
        self.valid_camus_loader  = DataLoader(self.valid_camus_dataset, batch_size=1, shuffle=False, num_workers=config['train']['num_workers'])

        self.test_camus_dataset  = DataLoaderCamus(dataset_path='please indicate your CAMUS dataset path',input_name="4CH_ED",target_name="4CH_ED",condition_name="4CH_ED_gt",stage="test", img_res=(124, 124), img_crop=(112, 112))
        self.test_camus_loader  = DataLoader(self.test_camus_dataset, batch_size=1, shuffle=False, num_workers=config['train']['num_workers'])
        

        if self.print_allow:
            self.writer = SummaryWriter(os.path.join(config['train']['log_dir']))

    def train(self):
        count = 0
        losses = {}

        for self.epoch in range(self.config['train']['num_epochs']):
            if self.print_allow:
                print('Start Epoch / Total Epoch: {} / {}'.format(self.epoch, self.config['train']['num_epochs']))

            y_true, y_pred = [], []
            if self.graph_matching:
                train_loader_target = iter(self.train_loader_target)
                self.graph_model.train()
            
            if self.cyc_loss:
                train_cyc_loader = iter(self.train_cyc_loader)
            
            if self.temporal_graph:
                train_loader_source_temp = iter(self.train_loader_source_temp)
                train_loader_target_temp = iter(self.train_loader_target_temp)

            self.network.train()

            progress_bar = tqdm(self.train_loader_source) if self.print_allow else self.train_loader_source
            for self.step, (imgs_source, masks, _, _) in enumerate(progress_bar):
                imgs_source = imgs_source.to(self.device)
                pred_source, features_source = self.network(imgs_source)
                
                masks = masks.to(self.device) / 1.0
                masks = masks[:, :1, ...]
                seg_loss = 0.1 * (self.dice_loss(pred_source, masks) + self.bce_loss(pred_source, masks)) / 2
                losses.update({'seg_loss': seg_loss})

                if self.graph_matching:
                    imgs_target, _, _, _ = train_loader_target.next()
                    imgs_target = imgs_target.to(self.device)  
                    pred_target, features_target = self.network(imgs_target)
                    score_maps = torch.where(nn.Sigmoid()(pred_target) > 0.5, 1, 0)
                    (features_s, features_t), _, middle_head_loss = \
                        self.graph_model((imgs_source, imgs_target), (features_source, features_target), targets=masks, score_maps=score_maps)
                    losses.update(middle_head_loss)

                    if self.discriminator:
                        for layer, layer_name in enumerate(['p2', 'p3', 'p4', 'p5']):
                            losses["loss_adv_%s" % layer_name] = \
                                0.1 * self.dis_dict["dis_%s" % layer_name]((features_s[layer],features_t[layer]))
                
                for name in self.optimizer_dict:
                    self.optimizer_dict[name].zero_grad()

                if self.temporal_graph:
                    temporal_graph_loss, temp_seg_loss = 0, 0
                    graph_features, source_features_, target_features_ = list(), list(), list()
                    source_masks_ = list()

                    imgs_source_temp, source_temp_masks, _, update_index_source = train_loader_source_temp.next()
                    imgs_target_temp, _, _, update_index_target = train_loader_target_temp.next()

                    update_index_source = update_index_source.to(self.device)
                    update_index_target = update_index_target.to(self.device)
                    update_index = torch.cat([update_index_source, torch.add(update_index_target, 150)])

                    imgs_source_temp = imgs_source_temp.to(self.device)
                    imgs_target_temp = imgs_target_temp.to(self.device)
                    imgs_temp = torch.cat([imgs_source_temp, imgs_target_temp], dim=0)
                    b, c, h, w, t = imgs_temp.shape
                    imgs_temp = imgs_temp.permute(0,4,1,2,3).reshape(-1, c, h, w)

                    source_temp_masks = source_temp_masks.to(self.device)
                    source_temp_masks = source_temp_masks.permute(0,4,1,2,3).reshape(b*t//2, -1, h, w) / 1.0

                    masks_select = torch.where(torch.sum(source_temp_masks, dim=(1,2,3)) > 100, 1, 0) 
                    preds_, features_ = self.network(imgs_temp)

                    pred_source_temp = preds_[:b*t//2]
                    for select_index, is_available in enumerate(masks_select):
                        if is_available:
                            source_masks_.append(source_temp_masks[select_index].unsqueeze(0))
                            temp_seg_loss += self.dice_loss(pred_source_temp[select_index], source_temp_masks[select_index]) + \
                                             self.bce_loss(pred_source_temp[select_index], source_temp_masks[select_index])
                        else:
                            source_masks_.append(pred_source_temp[select_index].unsqueeze(0))
                    source_masks_ = torch.cat(source_masks_, dim=0)

                    for feature in features_:
                        bt, c, h, w = feature.shape
                        source_features_.append(feature[:bt//2])
                        target_features_.append(feature[bt//2:])

                    (_, _), (source_nodes, target_nodes), temp_middle_head_loss = \
                        self.graph_model((imgs_temp[:b*t//2], imgs_temp[b*t//2:]), (source_features_, target_features_), targets=source_masks_, score_maps=preds_[b*t//2:])

                    for i, feature in enumerate(features_):
                        bt, c, h, w = feature.shape
                        graph_features.append(feature.reshape(b, -1, c, h, w))

                    temporal_graph_match_loss = self.tgcn_dict['tgcn_p5'](graph_features, (source_nodes.clone().detach(), target_nodes.clone().detach()), self.sinkhorn, self.ce_loss, (update_index_source, update_index_target), r=[8,4,2,1])
                    #ce_loss_p4, cost_p4 = self.tgcn_dict['tgcn_p4'](graph_features[-2], self.sinkhorn, self.ce_loss, (update_index_source, update_index_target))
                    #ce_loss_p3, cost_p3 = self.tgcn_dict['tgcn_p3'](graph_features[-3], self.sinkhorn, self.ce_loss, (update_index_source, update_index_target), r=2)
                    #ce_loss_p2, cost_p2 = self.tgcn_dict['tgcn_p2'](graph_features[-4], self.sinkhorn, self.ce_loss, (update_index_source, update_index_target), r=4)

                    #sinkhorn_loss = cost_p5[0]
                    #clustering_loss = ce_loss_p5
                    
                    temporal_graph_loss = sum(loss for loss in temporal_graph_match_loss.values()) + sum(loss for loss in temp_middle_head_loss.values())
                    
                    for name in self.g_optimizer_dict:
                        self.g_optimizer_dict[name].zero_grad()
                    losses.update({'temporal_graph_loss': temporal_graph_loss})

                total_loss = sum(loss for loss in losses.values())
                total_loss.backward(retain_graph=False)

                for name in self.optimizer_dict:
                    self.optimizer_dict[name].step()
                if self.temporal_graph:
                    for name in self.g_optimizer_dict:
                        self.g_optimizer_dict[name].step()

                if self.print_allow:
                    self.add_summary(self.writer, 'train/net_loss', total_loss.sum().item(), count)
                    count += 1
                    if count % len(progress_bar) == 0:
                        pixel_acc, dice, precision, specificity, recall = self._calculate_overlap_metrics(masks, torch.where(nn.Sigmoid()(pred_source) > 0.5, 1, 0))

                    if self.config['train']['record_params']:
                        for tag, value in self.network.named_parameters():
                            tag = tag.replace('.', '/').replace('module', '')
                            self.add_summary(self.writer, tag, value.data.cpu().numpy(), sum_type='histogram')
            
            for name in self.scheduler_dict:
                self.scheduler_dict[name].step()
            if self.temporal_graph:
                for name in self.g_scheduler_dict:
                    self.g_scheduler_dict[name].step()

            if self.print_allow:
                print_info = '------Training Result------\n \
                       Loss : {loss:.4f} \
                       Seg Loss : {seg_loss:.4f} \
                       Cyc Loss : {cyc_loss:.4f} \
                       Sinkhorn Loss : {sinkhorn_loss:.4f} \
                       Graph Clustering Loss : {graph_clustering_loss:.4f} \
                       Graph Temp Loss : {graph_temp_loss:.4f} \
                       Pixel Acc : {pixel_acc:.4f} \
                       Dice : {dice:.4f} \
                       Precision : {pre:.4f} \
                       Specificity : {specificity:.4f} \
                       Recall : {recall:.4f}'.\
                       format(loss=total_loss.item(), 
                              seg_loss=seg_loss.item(), 
                              cyc_loss=cyc_loss.item() if self.cyc_loss else 0,
                              sinkhorn_loss=0,
                              graph_clustering_loss=0,
                              graph_temp_loss=temporal_graph_loss.item() if self.temporal_graph else 0, 
                              pixel_acc=pixel_acc, 
                              dice=dice, pre=precision, specificity=specificity, recall=recall)
                self.logger.info(print_info)
                
                self.validation(self.valid_camus_loader, 'Inner-Val')
                # self.validation(self.test_camus_loader, 'Inner-Test')
                self.validation(self.valid_echo_loader, 'Target Domain - Valid')
                # self.validation(self.test_echo_loader, 'Target Domain - Test')
                if self.epoch > 0:
                    self.save(self.epoch)
                    print("model saved")
                print('End Training Epoch: {}'.format(self.epoch))

    def validation(self, datasets, dataset_type, is_video=False):
        count, y_true, y_pred, mse, mae, pred_frames_list, masks_list = 0, [], [], 0, 0, [], []
        self.network.eval()
        with torch.no_grad():    
            progress_bar = tqdm(datasets) if self.print_allow else datasets
            for self.step, (imgs, masks, _, _) in enumerate(progress_bar):
                imgs = imgs.to(self.device)
                masks = masks.to(self.device) / 1.0
                masks = masks[:, :1, ...]
                if is_video:
                    b, c, h, w, t = imgs.shape
                    imgs = imgs.permute(0,4,1,2,3).reshape(b*t, -1, h, w)
                    masks = masks.permute(0,4,1,2,3).reshape(b*t, -1, h, w)

                pred_frames, _ = self.network(imgs)
                pred_frames = pred_frames[:, :1, ...]
                loss = self.bce_loss(pred_frames, masks)

                pred_frames_list.append(pred_frames)
                masks_list.append(masks)

                count += 1
                self.add_summary(self.writer, 'train/net_loss', loss.item(), count)

            pred_frames_list = torch.cat(pred_frames_list, dim=0)
            masks_list = torch.cat(masks_list, dim=0)

            if self.print_allow:
                if count == len(progress_bar):
                    pixel_acc, dice, precision, specificity, recall = self._calculate_overlap_metrics(masks_list, torch.where(nn.Sigmoid()(pred_frames_list) > 0.5, 1, 0))

                print_info = '------Validation Result . {dataset_type} in |{current_epoch}/{total_epoch}| ------\n \
                       Loss : {loss:.4f} \
                       Pixel Acc : {pixel_acc:.4f} \
                       Dice : {dice:.4f} \
                       Precision : {pre:.4f} \
                       Specificity : {specificity:.4f} \
                       Recall : {recall:.4f}'.\
                       format(dataset_type=dataset_type, current_epoch=self.epoch, total_epoch=self.config['train']['num_epochs'],
                              loss=loss.item(), pixel_acc=pixel_acc, dice=dice, pre=precision, specificity=specificity, recall=recall)
                self.logger.info(print_info)

            if self.seg_parts:
                for part in range(pred_frames.shape[1]):
                    pred_view = pred_frames_list[:, part]
                    select_masks = masks_list[:, part]
                    _, part_dice, _, _, _ = self._calculate_overlap_metrics(select_masks, torch.where(nn.Sigmoid()(pred_view) > 0.5, 1, 0))
                    print('Part Result . ------ {part_num} ------ . \
                       Dice : {dice:.4f} '.\
                       format(part_num=part, dice=part_dice))


    def _calculate_overlap_metrics(self, gt, pred, eps=1e-5):
        output = pred.reshape(-1, )
        target = gt.reshape(-1, ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        specificity = (tn + eps) / (tn + fp + eps)

        return pixel_acc, dice, precision, specificity, recall

    def adjust_learning_rate(self, optimizer, epoch, args):
        """Decay the learning rate based on schedule"""
        cur_lr = self.config['net']['opt']['lr'] * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr

    def set_optimizer(self, net, config):
        if config['opt_name'] == 'SGD':
            optimizer = torch.optim.SGD(list(net.parameters()), 
                                        lr=config['lr'],
                                        weight_decay=config['weight_decay'], momentum=config['momentum'])
        elif config['opt_name'] == 'Adam':
            optimizer = torch.optim.Adam(list(net.parameters()), 
                                        lr=config['lr'], 
                                        weight_decay=config['weight_decay'])

        return optimizer

    def set_scheduler(self, optimizer, config):
        return WarmupMultiStepLR(
            optimizer,
            config['STEPS'],
            config['GAMMA'],
            warmup_factor=config['WARMUP_FACTOR'],
            warmup_iters=config['WARMUP_ITERS'],
            warmup_method=config['WARMUP_METHOD'],
        )

    def load(self):
        model_path = self.config['train']['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(
                model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
                os.path.join(model_path, '*.pth'))]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None
        if latest_epoch is not None:
            net_path = os.path.join(
                model_path, 'net_{}.pth'.format(str(latest_epoch).zfill(5)))
            #opt_path = os.path.join(
            #    model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))

            if self.local_rank == self.config['train']['enable_GPUs_id'][0]:
                print('Loading model from {}...'.format(net_path))

            data = torch.load(net_path, map_location=self.device)
            data['network'] = {k.replace('module.', ''): v for k, v in data['network'].items() if k.replace('module.', '') in self.network.state_dict()}
            self.network.load_state_dict(data['network'])

        else:
            if self.local_rank == config['train']['enable_GPUs_id'][0] == 0:
                print(
                    'Warnning: There is no trained model found. An initialized model will be used.')

    def save(self, it):
        if self.local_rank == self.config['train']['enable_GPUs_id'][0]:
            net_path = os.path.join(
                self.config['train']['save_dir'], 'net_{}.pth'.format(str(it).zfill(5)))
            opt_path = os.path.join(
                self.config['train']['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))
            print('\nsaving model to {} ...'.format(net_path))
            if isinstance(self.network, torch.nn.DataParallel) or isinstance(self.network, torch.utils.data.distributed.DistributedSampler):
                network = self.network.module
            else:
                network = self.network
            torch.save({'network': network.state_dict()}, net_path)
            os.system('echo {} > {}'.format(str(it).zfill(5),
                                            os.path.join(self.config['train']['save_dir'], 'latest.ckpt')))
    # add summary
    def add_summary(self, writer, name, val, count, sum_type = 'scalar'):
        def writer_in(writer, name, val, sum_type, count):
            if sum_type == 'scalar':
                writer.add_scalar(name, val, count)
            elif sum_type == 'image':
                writer.add_image(name, val, count)
            elif sum_type == 'histogram':
                writer.add_histogram(name, val, count)

        writer_in(writer, name, val, sum_type, count)

    def logger_config(self, log_path, logging_name):
        logger = logging.getLogger(logging_name)
        logger.setLevel(level=logging.DEBUG)
        handler = logging.FileHandler(log_path)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.addHandler(console)

        return logger

def main(rank, config):

    if 'local_rank' not in config:
        if config['train']['distributed']:
            config['train']['global_rank'] = rank
            config['train']['local_rank'] = config['train']['enable_GPUs_id'][rank]
        else:
            config['train']['global_rank'] = config['train']['local_rank'] = rank

    if config['train']['distributed']:
        torch.cuda.set_device(int(config['train']['local_rank']))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=config['train']['init_method'],
                                             world_size=config['train']['world_size'],
                                             rank=config['train']['global_rank'],
                                             group_name='mtorch'
                                             )
        print('using GPU {}-{} for training'.format(
            int(config['train']['global_rank']), int(config['train']['local_rank'])))

    if torch.cuda.is_available(): 
        config['train']['device'] = torch.device("cuda:{}".format(config['train']['local_rank']))
    else: 
        config['train']['device'] = 'cpu'

    Train_ = Trainer(config)
    Train_.train()


if __name__ == "__main__":
    config = {
                "train":{
                        "cudnn": True,
                        "enable_GPUs_id": [4],
                        "batch_size": 8,
                        "num_workers": 8,
                        "num_epochs": 400,
                        "view_num": ['2'],
                        "cyc_loss": False,
                        "temporal_graph": False,
                        "graph_matching" : True,
                        "discriminator" : True,
                        "seg_parts": True,
                        "record_params": False,
                        "save_dir": './result/model/seg/view_4',
                        "log_dir": './result/log_info/log_01',
                        },

                "net":  {
                        "opt":{
                              "opt_name": 'Adam',
                              "lr": 3e-4,
                              "params": (0.9, 0.999),
                              "weight_decay": 1e-4,
                              'momentum': 0.9,
                              },
                        
                        "sch":{
                              "STEPS": (90000,),
                              "GAMMA": 0.1,
                              "WARMUP_FACTOR": 1/3,
                              "WARMUP_ITERS": 1000,
                              "WARMUP_METHOD": 'constant',
                              },
                        },
                
                "gmn":  {
                        "opt":{
                              "opt_name": 'SGD',
                              "lr": 0.0025,
                              "params": (0.9, 0.999),
                              "weight_decay": 1e-4,
                              'momentum': 0.9,
                              },

                        "sch":{
                              "STEPS": (90000,),
                              "GAMMA": 0.1,
                              "WARMUP_FACTOR": 1/3,
                              "WARMUP_ITERS": 1000,
                              "WARMUP_METHOD": 'constant',
                              },
                        },
                
                "tgcn":  {
                        "opt":{
                              "opt_name": 'SGD',
                              "lr": 0.0025,
                              "params": (0.9, 0.999),
                              "weight_decay": 1e-4,
                              'momentum': 0.9,
                              },

                        "sch":{
                              "STEPS": (90000,),
                              "GAMMA": 0.1,
                              "WARMUP_FACTOR": 1/3,
                              "WARMUP_ITERS": 1000,
                              "WARMUP_METHOD": 'constant',
                              },
                        },

                "dis":  {
                        "opt":{
                              "opt_name": 'SGD',
                              "lr": 0.0025,
                              "params": (0.9, 0.999),
                              "weight_decay": 1e-4,
                              'momentum': 0.9,
                              },

                        "sch":{
                              "STEPS": (90000,),
                              "GAMMA": 0.1,
                              "WARMUP_FACTOR": 1/3,
                              "WARMUP_ITERS": 1000,
                              "WARMUP_METHOD": 'constant',
                              },
                        },

              }

    # setting distributed configurations
    config['train']['world_size'] = 1
    config['train']['init_method'] = f"tcp://{get_master_ip()}:{23455}"
    config['train']['distributed'] = True if config['train']['world_size'] > 1 else False

    # setup distributed parallel training environments
    if get_master_ip() == "127.0.0.1" and config['train']['distributed']:
        # manually launch distributed processes 
        torch.multiprocessing.spawn(main, nprocs=config['train']['world_size'], args=(config,))
    else:
        # multiple processes have been launched by openmpi
        config['train']['local_rank'] = config['train']['enable_GPUs_id'][0]
        config['train']['global_rank'] = config['train']['enable_GPUs_id'][0]

    main(config['train']['local_rank'], config)