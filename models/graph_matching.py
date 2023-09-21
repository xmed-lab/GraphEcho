import torch
import torch.nn.functional as F
from torch import nn

import sklearn.cluster as cluster

import numpy as np

from PIL import Image, ImageColor, ImageDraw, ImageFont

from models.transformer import MultiHeadAttention
from models.affinity_layer import Affinity
from models.gradient_reversal import GradientReversal

'''
from transformer import MultiHeadAttention
from affinity_layer import Affinity
from gradient_reversal import GradientReversal
'''

INF = 100000000

class BCEFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = _input
        alpha = self.alpha

        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction =='pos':
            loss = torch.sum(loss) / (2*pos)


        return loss


class GRAPHHead(torch.nn.Module):
    # Project the sampled visual features to the graph embeddings:
    # visual features: [0,+INF) -> graph embedding: (-INF, +INF)
    def __init__(self, cfg, in_channels, out_channel, mode='in'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(GRAPHHead, self).__init__()
        if mode == 'in':
            num_convs = cfg.MODEL.MIDDLE_HEAD.NUM_CONVS_IN
        elif mode == 'out':
            num_convs = cfg.MODEL.MIDDLE_HEAD.NUM_CONVS_OUT
        else:
            num_convs = cfg.MODEL.FCOS.NUM_CONVS
            print('undefined num_conv in middle head')

        middle_tower = []
        for i in range(num_convs):
            middle_tower.append(
                nn.Conv2d(
                    in_channels,
                    out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            if mode == 'in':
                if cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'GN':
                    middle_tower.append(nn.GroupNorm(32, in_channels))
                elif cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'IN':
                    middle_tower.append(nn.InstanceNorm2d(in_channels))
                elif cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'BN':
                    middle_tower.append(nn.BatchNorm2d(in_channels))
            if i != (num_convs - 1):
                middle_tower.append(nn.ReLU())

        self.add_module('middle_tower', nn.Sequential(*middle_tower))

        for modules in [self.middle_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        middle_tower = []
        for l, feature in enumerate(x):
            middle_tower.append(self.middle_tower(feature))
        return middle_tower


class GModule(torch.nn.Module):

    def __init__(self, in_channels, num_classes, device):
        super(GModule, self).__init__()

        init_item = []
        self.device = device
        self.fpn_strides                = [8, 16, 32, 64, 128]
        self.num_classes                = num_classes
        self.matching_loss_type         = 'FL'
        
        # One-to-one (o2o) matching or many-to-many (m2m) matching?
        self.matching_cfg               = 'o2o' # 'o2o' and 'm2m'
        self.with_cluster_update        = True # add spectral clustering to update seeds
        self.with_semantic_completion   = True # generate hallucination nodes
        
        # add quadratic matching constraints.
        #TODO qudratic matching is not very stable in end-to-end training
        self.with_quadratic_matching    = True

        # Several weights hyper-parameters
        self.weight_matching            = 0.1
        self.weight_nodes               = 1.0
        self.weight_dis                 = 0.1
        self.lambda_dis                 = 0.02

        # Detailed settings
        self.with_domain_interaction    = True
        self.with_complete_graph        = True
        self.with_node_dis              = True
        self.with_global_graph          = False

        # Test 3 positions to put the node alignment discriminator. (the former is better)
        self.node_dis_place             = 'feat'

        # future work
        self.with_cond_cls              = False # use conditional kernel for node classification? (didn't use)
        self.with_score_weight          = False # use scores for node loss (didn't use)

        # Node sampling
        self.graph_generator            = PrototypeComputation(num_classes)

        # Pre-processing for the vision-to-graph transformation
        self.head_in_cfg = 'LN'
        if self.head_in_cfg != 'LN':
            self.head_in = GRAPHHead(in_channels, in_channels, mode='in')
        else:
            self.head_in_ln = nn.Sequential(
                nn.Linear(256, 256),
                nn.LayerNorm(256, elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256, elementwise_affine=False),
            )
            init_item.append('head_in_ln')

        # node classification layers
        self.node_cls_middle = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        ) 
        init_item.append('node_cls_middle')
        
        # Graph-guided Memory Bank
        self.seed_project_left = nn.Linear(256, 256) # projection layer for the node completion
        self.register_buffer('sr_seed', torch.randn(self.num_classes, 256)) # seed = bank
        self.register_buffer('tg_seed', torch.randn(self.num_classes, 256))

        # We directly utilize the singe-head attention for the graph aggreagtion and cross-graph interaction, 
        # which will be improved in our future work
        self.cross_domain_graph = MultiHeadAttention(256, 1, dropout=0.1, version='v2') # Cross Graph Interaction
        self.intra_domain_graph = MultiHeadAttention(256, 1, dropout=0.1, version='v2') # Intra-domain graph aggregation

        # Semantic-aware Node Affinity
        self.node_affinity = Affinity(d=256)
        self.InstNorm_layer = nn.InstanceNorm2d(1)

        # Structure-aware Matching Loss
        # Different matching loss choices
        if self.matching_loss_type == 'L1':
            self.matching_loss = nn.L1Loss(reduction='sum')
        elif self.matching_loss_type == 'MSE':
            self.matching_loss = nn.MSELoss(reduction='sum')
        elif self.matching_loss_type == 'FL':
            self.matching_loss = BCEFocalLoss()
        self.quadratic_loss = torch.nn.L1Loss(reduction='mean')

        if self.with_node_dis: 
            self.grad_reverse = GradientReversal(self.lambda_dis)
            self.node_dis_2 = nn.Sequential(
                nn.Linear(256,256),
                nn.LayerNorm(256,elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.LayerNorm(256,elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256,elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256,1)
            )
            init_item.append('node_dis')
            self.loss_fn = nn.BCEWithLogitsLoss()
        self._init_weight(init_item)
    
    def _init_weight(self, init_item=None):
        nn.init.normal_(self.seed_project_left.weight, std=0.01)
        nn.init.constant_(self.seed_project_left.bias, 0)
        if 'node_dis' in init_item:
            for i in self.node_dis_2:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)
            print('node_dis initialized')
        if 'node_cls_middle' in init_item:
            for i in self.node_cls_middle:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)
            print('node_cls_middle initialized')
        if 'head_in_ln' in init_item:
            for i in self.head_in_ln:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)
            print('head_in_ln initialized')

    def forward(self, images, features, targets=None, score_maps=None):
        '''
        We have equal number of source/target feature maps
        features: [sr_feats, tg_feats]
        targets: [sr_targets, None]

        '''
        if targets is not None:
            features, nodes, feat_loss = self._forward_train(images, features, targets, score_maps)
            return features, nodes, feat_loss

        else:
            features = self._forward_inference(images, features)
            return features, None

    def _forward_train(self, images, features, targets=None, score_maps=None):
        features_s, features_t = features
        middle_head_loss = {}

        # node_1 and node_2 mean the source/target raw nodes
        # label_1 and label_2 mean the GT and pseudo labels
        nodes_1, labels_1, weights_1 = self.graph_generator(
            self.compute_locations(features_s), features_s, self.find_bbox(targets)
        )

        nodes_2, labels_2, weights_2 = self.graph_generator(
            self.compute_locations(features_t), features_t, self.find_bbox(score_maps)
        )

        # to avoid the failure of extreme cases with limited bs
        if nodes_1.size(0) < 6 or len(nodes_1.size()) == 1:
            return features, (nodes_1, nodes_2), middle_head_loss

        #  conduct node alignment to prevent overfit
        if self.with_node_dis and nodes_2 is not None and self.node_dis_place =='feat' :
            nodes_rev = self.grad_reverse(torch.cat([nodes_1, nodes_2], dim=0))
            target_1 = torch.full([nodes_1.size(0), 1], 1.0, dtype=torch.float, device=nodes_1.device)
            target_2 = torch.full([nodes_2.size(0), 1], 0.0, dtype=torch.float, device=nodes_2.device)
            tg_rev = torch.cat([target_1, target_2], dim=0)
            nodes_rev = self.node_dis_2(nodes_rev)
            node_dis_loss = self.weight_dis * self.loss_fn(nodes_rev.view(-1), tg_rev.view(-1))
            middle_head_loss.update({'dis_loss': node_dis_loss})

        # LN is conducted on the node embedding
        # GN/BN are conducted on the whole image feature
        if  self.head_in_cfg != 'LN':
            features_s = self.head_in(features_s)
            features_t = self.head_in(features_t)
            nodes_1, labels_1, weights_1 = self.graph_generator(
                self.compute_locations(features_s), features_s, targets
            )
            nodes_2, labels_2, weights_2 = self.graph_generator(
                None, features_t, score_maps
            )
        else:
            nodes_1 = self.head_in_ln(nodes_1)
            nodes_2 = self.head_in_ln(nodes_2) if nodes_2 is not None else None

        # TODO: Matching can only work for adaptation when both source and target nodes exist. 
        # Otherwise, we split the source nodes half-to-half to train SIGMA

        if nodes_2 is not None: # Both domains have graph nodes
            (nodes_1, nodes_2), (labels_1, labels_2), (weights_1, weights_2) = \
                self._forward_preprocessing_source_target((nodes_1, nodes_2), (labels_1, labels_2), (weights_1,weights_2))

            if self.with_complete_graph:
                nodes_1, edges_1 = self._forward_intra_domain_graph(nodes_1)
                nodes_2, edges_2 = self._forward_intra_domain_graph(nodes_2)

            self.update_seed(nodes_1, labels_1, nodes_2, labels_2)

            if self.with_node_dis and self.node_dis_place =='intra':
                nodes_rev = self.grad_reverse(torch.cat([nodes_1, nodes_2], dim=0))
                target_1 = torch.full([nodes_1.size(0), 1], 1.0, dtype=torch.float, device=nodes_1.device)
                target_2 = torch.full([nodes_2.size(0), 1], 0.0, dtype=torch.float, device=nodes_2.device)
                tg_rev = torch.cat([target_1, target_2], dim=0)
                nodes_rev = self.node_dis_2(nodes_rev)
                node_dis_loss = self.weight_dis * self.loss_fn(nodes_rev.view(-1), tg_rev.view(-1))
                middle_head_loss.update({'dis_loss': node_dis_loss})

            if self.with_domain_interaction:
                nodes_1, nodes_2 = self._forward_cross_domain_graph(nodes_1, nodes_2)

            if self.with_node_dis and self.node_dis_place =='inter':
                nodes_rev = self.grad_reverse(torch.cat([nodes_1, nodes_2], dim=0))
                target_1 = torch.full([nodes_1.size(0), 1], 1.0, dtype=torch.float, device=nodes_1.device)
                target_2 = torch.full([nodes_2.size(0), 1], 0.0, dtype=torch.float, device=nodes_2.device)
                tg_rev = torch.cat([target_1, target_2], dim=0)
                nodes_rev = self.node_dis_2(nodes_rev)
                node_dis_loss = self.weight_dis * self.loss_fn(nodes_rev.view(-1), tg_rev.view(-1))
                middle_head_loss.update({'dis_loss': node_dis_loss})

            node_loss = self._forward_node_loss(
                torch.cat([nodes_1, nodes_2], dim=0),
                torch.cat([labels_1, labels_2], dim=0),
                torch.cat([weights_1, weights_2], dim=0)
                                                )

        else: # Use all source nodes for training if no target nodes in the early training stage
            (nodes_1, nodes_2),(labels_1, labels_2) = \
                self._forward_preprocessing_source(nodes_1, labels_1)

            nodes_1, edges_1 = self._forward_intra_domain_graph(nodes_1)
            nodes_2, edges_2 = self._forward_intra_domain_graph(nodes_2)

            self.update_seed(nodes_1, labels_1, nodes_1, labels_1)

            nodes_1, nodes_2 = self._forward_cross_domain_graph(nodes_1, nodes_2)
            node_loss = self._forward_node_loss(
                torch.cat([nodes_1, nodes_2],dim=0),
                torch.cat([labels_1, labels_2],dim=0)
            )
        middle_head_loss.update({'node_loss': self.weight_nodes * node_loss})

        # STEP8: Generate Semantic-aware Node Affinity and Structure-aware Matching loss
        if self.matching_cfg != 'none':
            matching_loss_affinity, affinity = self._forward_aff(nodes_1, nodes_2, labels_1, labels_2)
            middle_head_loss.update({'mat_loss_aff': self.weight_matching * matching_loss_affinity })

            if self.with_quadratic_matching:
                matching_loss_quadratic = self._forward_qu(edges_1.detach(), edges_2.detach(), affinity)
                middle_head_loss.update({'mat_loss_qu':  matching_loss_quadratic})

        return features, (nodes_1, nodes_2), middle_head_loss
    
    def _forward_preprocessing_source(self, sr_nodes, sr_nodes_label):
        labels_exist = sr_nodes_label.unique()

        nodes_1_cls_first = []
        nodes_2_cls_first = []
        labels_1_cls_first = []
        labels_2_cls_first = []

        for c in labels_exist:
            sr_nodes_c = sr_nodes[sr_nodes_label == c]
            nodes_1_cls_first.append(torch.cat([sr_nodes_c[::2, :]]))
            nodes_2_cls_first.append(torch.cat([sr_nodes_c[1::2, :]]))

            labels_side1 = sr_nodes_c.new_ones(len(nodes_1_cls_first[-1])) * c
            labels_side2 = sr_nodes_c.new_ones(len(nodes_2_cls_first[-1])) * c

            labels_1_cls_first.append(labels_side1)
            labels_2_cls_first.append(labels_side2)

        nodes_1 = torch.cat(nodes_1_cls_first, dim=0)
        nodes_2 = torch.cat(nodes_2_cls_first, dim=0)

        labels_1 = torch.cat(labels_1_cls_first, dim=0)
        labels_2 = torch.cat(labels_2_cls_first, dim=0)

        return (nodes_1, nodes_2), (labels_1, labels_2)
    
    def _forward_preprocessing_source_target(self, nodes, labels, weights):

        '''
        nodes: sampled raw source/target nodes
        labels: the ground-truth/pseudo-label of sampled source/target nodes
        weights: the confidence of sampled source/target nodes ([0.0,1.0] scores for target nodes and 1.0 for source nodes )

        We permute graph nodes according to the class from 1 to K and complete the missing class.

        '''

        sr_nodes, tg_nodes = nodes
        sr_nodes_label, tg_nodes_label = labels
        sr_loss_weight, tg_loss_weight = weights

        labels_exist = torch.cat([sr_nodes_label, tg_nodes_label]).unique()

        sr_nodes_category_first = []
        tg_nodes_category_first = []

        sr_labels_category_first = []
        tg_labels_category_first = []

        sr_weight_category_first = []
        tg_weight_category_first = []

        for c in labels_exist:

            sr_indx = sr_nodes_label == c
            tg_indx = tg_nodes_label == c

            sr_nodes_c = sr_nodes[sr_indx]
            tg_nodes_c = tg_nodes[tg_indx]

            sr_weight_c = sr_loss_weight[sr_indx]
            tg_weight_c = tg_loss_weight[tg_indx]

            if sr_indx.any() and tg_indx.any(): # If the category appear in both domains, we directly collect them!

                sr_nodes_category_first.append(sr_nodes_c)
                tg_nodes_category_first.append(tg_nodes_c)

                labels_sr = sr_nodes_c.new_ones(len(sr_nodes_c)) * c
                labels_tg = tg_nodes_c.new_ones(len(tg_nodes_c)) * c

                sr_labels_category_first.append(labels_sr)
                tg_labels_category_first.append(labels_tg)

                sr_weight_category_first.append(sr_weight_c)
                tg_weight_category_first.append(tg_weight_c)

            elif tg_indx.any():  # If there're no source nodes in this category, we complete it with hallucination nodes!

                num_nodes = len(tg_nodes_c)
                sr_nodes_c = self.sr_seed[int(c.item())].unsqueeze(0).expand(num_nodes, 256)

                if self.with_semantic_completion:
                    sr_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).to(self.device) + sr_nodes_c if len(tg_nodes_c)<5 \
                        else  torch.normal(mean=sr_nodes_c, std=tg_nodes_c.std(0).unsqueeze(0).expand(sr_nodes_c.size())).to(self.device)
                else:
                    sr_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).to(self.device)

                sr_nodes_c = self.seed_project_left(sr_nodes_c)
                sr_nodes_category_first.append(sr_nodes_c)
                tg_nodes_category_first.append(tg_nodes_c)
                sr_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).to(self.device) * c)
                tg_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).to(self.device) * c)
                sr_weight_category_first.append(torch.ones(num_nodes, dtype=torch.long).to(self.device))
                tg_weight_category_first.append(tg_weight_c)

            elif sr_indx.any():  # If there're no target nodes in this category, we complete it with hallucination nodes!

                num_nodes = len(sr_nodes_c)

                sr_nodes_category_first.append(sr_nodes_c)
                tg_nodes_c = self.tg_seed[int(c.item())].unsqueeze(0).expand(num_nodes, 256)

                if self.with_semantic_completion:
                    tg_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).to(self.device) + tg_nodes_c if len(sr_nodes_c)<5 \
                        else torch.normal(mean=tg_nodes_c,
                                              std=sr_nodes_c.std(0).unsqueeze(0).expand(sr_nodes_c.size())).to(self.device)
                else:
                    tg_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).to(self.device)

                tg_nodes_c = self.seed_project_left(tg_nodes_c)
                tg_nodes_category_first.append(tg_nodes_c)

                sr_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).to(self.device) * c)
                tg_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).to(self.device) * c)

                sr_weight_category_first.append(sr_weight_c)
                tg_weight_category_first.append(torch.ones(num_nodes, dtype=torch.long))

        nodes_sr = torch.cat(sr_nodes_category_first, dim=0)
        nodes_tg = torch.cat(tg_nodes_category_first, dim=0)

        weight_sr = torch.cat(sr_weight_category_first, dim=0)
        weight_tg = torch.cat(tg_weight_category_first, dim=0)

        label_sr = torch.cat(sr_labels_category_first, dim=0)
        label_tg = torch.cat(tg_labels_category_first, dim=0)

        return (nodes_sr, nodes_tg), (label_sr, label_tg), (weight_sr, weight_tg)

    def _forward_intra_domain_graph(self, nodes):
        nodes, edges = self.intra_domain_graph(nodes, nodes, nodes)
        return nodes, edges

    def _forward_cross_domain_graph(self, nodes_1, nodes_2):

        if self.with_global_graph:
            n_1 = len(nodes_1)
            n_2 = len(nodes_2)
            global_nodes = torch.cat([nodes_1, nodes_2], dim=0)
            global_nodes = self.cross_domain_graph(global_nodes, global_nodes, global_nodes)[0]

            nodes1_enahnced = global_nodes[:n_1]
            nodes2_enahnced = global_nodes[n_1:]
        else:
            nodes2_enahnced = self.cross_domain_graph(nodes_1, nodes_1, nodes_2)[0]
            nodes1_enahnced = self.cross_domain_graph(nodes_2, nodes_2, nodes_1)[0]

        return nodes1_enahnced, nodes2_enahnced

    def _forward_node_loss(self, nodes, labels, weights=None):

        labels= labels.long()
        assert len(nodes) == len(labels)

        if weights is None:  # Source domain
            if self.with_cond_cls:
                tg_embeds = self.node_cls_middle(self.tg_seed)
                logits = self.dynamic_fc(nodes, tg_embeds)
            else:
                logits = self.node_cls_middle(nodes)

            node_loss = F.cross_entropy(logits, labels,
                                        reduction='mean')
        else:  # Target domain
            if self.with_cond_cls:
                sr_embeds = self.node_cls_middle(self.sr_seed)
                logits = self.dynamic_fc(nodes, sr_embeds)
            else:
                logits = self.node_cls_middle(nodes)

            node_loss = F.cross_entropy(logits, labels.long(),
                                        reduction='none')
            node_loss = (node_loss * weights).float().mean() if self.with_score_weight else node_loss.float().mean()

        return node_loss

    def update_seed(self, sr_nodes, sr_labels, tg_nodes=None, tg_labels=None):

        k = 20 # conduct clustering when we have enough graph nodes
        for cls in sr_labels.unique().long():
            bs = sr_nodes[sr_labels == cls].detach()
            if len(bs) > k and self.with_cluster_update:
                #TODO Use Pytorch-based GPU version
                sp = cluster.SpectralClustering(2, affinity='nearest_neighbors', n_jobs=-1,
                                                assign_labels='kmeans', random_state=1234, n_neighbors=len(bs) // 2)

                seed_cls = self.sr_seed[cls]
                indx = sp.fit_predict(torch.cat([seed_cls[None, :], bs]).cpu().numpy())
                indx = (indx == indx[0])[1:]
                bs = bs[indx].mean(0)
            else:
                bs = bs.mean(0)

            momentum = torch.nn.functional.cosine_similarity(bs.unsqueeze(0), self.sr_seed[cls].unsqueeze(0))
            self.sr_seed[cls] = self.sr_seed[cls] * momentum + bs * (1.0 - momentum)

        if tg_nodes is not None:
            for cls in tg_labels.unique().long():
                bs = tg_nodes[tg_labels == cls].detach()
                
                if len(bs) > k and self.with_cluster_update:
                    sp = cluster.SpectralClustering(2, affinity='nearest_neighbors', n_jobs=-1,
                                                    assign_labels='kmeans', random_state=1234, n_neighbors=len(bs) // 2)
                    seed_cls = self.tg_seed[cls]
                    indx = sp.fit_predict(torch.cat([seed_cls[None, :], bs]).cpu().numpy())
                    indx = (indx == indx[0])[1:]
                    bs = bs[indx].mean(0)
                else:
                    bs = bs.mean(0)
                
                momentum = torch.nn.functional.cosine_similarity(bs.unsqueeze(0), self.tg_seed[cls].unsqueeze(0))
                self.tg_seed[cls] = self.tg_seed[cls] * momentum + bs * (1.0 - momentum)

    def _forward_aff(self, nodes_1, nodes_2, labels_side1, labels_side2):
        if self.matching_cfg == 'o2o':
            M = self.node_affinity(nodes_1, nodes_2)
            matching_target = torch.mm(self.one_hot(labels_side1), self.one_hot(labels_side2).t()).to(M.device)

            M = self.InstNorm_layer(M[None, None, :, :])
            M = self.sinkhorn_rpm(M[:, 0, :, :], n_iters=20).squeeze().exp()

            TP_mask = (matching_target == 1).float()
            indx = (M * TP_mask).max(-1)[1]
            TP_samples = M[range(M.size(0)), indx].view(-1, 1)
            TP_target = torch.full(TP_samples.shape, 1, dtype=torch.float, device=TP_samples.device).float()

            FP_samples = M[matching_target == 0].view(-1, 1)
            FP_target = torch.full(FP_samples.shape, 0, dtype=torch.float, device=FP_samples.device).float()

            # TP_loss = self.matching_loss(TP_sample, TP_target.float())
            #TODO Find a better reduction strategy
            TP_loss = self.matching_loss(TP_samples, TP_target.float())/ len(TP_samples)
            FP_loss = self.matching_loss(FP_samples, FP_target.float())/ torch.sum(FP_samples).detach()
            # print('FP: ', FP_loss, 'TP: ', TP_loss)
            matching_loss = TP_loss + FP_loss

        elif self.matching_cfg == 'm2m': # Refer to the Appendix
            M = self.node_affinity(nodes_1, nodes_2)
            matching_target = torch.mm(self.one_hot(labels_side1), self.one_hot(labels_side2).t())
            matching_loss = self.matching_loss(M.sigmoid(), matching_target.float()).mean()
        else:
            M = None
            matching_loss = 0
        return matching_loss, M

    def _forward_inference(self, images, features):
        return features

    def _forward_qu(self, edge_1, edge_2, affinity):
        R =  torch.mm(edge_1, affinity) - torch.mm(affinity, edge_2)
        loss = self.quadratic_loss(R, R.new_zeros(R.size()))
        return loss

    def compute_locations(self, features):
        locations = []
        fpn_strides = [8, 16, 32, 64, 128]
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    def sinkhorn_rpm(self, log_alpha, n_iters=5, slack=True, eps=-1):
        ''' Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

        Args:
            log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
            n_iters (int): Number of normalization iterations
            slack (bool): Whether to include slack row and column
            eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

        Returns:
            log(perm_matrix): Doubly stochastic matrix (B, J, K)

        Modified from original source taken from:
            Learning Latent Permutations with Gumbel-Sinkhorn Networks
            https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
        '''
        prev_alpha = None
        if slack:
            zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
            log_alpha_padded = zero_pad(log_alpha[:, None, :, :])
            log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

            for i in range(n_iters):
                # Row normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                    dim=1)
                # Column normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                    dim=2)
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()
            log_alpha = log_alpha_padded[:, :-1, :-1]
        else:
            for i in range(n_iters):
                # Row normalization (i.e. each row sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))
                # Column normalization (i.e. each column sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha).clone()              
        return log_alpha

    def dynamic_fc(self, features, kernel_par):
        weight = kernel_par
        return torch.nn.functional.linear(features, weight, bias=None)

    def dynamic_conv(self, features, kernel_par):
        weight = kernel_par.view(self.num_classes, -1, 1, 1)
        return torch.nn.functional.conv2d(features, weight)

    def one_hot(self, x):
        return torch.eye(self.num_classes)[x.long(), :].to(self.device)

    def masks_to_boxes(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute the bounding boxes around the provided masks.

        Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
        ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

        Args:
            masks (Tensor[N, H, W]): masks to transform where N is the number of masks
                and (H, W) are the spatial dimensions.

        Returns:
            Tensor[N, 4]: bounding boxes
        """
        N, H, W = masks.shape

        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        n = masks.shape[0]

        bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

        for index, mask in enumerate(masks):
            y, x = torch.where(mask != 0)

            if x.numel() == 0:
                bounding_boxes[index, 0] = 0
                bounding_boxes[index, 1] = 0
                bounding_boxes[index, 2] = W
                bounding_boxes[index, 3] = H

            else:
                bounding_boxes[index, 0] = torch.min(x)
                bounding_boxes[index, 1] = torch.min(y)
                bounding_boxes[index, 2] = torch.max(x)
                bounding_boxes[index, 3] = torch.max(y)

        return bounding_boxes

    def find_bbox(self, masks):
        bboxs = []
        for mask in masks:
            bboxs.append(self.masks_to_boxes(mask))
        return bboxs

    @torch.no_grad()
    def draw_bounding_boxes(
        self,
        image,
        boxes,
        labels=None,
        colors=None,
        fill=False,
        width=1,
        font=None,
        font_size=None,
    ) -> torch.Tensor:

        """
        Draws bounding boxes on given image.
        The values of the input image should be uint8 between 0 and 255.
        If fill is True, Resulting Tensor should be saved as PNG image.

        Args:
            image (Tensor): Tensor of shape (C x H x W) and dtype uint8.
            boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
                the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
                `0 <= ymin < ymax < H`.
            labels (List[str]): List containing the labels of bounding boxes.
            colors (color or list of colors, optional): List containing the colors
                of the boxes or single color for all boxes. The color can be represented as
                PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
                By default, random colors are generated for boxes.
            fill (bool): If `True` fills the bounding box with specified color.
            width (int): Width of bounding box.
            font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
                also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
                `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
            font_size (int): The requested font size in points.

        Returns:
            img (Tensor[C, H, W]): Image Tensor of dtype uint8 with bounding boxes plotted.
        """

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Tensor expected, got {type(image)}")
        elif image.dtype != torch.uint8:
            raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
        elif image.dim() != 3:
            raise ValueError("Pass individual images, not batches")
        elif image.size(0) not in {1, 3}:
            raise ValueError("Only grayscale and RGB images are supported")

        num_boxes = boxes.shape[0]

        if num_boxes == 0:
            warnings.warn("boxes doesn't contain any box. No box was drawn")
            return image

        if labels is None:
            labels: Union[List[str], List[None]] = [None] * num_boxes  # type: ignore[no-redef]
        elif len(labels) != num_boxes:
            raise ValueError(
                f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
            )

        if colors is None:
            colors = _generate_color_palette(num_boxes)
        elif isinstance(colors, list):
            if len(colors) < num_boxes:
                raise ValueError(f"Number of colors ({len(colors)}) is less than number of boxes ({num_boxes}). ")
        else:  # colors specifies a single color for all boxes
            colors = [colors] * num_boxes

        colors = [(ImageColor.getrgb(color) if isinstance(color, str) else color) for color in colors]

        if font is None:
            if font_size is not None:
                warnings.warn("Argument 'font_size' will be ignored since 'font' is not set.")
            txt_font = ImageFont.load_default()
        else:
            txt_font = ImageFont.truetype(font=font, size=font_size or 10)

        # Handle Grayscale images
        if image.size(0) == 1:
            image = torch.tile(image, (3, 1, 1))

        ndarr = image.permute(1, 2, 0).cpu().numpy()
        img_to_draw = Image.fromarray(ndarr)
        img_boxes = boxes.to(torch.int64).tolist()

        if fill:
            draw = ImageDraw.Draw(img_to_draw, "RGBA")
        else:
            draw = ImageDraw.Draw(img_to_draw)

        for bbox, color, label in zip(img_boxes, colors, labels):  # type: ignore[arg-type]
            if fill:
                fill_color = color + (100,)
                draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
            else:
                draw.rectangle(bbox, width=width, outline=color)

            if label is not None:
                margin = width + 1
                draw.text((bbox[0] + margin, bbox[1] + margin), label, fill=color, font=txt_font)

        return torch.from_numpy(np.array(img_to_draw)).to(dtype=torch.uint8)

    def draw_bbox(self, masks):

        mask = masks[0]
        bbox = self.masks_to_boxes(mask)
        drawn_boxes = self.draw_bounding_boxes((mask * 255).to(dtype=torch.uint8), bbox, colors="red").cpu().numpy()
        drawn_boxes = Image.fromarray(drawn_boxes)
        drawn_boxes.save('./graph_matching.png')
        

class PrototypeComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, num_class):
        self.num_class = num_class
        self.class_threshold = (0.5, 1.0)
        self.num_nodes_per_class = 100
        self.num_nodes_per_lvl = 100
        self.bg_ratio = 8
        self.sample_bg_nodes = True
    
    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            # reg_targets_level_first.append(
            #     torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            # )

        return labels_level_first
    
    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            #assert targets_per_im.mode == "xyxy"
            #bboxes = targets_per_im.bbox
            bboxes = targets_per_im
            labels_per_im = torch.arange(self.num_class)
            #area = targets_per_im.area()
            area = []
            for bbox in bboxes:
                area.append((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))
            area = torch.Tensor(area)

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]

            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0
            
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)


    def __call__(self, locations, features, targets):

        if locations: # Sampling in the source domain
            N, C, _, _ = features[0].size() # N, C, H, W
            labels = self.prepare_targets(locations, targets)

            pos_points = []
            pos_labels = []
            neg_points = []
            for l in range(len(labels)):
                pos_indx =  labels[l].reshape(-1) > 0
                neg_indx =  labels[l].reshape(-1) == 0

                # Sparse sampling to save GPU memory
                pos_nodes_all = features[l].permute(0, 2, 3, 1).reshape(-1, C)[pos_indx]
                pos_labels_all = labels[l][pos_indx]
                step = len(pos_labels_all) //self.num_nodes_per_class
                if step>1:
                    pos_points.append(pos_nodes_all[::step])
                    pos_labels.append(pos_labels_all[::step])
                else:
                    pos_points.append(pos_nodes_all)
                    pos_labels.append(pos_labels_all)
                num_pos = len(pos_points[-1])

                # Sampling Background Nodes
                if self.sample_bg_nodes:
                    neg_points_temp = features[l].permute(0, 2, 3, 1).reshape(-1, features[l].shape[1])[neg_indx]
                    if len(labels[l][pos_indx]) > len(labels[l][neg_indx]):
                        neg_points.append(features[l].permute(0, 2, 3, 1).reshape(-1, C)[neg_indx])
                    else:
                        # neg_indx = list(np.floor(np.linspace(0,len(labels[l][neg_indx])-2, (len(labels[l][pos_indx])))/8).astype(int))
                        neg_indx = list(np.floor(np.linspace(0,len(labels[l][neg_indx])-2, num_pos//self.bg_ratio)))
                        neg_points.append(neg_points_temp[neg_indx])

            pos_points = torch.cat(pos_points,dim=0)
            pos_labels = torch.cat(pos_labels,dim=0)

            if self.sample_bg_nodes:
                neg_points = torch.cat(neg_points, dim=0)
                neg_labels = pos_labels.new_zeros((neg_points.size(0)))
                pos_points = torch.cat([neg_points, pos_points] ,dim=0)
                pos_labels = torch.cat([neg_labels, pos_labels] )

            return pos_points, pos_labels, pos_labels.new_ones(pos_labels.shape).long()


        else: # Sampling in the target domain
            act_maps_lvl_first = targets
            N, C, _, _ = features[0].size()
            N, Cls, _, _ = targets[0].size()
            neg_points =[]
            pos_plabels = []
            pos_points = []
            pos_weight = []
            for l, feature in enumerate(features):
                act_maps = act_maps_lvl_first[l].permute(0, 2, 3, 1).reshape(-1, self.num_class)
                conf_pos_indx = (act_maps > self.class_threshold[0]).sum(dim=-1).bool()
                neg_indx = (act_maps < 0.05).sum(dim=-1).bool()
                # Balanced sampling BG pixels
                if conf_pos_indx.any():
                    act_maps = act_maps_lvl_first[l].permute(0, 2, 3, 1).reshape(-1, self.num_class)
                    if conf_pos_indx.sum() > self.num_nodes_per_lvl :
                        raw_features = features[l].permute(0, 2, 3, 1).reshape(-1, C)[conf_pos_indx]
                        twice_indx = torch.randperm(raw_features.size(0))[:100]
                        pos_points.append(raw_features[twice_indx])
                        scores, indx = act_maps[conf_pos_indx, :].max(-1)
                        scores = scores[twice_indx]
                        indx = indx[twice_indx]
                    else:
                        pos_points.append(features[l].permute(0, 2, 3, 1).reshape(-1, C)[conf_pos_indx])
                        scores, indx = act_maps[conf_pos_indx,:].max(-1)

                    # pos_plabels.append(act_maps[conf_pos_indx,:].argmax(dim=-1) + 1)
                    pos_plabels.append(indx + 1)
                    pos_weight.append(scores.detach())
                    # neg_indx = ~conf_pos_indx
                    neg_points_temp = features[l].permute(0, 2, 3, 1).reshape(-1, C)[neg_indx]
                    num_pos = len(scores)
                    neg_indx_new = list(np.floor(np.linspace(0, (neg_indx.sum()- 2).item(), (num_pos//self.bg_ratio))).astype(int))
                    neg_points.append(neg_points_temp[neg_indx_new])
            # print(end-start)
            if len(pos_points)>0:
                pos_points = torch.cat(pos_points,dim=0)
                pos_plabels = torch.cat(pos_plabels,dim=0)
                neg_points = torch.cat(neg_points, dim=0)
                neg_plabels = pos_plabels.new_zeros((neg_points.size(0)))

                pos_weight = torch.cat(pos_weight, dim=0)
                neg_weight = pos_weight.new_ones(neg_points.size(0)) * 0.5
                points = torch.cat([neg_points, pos_points], dim=0)
                plabels = torch.cat([neg_plabels, pos_plabels])

                loss_weight = torch.cat([neg_weight, pos_weight])
                return points, plabels, loss_weight.long()
            else:
                return None, None, None
