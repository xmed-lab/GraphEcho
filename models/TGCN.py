import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from models.vig import GraphConv2d, DenseDilatedKnnGraph
from models.transformer import MultiHeadAttention
from models.gradient_reversal import GradientReversal

def calculate_laplacian_with_self_loop(matrixs):
    normalized_laplacians = []
    for matrix in matrixs:
        matrix = matrix + torch.eye(matrix.size(0), device=matrix.device)
        row_sum = matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = (
            matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        )
        normalized_laplacians.append(normalized_laplacian.unsqueeze(0))
    return torch.cat(normalized_laplacians, dim=0)

def calculate_laplacian_without_self_loop(graph, normalize=None):
    """
    return the laplacian of the graph.
    :param graph: the graph structure without self loop, [N, N].
    :param normalize: whether to used the normalized laplacian.
    :return: graph laplacian.
    """
    if normalize:
        D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
        L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
    else:
        D = torch.diag(torch.sum(graph, dim=-1))
        L = D - graph
    return L


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='mr', act='gelu',
                 norm=None, bias=True, stochastic=False, epsilon=0.2):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        #self.inner_attention = MultiHeadAttention(256, 1)

        self.MLP = nn.Sequential(
                nn.Conv2d(in_channels * 4, out_channels, 1, stride=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Conv2d(out_channels, out_channels, 1, stride=1, bias=True),
                ) # output layer

        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, input, rs, y, learnable_pos, relative_pos=None):
        x_list = []
        for i, r in enumerate(rs):
            if r > 1:
                x_list.append(F.avg_pool2d(input[i], r, r))
            else:
                x_list.append(input[i])

        x = torch.cat(x_list, dim=1)
        x = self.MLP(x)
        x = x + learnable_pos
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()
        
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H*W).contiguous(), H, W


class TGCNGraphConvolution(nn.Module):
    def __init__(self, in_feature_dim: int, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._in_feature_num = in_feature_dim
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias

        '''
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(nn.Parameter(torch.FloatTensor(self._in_feature_num, self._num_gru_units)))
        )
        '''
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units+self._in_feature_num, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes, features = inputs.shape
        laplacian = calculate_laplacian_with_self_loop(inputs)
 
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )

        concatenation = torch.cat((inputs, hidden_state), dim=2)

        a_times_concat = torch.einsum('bnc,bck->bnk', laplacian, concatenation)

        a_times_concat = a_times_concat.permute(1,2,0)

        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)

        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, (self._num_gru_units + features))
        )

        outputs = a_times_concat @ self.weights + self.biases

        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))

        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGCNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.graph_conv1 = TGCNGraphConvolution(
            self._input_dim, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self._input_dim, self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):

        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))

        r, u = torch.chunk(concatenation, chunks=2, dim=1)

        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))

        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, clip_shape: tuple, soucre_class: int, target_class: int, 
                       cluster_method=None, transport_method='node_discriminate'):
        super(TGCN, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim

        self.grapher = DyGraphConv2d(input_dim, hidden_dim)
        self.graph_attention = MultiHeadAttention(256, 1, dropout=0.1, version='v2')

        self.clip_l, self.clip_h, self.clip_w = clip_shape

        self.cluster_method = cluster_method
        self.transport_method = transport_method
        self.pos_embed = nn.Parameter(torch.zeros(self.clip_l, 1, input_dim ,self.clip_h, self.clip_w))

        self.prediction = nn.Sequential(
                            nn.Conv2d(self._hidden_dim , self._hidden_dim, 3, stride=2, bias=True),
                            nn.BatchNorm2d(self._hidden_dim),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.AdaptiveAvgPool2d(1),
                            )

        if self.cluster_method == 'momentum_queue':
            self.m = 0.99
            self.K = 150
            self.register_buffer("queue_source", torch.randn(self._hidden_dim, self.K))
            self.register_buffer("queue_target", torch.randn(self._hidden_dim, self.K))
            self.queue_source = nn.functional.normalize(self.queue_source, dim=0)
            self.queue_target = nn.functional.normalize(self.queue_target, dim=0)
       
        elif self.cluster_method == 'linear_clustering':
            self.classifer_source  = nn.Linear(self._hidden_dim, soucre_class)
            self.classifer_target  = nn.Linear(self._hidden_dim, target_class)

        if self.transport_method == 'node_discriminate': 
            self.loss_bce = nn.BCEWithLogitsLoss()
            self.grad_reverse = GradientReversal(0.02)
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
            for i in self.node_dis_2:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)

    def forward(self, input_features, input_feature_nodes, loss_trans, loss_cluster, update_index, r=1.0):
        losses = dict()
        x_f1, x_f2, x_f3, x_f4 = input_features
        source_nodes, target_nodes = input_feature_nodes
        batch_size, seq_len, _, _, _ = x_f1.shape
        
        hidden_state = torch.zeros(batch_size, self._input_dim, self.clip_h*self.clip_w).type_as(x_f1)
        output = None
        for i in range(seq_len):
            input = [x_f1[:, i, ...], x_f2[:, i, ...], x_f3[:, i, ...], x_f4[:, i, ...]]
            current_graph, H, W = self.grapher(input, r, hidden_state, self.pos_embed[i, ...])
            hidden_state = current_graph
            #assert self._input_dim == num_nodes
        batch_size, features, num_nodes = current_graph.shape
        output_f = current_graph.reshape(batch_size, features, H, W)
        output_f = self.prediction(output_f).view(batch_size, -1)

        update_index_source, update_index_target = update_index

        if self.cluster_method == 'momentum_queue':
            q = nn.functional.normalize(output_f, dim=1)
            l_pos = torch.einsum('nc,ck->nk', [q, torch.cat([self.queue_source, self.queue_target], dim=-1).clone().detach()])

            self._dequeue_and_enqueue(q[:batch_size//2, ...], self.queue_source, update_index_source)
            self._dequeue_and_enqueue(q[batch_size//2:, ...], self.queue_target, update_index_target)

            loss_c = loss_cluster(l_pos, torch.cat([update_index_source, torch.add(update_index_target, 150)]))
            losses.update({'clustering_loss': loss_c})

        elif self.cluster_method == 'linear_clustering':
            loss_c = loss_cluster(self.classifer_source(output_f[:batch_size//2, ...]), update_index_source) + \
                     loss_cluster(self.classifer_target(output_f[batch_size//2:, ...]), update_index_target)
            losses.update({'clustering_loss': loss_c})

        output_g = current_graph.transpose(1,2)
        b_g, d_g, n_g = output_g.shape
        output_g = output_g.reshape(b_g*d_g, n_g)
        source_sample_nodes_num = len(source_nodes)
        target_sample_nodes_num = len(target_nodes)
        output_sample_nodes_num = len(output_g)

        nodes_ = torch.cat([output_g, source_nodes, target_nodes])
        nodes_ = self.graph_attention(nodes_, nodes_, nodes_)[0]
        nodes_g = nodes_[:output_sample_nodes_num].reshape(b_g, d_g, n_g)
        nodes_source, nodes_target = nodes_g[:b_g//2, ...], nodes_g[b_g//2:, ...]
        nodes_source = nodes_source.reshape(-1, n_g)
        nodes_target = nodes_target.reshape(-1, n_g)
        
        if self.transport_method == 'node_discriminate':
            nodes_rev = self.grad_reverse(torch.cat([nodes_source, nodes_target], dim=0))
            target_1 = torch.full([nodes_source.size(0), 1], 1.0, dtype=torch.float, device=nodes_g.device)
            target_2 = torch.full([nodes_target.size(0), 1], 0.0, dtype=torch.float, device=nodes_g.device)
            tg_rev = torch.cat([target_1, target_2], dim=0)
            nodes_rev = self.node_dis_2(nodes_rev)
            node_dis_loss = 0.1 * self.loss_bce(nodes_rev.view(-1), tg_rev.view(-1))
            losses.update({'node_dis_loss': node_dis_loss})
        
        elif self.transport_method == 'sinkhorn_distance':
            loss_t = loss_trans(nodes_g[:batch_size//2, ...], nodes_g[batch_size//2:, ...])[0]
            losses.update({'sinkhorn_loss': loss_t})

        return losses

    @torch.no_grad()
    def _momentum_update_key_encoder(self, encoder_q, encoder_k):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, features, queue, labels):
        # gather keys before updating queue
        #features = concat_all_gather(features)
        #labels   = concat_all_gather(labels)

        for idx, l_idx in enumerate(labels):
            queue[:, l_idx] = queue[:, l_idx] * self.m + features[idx, ...] * (1.0 - self.m)

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

if __name__ == "__main__":
    from sinkhorn_distance import SinkhornDistance
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=5)
    ce_loss = nn.CrossEntropyLoss()
    model = TGCN(input_dim=256, hidden_dim=256, soucre_class=100, target_class=100, clip_shape=(32,8,8))
    input = [torch.rand(4,32,256,64,64), torch.rand(4,32,256,32,32), torch.rand(4,32,256,16,16), torch.rand(4,32,256,8,8)]
    nodes = [torch.rand(233,256), torch.rand(234,256)]
    update_index_1 = torch.randint(low=0, high=100, size=(2,), requires_grad=False)
    update_index_2 = torch.randint(low=0, high=100, size=(2,), requires_grad=False)
    loss = model(input, nodes, sinkhorn, ce_loss, (update_index_1, update_index_2), r=[8,4,2,1])

    print(loss)
