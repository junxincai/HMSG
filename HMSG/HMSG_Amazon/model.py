import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.ops import edge_softmax
import math


class GATConv(nn.Module):
    def __init__(self,
                 in_size,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self.in_size = in_size
        self.residual = residual
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, in_size)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, in_size)))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation

        nn.init.xavier_normal_(self.attn_l, gain=1.414)
        nn.init.xavier_normal_(self.attn_r, gain=1.414)

    def forward(self, graph, feat):
        with graph.local_scope():
            h_src = h_dst = self.feat_drop(feat).view(-1, self._num_heads, self.in_size)

            el = (h_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (h_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': h_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.residual:
                rst = rst + h_dst
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst

class HetGCNLayer(nn.Module):
    def __init__(self, 
                in_size,
                aggregator_type='attention',
                num_heads=8,
                feat_drop=0.,
                attn_drop=0.,
                negative_slope=0.2,
                residual=False,
                activation=None):
        super(HetGCNLayer, self).__init__()

        self.num_heads = num_heads
        self.in_size = in_size
        # self.fc_mean = nn.Linear(in_size * num_heads, in_size * num_heads)
        # nn.init.xavier_normal_(self.fc_mean.weight, gain=1.414)
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(in_size * num_heads, in_size * num_heads)
            nn.init.xavier_normal_(self.fc_pool.weight, gain=1.414)
        self.aggre_type = aggregator_type

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, self.num_heads, in_size)))
        # self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, self.num_heads, in_size)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual
        self.activation = activation

        nn.init.xavier_normal_(self.attn_l, gain=1.414)
        # nn.init.xavier_normal_(self.attn_r, gain=1.414)

    def forward(self, g, feat):
        with g.local_scope():
            if self.aggre_type == 'attention':
                h_src = self.feat_drop(feat[0]).view(-1, self.num_heads, self.in_size)
                h_dst = self.feat_drop(feat[1]).view(-1, self.num_heads, self.in_size)
                el = (h_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                # er = (h_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                g.srcdata.update({'ft': h_src, 'el': el})
                # g.srcdata.update({'ft': h_src, 'er': er})
                g.apply_edges(fn.copy_u('el', 'e'))
                # g.apply_edges(fn.u_add_v('el', 'er', 'e'))
                e = self.leaky_relu(g.edata.pop('e'))

                g.edata['a'] = self.attn_drop(edge_softmax(g, e))
                g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
                rst = g.dstdata['ft'].flatten(1)
                if self.residual:
                    rst = rst + h_dst.flatten(1)
                if self.activation:
                    rst = self.activation(rst)

            elif self.aggre_type == 'mean':
                h_src = self.feat_drop(feat[0]).view(-1, self.in_size*self.num_heads)
                h_dst = self.feat_drop(feat[1]).view(-1, self.in_size * self.num_heads)
                g.srcdata['ft'] = h_src
                g.update_all(fn.copy_u('ft', 'm'), fn.mean('m', 'ft'))
                rst = g.dstdata['ft'] # + h_dst


            elif self.aggre_type == 'pool':
                h_src = self.feat_drop(feat[0]).view(-1, self.in_size*self.num_heads)
                h_dst = self.feat_drop(feat[1]).view(-1, self.in_size * self.num_heads)
                g.srcdata['ft'] = F.relu(self.fc_pool(h_src))
                g.update_all(fn.copy_u('ft', 'm'), fn.max('m', 'ft'))
                rst = g.dstdata['ft'] #+ h_dst
            return rst

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        # self.fc = nn.Linear(in_size, 64, bias=False)
        # nn.init.xavier_normal_(self.fc.weight, gain=1.414)
    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        # return self.fc((beta * z).sum(1))
        return (beta * z).sum(1)

class HMSGLayer(nn.Module):
    def __init__(self, meta_paths, in_size, aggre_type, layer_num_heads, dropout):
        super(HMSGLayer, self).__init__()
        self.nunm_heads = layer_num_heads
        self.semantic_attention_i = SemanticAttention(in_size=in_size * layer_num_heads)
        self.semantic_attention_u = SemanticAttention(in_size=in_size * layer_num_heads)

        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            if meta_paths[i][0][0] == meta_paths[i][-1][-1]:
                self.gat_layers.append(GATConv(in_size, layer_num_heads,
                    dropout, dropout, activation=F.elu, residual=False))
            else:
                self.gat_layers.append(HetGCNLayer(in_size, aggre_type, self.nunm_heads,
                                                   dropout, dropout, activation=F.elu, residual=False))
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = {'user':[], 'item':[]}
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                if len(meta_path) > 1:
                    self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(g, meta_path)  # return a homogeneous or unidirectional bipartite graphs
                elif len(meta_path) == 1:
                    if meta_path in {('ui',)}:
                        print('******************ui**********************')
                        self._cached_coalesced_graph[meta_path] = dgl.edge_type_subgraph(g, [('user', 'ui', 'item')])
                    elif meta_path in {('iu',)}:
                        print('******************iu**********************')
                        self._cached_coalesced_graph[meta_path] = dgl.edge_type_subgraph(g, [('item', 'iu', 'user')])

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            if new_g.is_homogeneous:
                ntype = new_g.ntypes[0]
                if ntype == 'user':
                    h_ = h['user']
                elif ntype == 'item':
                    h_ = h['item']
                semantic_embeddings[ntype].append(self.gat_layers[i](new_g, h_).flatten(1))

            else:  
                if meta_path in {('ui',), }:
                    semantic_embeddings['item'].append(self.gat_layers[i](new_g, (h['user'], h['item'])))
                elif meta_path in {('iu',)}:
                    semantic_embeddings['user'].append(self.gat_layers[i](new_g, (h['item'], h['user'])))

        embedings = {}
        for ntype in semantic_embeddings.keys():
            if ntype=='user':
                semantic_embeddings[ntype] = torch.stack(semantic_embeddings[ntype], dim=1) 
                embedings[ntype] = self.semantic_attention_u(semantic_embeddings[ntype])  
            elif ntype=='item' and semantic_embeddings[ntype]:
                semantic_embeddings[ntype] = torch.stack(semantic_embeddings[ntype], dim=1) 
                embedings[ntype] = self.semantic_attention_i(semantic_embeddings[ntype])
        return embedings


class HMSG(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, aggre_type, num_heads, dropout):
        super(HMSG, self).__init__()
        
        self.fc_u = nn.Linear(in_size['user'], hidden_size*num_heads)
        self.fc_i = nn.Linear(in_size['item'], hidden_size*num_heads)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layers1 = HMSGLayer(meta_paths, hidden_size, aggre_type, num_heads, dropout)
        # self.layers2 = HMSGLayer(meta_paths, hidden_size, aggre_type, num_heads, dropout)
        # self.layers3 = HMSGLayer(meta_paths, hidden_size, aggre_type, num_heads, dropout)
        # self.fc = nn.Linear(hidden_size*num_heads, hidden_size)

        nn.init.xavier_normal_(self.fc_u.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_i.weight, gain=1.414)
        # nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, g, inputs):
        h_trans = {}
        h_trans['user'] = self.fc_u(inputs['user']).view(-1, self.num_heads, self.hidden_size)
        h_trans['item'] = self.fc_i(inputs['item']).view(-1, self.num_heads, self.hidden_size)
        h_trans = self.layers1(g, h_trans)
        # h_trans = self.layers2(g, h_trans)
        # h_trans = self.layers3(g, h_trans)
        # for k, v in h_trans.items():
        #     h_trans[k] = self.fc(v)

        # return self.fc(h_trans)
        return h_trans