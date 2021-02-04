import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.ops import edge_softmax

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
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(in_size*num_heads, in_size*num_heads)
            nn.init.xavier_normal_(self.fc_pool.weight, gain=1.414)
        self.aggre_type = aggregator_type

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, self.num_heads, in_size)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual
        self.activation = activation

        nn.init.xavier_normal_(self.attn_l, gain=1.414)

    def forward(self, g, feat):
        with g.local_scope():
            if self.aggre_type == 'attention':
                if isinstance(feat, tuple):
                    h_src = self.feat_drop(feat[0]).view(-1, self.num_heads, self.in_size)
                    h_dst = self.feat_drop(feat[1]).view(-1, self.num_heads, self.in_size)
                el = (h_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                g.srcdata.update({'ft': h_src, 'el': el})
                g.apply_edges(fn.copy_u('el', 'e'))
                e = self.leaky_relu(g.edata.pop('e'))
                g.edata['a'] = self.attn_drop(edge_softmax(g, e))
                g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
                rst = g.dstdata['ft'].flatten(1)
                if self.residual:
                    rst = rst + h_dst
                if self.activation:
                    rst = self.activation(rst)

            elif self.aggre_type == 'mean':
                h_src = self.feat_drop(feat[0]).view(-1, self.in_size*self.num_heads)
                g.srcdata['ft'] = h_src
                g.update_all(fn.copy_u('ft', 'm'), fn.mean('m', 'ft'))
                rst = g.dstdata['ft']

            elif self.aggre_type == 'pool':
                h_src = self.feat_drop(feat[0]).view(-1, self.in_size*self.num_heads)
                g.srcdata['ft'] = F.relu(self.fc_pool(h_src))
                g.update_all(fn.copy_u('ft', 'm'), fn.mean('m', 'ft'))
                rst = g.dstdata['ft']
            return rst

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)
        return (beta * z).sum(1)                       # (N, D * K)

class HMSGLayer(nn.Module):
    def __init__(self, meta_paths, in_size, aggre_type, layer_num_heads, dropout):
        super(HMSGLayer, self).__init__()
        self.nunm_heads = layer_num_heads
        self.semantic_attention_m = SemanticAttention(in_size=in_size * layer_num_heads)
        # self.semantic_attention_a = SemanticAttention(in_size=in_size * layer_num_heads)
        # self.semantic_attention_d = SemanticAttention(in_size=in_size * layer_num_heads)

        self.hsmg_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            if meta_paths[i][0][0] == meta_paths[i][-1][-1]:
                self.hsmg_layers.append(GATConv(in_size, layer_num_heads,
                    dropout, dropout, activation=F.elu, residual=False))
            else:
                self.hsmg_layers.append(HetGCNLayer(in_size, aggre_type, self.nunm_heads,
                   dropout, dropout, activation=F.elu, residual=False))

        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = {'movie':[], 'actor':[], 'director':[]}
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                if len(meta_path) > 1:
                    self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(g, meta_path)  # return a homogeneous or unidirectional bipartite graphs
                elif len(meta_path) == 1:
                    if meta_path in {('am',),}:
                        print('******************am**********************')
                        self._cached_coalesced_graph[meta_path] = dgl.edge_type_subgraph(g, [('actor', 'am', 'movie')])
                    elif meta_path in { ('dm',)}:
                        print('******************dm**********************')
                        self._cached_coalesced_graph[meta_path] = dgl.edge_type_subgraph(g, [('director', 'dm', 'movie')])
                
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            if new_g.is_homogeneous:
                ntype = new_g.ntypes[0]
                semantic_embeddings[ntype].append(self.hsmg_layers[i](new_g, h[ntype]).flatten(1))
            else:   
                if meta_path in {('am',),}:
                    h_ = (h['actor'], h['movie'])
                    semantic_embeddings['movie'].append(self.hsmg_layers[i](new_g, h_))
                elif meta_path in { ('dm',)}:
                    h_ = (h['director'], h['movie'])
                    semantic_embeddings['movie'].append(self.hsmg_layers[i](new_g, h_))

        embedings = {}
        for ntype in semantic_embeddings.keys():
            if ntype=='movie':
                semantic_embeddings[ntype] = torch.stack(semantic_embeddings[ntype], dim=1) 
                embedings[ntype] = self.semantic_attention_m(semantic_embeddings[ntype])
            # elif ntype=='actor' and semantic_embeddings[ntype]:
            #     semantic_embeddings[ntype] = torch.stack(semantic_embeddings[ntype], dim=1)
            #     embedings[ntype] = self.semantic_attention_a(semantic_embeddings[ntype])
            # elif ntype=='director' and semantic_embeddings[ntype]:
            #     semantic_embeddings[ntype] = torch.stack(semantic_embeddings[ntype], dim=1)
            #     embedings[ntype] = self.semantic_attention_d(semantic_embeddings[ntype])
        return embedings


class HMSG(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, aggre_type, num_heads, dropout):
        super(HMSG, self).__init__()
        
        self.fc_m = nn.Linear(in_size['movie'], hidden_size*num_heads, bias=True)
        self.fc_a = nn.Linear(in_size['actor'], hidden_size*num_heads, bias=True)
        self.fc_d = nn.Linear(in_size['director'], hidden_size*num_heads, bias=True)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layers = HMSGLayer(meta_paths, hidden_size, aggre_type, num_heads, dropout)
        self.predict = nn.Linear(hidden_size * num_heads, out_size)
        nn.init.xavier_normal_(self.fc_m.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_a.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_d.weight, gain=1.414)

    def forward(self, g, inputs):
        h_trans = {}
        h_trans['movie'] = self.fc_m(inputs['movie']).view(-1, self.num_heads, self.hidden_size)
        h_trans['actor'] = self.fc_a(inputs['actor']).view(-1, self.num_heads, self.hidden_size)
        h_trans['director'] = self.fc_d(inputs['director']).view(-1, self.num_heads, self.hidden_size)

        h_trans = self.layers(g, h_trans)
        return h_trans['movie'], self.predict(h_trans['movie'])