from layers import *
from metrics import *
import torch
import ot

class GCN_dense_mse_2s(torch.nn.Module):
    def __init__(self, input_dict, lookup_table_act_att, input_dim, FLAGS, **kwargs):
        super(GCN_dense_mse_2s, self).__init__()


        self.input_dim = input_dim
        self.output_dim = FLAGS.output_dim
        self.input_dict = input_dict
        self.lookup_table_act_att = lookup_table_act_att
        self.FLAGS = FLAGS

        self.layers_att1 = GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.FLAGS.hidden1,
                                            input_dict=self.input_dict,
                                            act=lambda x: torch.max(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            type='att')

        self.layers_att2 = GraphConvolution(input_dim=self.FLAGS.hidden1,
                                            output_dim=self.FLAGS.output_dim,
                                            input_dict=self.input_dict,
                                            act=lambda x: torch.max(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            type='att')

        self.layers_all1 = GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.FLAGS.hidden1,
                                            input_dict=self.input_dict,
                                            act=lambda x: torch.max(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            type='all')


        self.layers_all2 = GraphConvolution(input_dim=self.FLAGS.hidden1,
                                            output_dim=self.FLAGS.output_dim,
                                            input_dict=self.input_dict,
                                            act=lambda x: torch.max(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            type='all')

    def forward(self, input_dict):
        self.inputs_all = input_dict['features_all']

        x = torch.Tensor(input_dict['features_att']).cuda()
        x = x.transpose(-2,-1)
        x = x.transpose(-2,-3)

        x = torch.squeeze(x)
        x = torch.sum(x, dim=-1)

        S_ALL = input_dict['support_all']
        S_ATT = input_dict['support_att']

        att_feature = torch.Tensor(self.inputs_all[self.FLAGS.label_num:,:]).cuda()
        att_feature = torch.unsqueeze(att_feature,0)
        att_feature = att_feature.repeat([x.shape[0],1,1])

        x = torch.unsqueeze(x, 2)
        x = x.repeat([1,1,self.FLAGS.w2v_dim])
        att_feature = att_feature*x

        self.inputs_att = att_feature.view(-1,self.FLAGS.w2v_dim)
        hidden = self.layers_att1(self.inputs_att, S_ATT)
        gcn_out_att = self.layers_att2(hidden, S_ATT)

        mask = self.input_dict['train_mask'].astype(np.int32)
        cls_emb = torch.nn.functional.normalize(torch.Tensor(self.inputs_all), 2)[:mask.shape[0]].cuda()
        train_cls = []
        test_cls = []

        for i in range(mask.shape[0]):
            if mask[i]==1:
                train_cls.append(cls_emb[i])
            else:
                test_cls.append(cls_emb[i])

        train_cls=torch.stack(train_cls)
        test_cls=torch.stack(test_cls)

        M = ot.dist(train_cls.cpu().numpy(), test_cls.cpu().numpy())
        a, b = np.ones((M.shape[0],)) / M.shape[0], np.ones((M.shape[1],)) / M.shape[1]  # uniform distribution on samples

        M /= M.max()
        G0 = ot.emd(a, b, M)


        sim = 1-(torch.matmul(torch.Tensor(G0).cuda()*M.shape[0], test_cls)*train_cls).sum(1).mean()
 
        if self.FLAGS.mode=="GCN_ONLY": 
            self.outputs_att = gcn_out_att
            self.outputs_att_f = gcn_out_att
        elif self.FLAGS.mode=="INIT": 
            self.outputs_att = att_feature.view(-1,self.output_dim)
            self.outputs_att_f = att_feature.view(-1,self.output_dim)
        elif self.FLAGS.mode=="GCN+INIT": 
            self.outputs_att = gcn_out_att+att_feature.view(-1,self.output_dim)
            self.outputs_att_f = gcn_out_att+  att_feature.view(-1,self.output_dim)
        elif self.FLAGS.mode=="GCN+INIT+SIM": 
            self.outputs_att = gcn_out_att+att_feature.view(-1,self.output_dim) *sim
            self.outputs_att_f = gcn_out_att+  att_feature.view(-1,self.output_dim)

        hidden = self.layers_all1(torch.Tensor(self.inputs_all).cuda(), S_ALL)
        gcn_out_all = self.layers_all2(hidden, S_ALL)

        if self.FLAGS.mode=="GCN_ONLY": 
            self.outputs_all = gcn_out_all
        elif self.FLAGS.mode=="INIT": 
            self.outputs_all = torch.Tensor(self.inputs_all).cuda().view(-1,self.output_dim)
        else: 
            self.outputs_all = gcn_out_all + torch.Tensor(self.inputs_all).cuda().view(-1,self.output_dim)

        if self.FLAGS.setting == "inductive":
            loss_tmp = mask_classification_softmax_loss(self.outputs_all,
                                                                  self.outputs_att,
                                                                  self.input_dict['train_mask'],
                                                                  self.lookup_table_act_att,
                                                                  self.input_dict['labels'], self.FLAGS)
            accuracy = mask_classification_softmax_accuracy(self.outputs_all, self.outputs_att_f,
                                                           self.input_dict['train_mask'],
                                                           self.lookup_table_act_att,
                                                           self.input_dict['labels'], self.FLAGS)
        elif self.FLAGS.setting == "transductive":
            loss_tmp = mask_TD_classification_softmax_loss(self.outputs_all,
                                                                  self.outputs_att,
                                                                  self.input_dict['train_mask'],
                                                                  self.lookup_table_act_att,
                                                                  self.input_dict['labels'], self.FLAGS)
            accuracy = mask_classification_softmax_accuracy(self.outputs_all, self.outputs_att_f,
                                                           self.input_dict['train_mask'],
                                                           self.lookup_table_act_att,
                                                           self.input_dict['labels'], self.FLAGS)
        elif self.FLAGS.setting == "generalized_transductive":
            loss_tmp = mask_TDGZL_classification_softmax_loss(self.outputs_all,
                                                                  gcn_out_att,
                                                                  self.input_dict['train_mask'],
                                                                  self.lookup_table_act_att,
                                                                  self.input_dict['labels'], self.FLAGS, sim, att_feature.view(-1,self.output_dim))
            accuracy = mask_TDGZL_classification_softmax_accuracy(self.outputs_all, self.outputs_att_f,
                                                           self.input_dict['train_mask'],
                                                           self.lookup_table_act_att,
                                                           self.input_dict['labels'], self.FLAGS)
        return loss_tmp, accuracy

