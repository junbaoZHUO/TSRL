from inits import *
import torch

class SparseDropout(torch.nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob=1-dprob

    def forward(self, x):
        mask=((torch.rand(x._values().size())+(self.kprob)).floor()).type(torch.bool)
        rc=x._indices()[:,mask]
        val=x._values()[mask]*(1.0/self.kprob)
        return torch.sparse.FloatTensor(rc, val)

def to_torch_sparse_tensor(M):
    M = M.tocoo().astype(np.float32)
    print(M.shape)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    T = torch.sparse.FloatTensor(indices, values, shape)
    return T

def dot(x, y, sparse=False):
    """Wrapper for torch.matmul (sparse vs dense)."""
    if sparse:
        res = torch.matmul(to_torch_sparse_tensor(x).cuda(), y)#check if y is dense
    else:
        res = torch.matmul(x, y)
    return res

class GraphConvolution(torch.nn.Module):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, dropout=0.,
                 sparse_inputs=False, act=torch.nn.ReLU(), bias=False,
                 featureless=False, type='', **kwargs):
        super(GraphConvolution, self).__init__()#**kwargs)
        self.input_dim=input_dim

        if dropout:
            self.dropout = dropout
        else:
            self.dropout = 0.

        self.act = act
        if type == 'all':
            self.support = [1]
        elif type =='att':
            self.support = [1]
        else:
            raise NameError
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias


        self.W = torch.nn.parameter.Parameter(glorot([input_dim, output_dim]).cuda())

        if self.bias:
            self.B = torch.nn.Parameter(zeros([output_dim]).cuda())

    def forward(self, inputs, S):
        x = inputs

        supports = list()
        for i in range(len(S)):
            if not self.featureless:
                pre_sup = dot(x, self.W,
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.W
            support = torch.spmm(torch.sparse.FloatTensor(torch.LongTensor(S[i][0]).t(),torch.FloatTensor(S[i][1]),torch.Size(S[i][2])).cuda(), pre_sup)
            supports.append(support)
        output = torch.sum(torch.stack(supports,0),0)

        # bias
        if self.bias:
            output += self.B
        output = self.act(output)

        return self.act(output)

class conv(torch.nn.Module):
    def __init__(self, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True):
        super(conv, self).__init__()#**kwargs)
        self.w = torch.nn.Parameter(torch.rand(channels, channels, kernel, kernel).normal_(1.0, 0.02))
        self.bias = torch.nn.Parameter(torch.zeros(channels, 1, 1))
        self.pad=pad
        self.stride = stride

    def forward(self, x):
        pad = self.pad
        x = torch.nn.functional.pad(x, [0, 0, pad, pad, pad, pad], mode='constant', value=0)

        x = torch.nn.functional.conv2d(x, spectral_norm(self.w),
                         stride=[1, self.stride])
        x += self.bias


        return x


def spectral_norm(w, iteration=1):
    w_shape = w.shape
    w = w.view([-1, w_shape[-1]])

    u = torch.rand(1, w_shape[-1]).cuda()

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = torch.matmul(u_hat, w.t())
        v_hat = l2_norm(v_)

        u_ = torch.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = torch.matmul(torch.matmul(v_hat, w), u_hat.t())
    w_norm = w / sigma

    w_norm = w_norm.view(w_shape)

    return w_norm

def l2_norm(v, eps=1e-12):
    return v / (torch.sum(v ** 2) ** 0.5 + eps)

