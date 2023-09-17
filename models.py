import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, att, noise):
        h = torch.cat((att, noise), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class MLP_G_2(nn.Module):
    def __init__(self, opt):
        super(MLP_G_2, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.fc3 = nn.Linear(opt.resSize, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, att, noise):
        h = torch.cat((att, noise), 1)
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return h

class MLP_G_InfoGAN(nn.Module):
    def __init__(self, opt):
        super(MLP_G_InfoGAN, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.z_dim + opt.disc_dim +opt.cont_dim, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, att, latent):
        h = torch.cat((att, latent), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class CNZSLModel(nn.Module):
    def __init__(self, attr_dim: int, hid_dim: int, proto_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(attr_dim, hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, hid_dim),
            ClassStandardization(hid_dim),
            nn.ReLU(),

            ClassStandardization(hid_dim),
            nn.Linear(hid_dim, proto_dim),
            nn.ReLU(),
        )

        weight_var = 1 / (hid_dim * proto_dim)
        b = np.sqrt(3 * weight_var)
        self.model[-2].weight.data.uniform_(-b, b)

    def forward(self, x, attrs):
        protos = self.model(attrs)
        x_ns = 5 * x / x.norm(dim=1, keepdim=True)  # [batch_size, x_dim]
        protos_ns = 5 * protos / protos.norm(dim=1, keepdim=True)  # [num_classes, x_dim]
        logits = x_ns @ protos_ns.t()  # [batch_size, num_classes]
        return logits

class ClassStandardization(nn.Module):
    def __init__(self, feat_dim: int):
        super(ClassStandardization, self).__init__()
        self.running_mean = nn.Parameter(torch.zeros(feat_dim), requires_grad = False)
        self.running_var = nn.Parameter(torch.ones(feat_dim), requires_grad = False)

    def forward(self, class_feats):
        """
        Input: class_feats of shape [num_classes, feat_dim]
        Output: class_feats (standardized) of shape [num_classes, feat_dim]
        """
        if self.training:
            batch_mean = class_feats.mean(dim = 0)
            batch_var = class_feats.var(dim = 0)

            # Normalizing the batch
            result = (class_feats - batch_mean.unsqueeze(0)) / (batch_var.unsqueeze(0) + 1e-8)

            # Updating the running mean/std
            self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * batch_mean.detach()
            self.running_var.data = 0.9 * self.running_var.data + 0.1 * batch_var.detach()
        else:
            # Using accumulated statistics
            # Attention! For the test inference, we cant use batch-wise statistics,
            # only the accumulated ones. Otherwise, it will be quite transductive
            result = (class_feats - self.running_mean.unsqueeze(0)) / (self.running_var.unsqueeze(0) + 1e-8)
        return result

class MLP_D(nn.Module):
    def __init__(self, opt):
        super(MLP_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.fc3 = nn.Linear(opt.ndh, opt.nclass_all)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, x, att):
        # pred1 is discriminator,pred2 is classifier pred
        hidden = torch.cat((x, att), 1)
        hidden = self.lrelu(self.fc1(hidden))
        pred1 = self.fc2(hidden)
        pred2 = self.logic(self.fc3(hidden))
        return pred1, pred2

class MLP_D_2(nn.Module):
    '''This Discriminator's output is by sigmoid -->(0,1)'''
    def __init__(self, opt):
        super(MLP_D_2, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.fc3=nn.Linear(opt.ndh,opt.nclass_all)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, x, att):
        #pred1 is discriminator,pred2 is classifier pred
        hidden=torch.cat((x,att),1)
        hidden = self.lrelu(self.fc1(hidden))
        pred1=torch.sigmoid(self.fc2(hidden))
        pred2=self.logic(self.fc3(hidden))
        return pred1,pred2

class MLP_D_3(nn.Module):
    def __init__(self, opt):
        '''This discriminator is designed for BEGAN'''
        super(MLP_D_3, self).__init__()
        self.encoder = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.decoder = nn.Linear(opt.ndh, opt.resSize + opt.attSize)
        self.fc3 = nn.Linear(opt.ndh, opt.nclass_all)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)
    def forward(self, x, att):
        # pred1 is reconstruct features,pred2 is classifier pred
        hidden = torch.cat((x, att), 1)
        hidden = self.lrelu(self.encoder(hidden))
        pred1 = self.decoder(hidden)
        pred2 = self.logic(self.fc3(hidden))
        return pred1, pred2

class MLP_D_4(nn.Module):
    def __init__(self, opt):
        super(MLP_D_4, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self,x,att):
        #pred1 is discriminator,pred2 is classifier pred
        hidden=torch.cat((x,att),1)
        hidden = self.lrelu(self.fc1(hidden))
        pred1=self.fc2(hidden)
        return pred1

class ConvertNet(nn.Module):
    def __init__(self, opt):
        super(ConvertNet, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

    def forward(self, ori_visual_embeddings, netG):
        reg_attributes = self.relu(self.fc1(ori_visual_embeddings))

        zero_noise = torch.zeros((reg_attributes.shape[0], reg_attributes.shape[1])).cuda()
        with torch.no_grad():
            reg_visual_embeddings = netG(Variable(reg_attributes), Variable(zero_noise))
        return reg_attributes,reg_visual_embeddings

class Auxiliary(nn.Module):
    '''
    Auxiliary network Q(c|x) that approximates P(c|x), the true posterier.
    Input are visual features and output are latent variables.
    '''
    def __init__(self, opt):
        super(Auxiliary, self).__init__()
        self.__dict__.update(locals())

        self.linear = nn.Linear(opt.resSize, opt.ndh)
        self.inference = nn.Linear(opt.ndh, opt.disc_dim + opt.cont_dim)
        self.relu = nn.ReLU(True)
        self.disc_dim = opt.disc_dim
        self.cont_dim = opt.cont_dim

    def forward(self, x):
        h = self.relu(self.linear(x))
        inferred = self.inference(h)
        discrete, continuous = inferred[:, :self.disc_dim], inferred[:, self.disc_dim:]
        return discrete, continuous

class Encoder(nn.Module):
    '''Encoder of AutoEncoder network'''
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, att, noise):
        h = torch.cat((att, noise), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class Decoder(nn.Module):
    '''Decoder of AutoEncoder network'''
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return h

class AutoEncoder(nn.Module):
    def __init__(self, opt):
        super(AutoEncoder, self).__init__()
        self.__dict__.update(locals())
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)

    def forward(self, att, noise):
        res_att = self.decoder(self.encoder(att, noise))
        return res_att

class Encoder_VAE(nn.Module):
    '''encoder for VAE'''
    def __init__(self, opt):
        super(Encoder_VAE, self).__init__()
        self.linear = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.mu = nn.Linear(opt.ngh, opt.nz)
        self.log_var = nn.Linear(opt.ngh, opt.nz)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, att, noise):
        h = torch.cat((att, noise), 1)
        h = self.lrelu(self.linear(h))
        mu, log_var = self.mu(h),self.log_var(h)
        return mu, log_var

class Decoder_VAE(nn.Module):
    '''decoder for VAE'''
    def __init__(self, opt):
        super(Decoder_VAE, self).__init__()
        self.linear = nn.Linear(opt.nz, opt.ndh)
        self.recon = nn.Linear(opt.ndh,opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, z):
        h = self.lrelu(self.linear(z))
        h = self.relu(self.recon(h))
        return h

class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()
        self.__dict__.update(locals())

        self.encoder = Encoder_VAE(opt)
        self.decoder = Decoder_VAE(opt)

    def reparameterize(self, mu, log_var):
        epsilon = torch.randn(mu.shape).cuda()
        z = mu + epsilon * torch.exp(log_var/2)
        return z

    def forward(self , att, noise):
        mu, log_var = self.encoder(att, noise)
        z = self.reparameterize(mu,log_var)
        out = self.decoder(z)
        return out, mu, log_var

class Encoder_noise(nn.Module):
    '''This module is for generating noise from visual features'''
    def __init__(self, opt):
        super(Encoder_noise, self).__init__()
        self.__dict__.update(locals())
        self.linear = nn.Linear(opt.resSize + opt.attSize, opt.ngh)
        # self.linear = nn.Linear(opt.resSize, opt.ngh)
        self.mu = nn.Linear(opt.ngh, opt.nz)
        self.var = nn.Linear(opt.ngh, opt.nz)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, visual_feats, attrs):
        concat_feats = torch.cat((visual_feats,attrs), dim=1)
        # concat_feats = visual_feats
        hidden = torch.tanh(self.linear(concat_feats))
        mu, var = torch.tanh(self.mu(hidden)), torch.tanh(self.var(hidden))
        return mu, var

class Decoder_noise(nn.Module):
    '''This module is for decoding mu and var to reconstructed visual features'''
    def __init__(self, opt):
        super(Decoder_noise, self).__init__()
        self.__dict__.update(locals())
        self.linear = nn.Linear(opt.nz, opt.ndh)
        self.recon = nn.Linear(opt.ndh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, z):
        h = self.lrelu(self.linear(z))
        h = self.relu(self.recon(h))
        return h

class VAE_noise(nn.Module):
    def __init__(self, opt):
        super(VAE_noise, self).__init__()
        self.__dict__.update(locals())
        self.encoder = Encoder_noise(opt)
        self.decoder = Decoder_noise(opt)

    def reparameterize(self, mu, log_var):
        epsilon = torch.randn(mu.shape).cuda()
        z = mu + epsilon * torch.exp(log_var / 2)
        return z

    def forward(self, visual_feats, attrs):
        mu, log_var = self.encoder(visual_feats, attrs)
        z = self.reparameterize(mu, log_var)
        recon_visual_feats = self.decoder(z)
        return recon_visual_feats, mu, log_var

class TripCenterLoss_min_margin(nn.Module):
    def __init__(self, num_classes=40, feat_dim=85, use_gpu=True):
        super(TripCenterLoss_min_margin, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, margin):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat[mask]

        other=torch.FloatTensor(batch_size,self.num_classes-1).cuda()
        for i in range(batch_size):
            other[i]=(distmat[i,mask[i,:]==0])
        dist_min,_=other.min(dim=1)
        loss = torch.max(margin+dist-dist_min,torch.tensor(0.0).cuda()).sum() / batch_size
        return loss

class BIRVAE(nn.Module):
    def __init__(self, opt):
        super(BIRVAE, self).__init__()
        self.__dict__.update(locals())
        self.encoder = Encoder_VAE(opt)
        self.decoder = Decoder_VAE(opt)

        self.shape = int(opt.resSize ** 0.5)
        self.I = 13.3 # I indicates how many 'bits' should be let through
        self.set_var = 1/(4**(self.I/opt.z_dim))

    def reparameterize(self, mu):
        eps = torch.from_numpy(np.random.normal(loc=0.0,scale=self.set_var,size=mu.shape)).float().cuda()
        z = mu + eps
        return z

    def forward(self, att, noise):
        mu, log_var = self.encoder(att,noise)
        z = self.reparameterize(mu)
        out = self.decoder(z)
        return out, z

class Mapping(nn.Module):
    def __init__(self, opt):
        super(Mapping, self).__init__()
        self.latensize=opt.latenSize
        self.encoder_linear = nn.Linear(opt.resSize, opt.latenSize*2)
        self.discriminator = nn.Linear(opt.latenSize, 1)
        self.classifier = nn.Linear(opt.latenSize, opt.nclass_seen)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, x, train_G=False):
        laten=self.lrelu(self.encoder_linear(x))
        mus,stds = laten[:,:self.latensize],laten[:,self.latensize:]
        stds=self.sigmoid(stds)
        encoder_out = reparameter(mus, stds)
        if not train_G:
            dis_out = self.discriminator(encoder_out)
        else:
            dis_out = self.discriminator(mus)
        pred=self.logic(self.classifier(mus))
        return mus,stds,dis_out,pred,encoder_out

def reparameter(mu,sigma):
    return (torch.randn_like(mu) *sigma) + mu

class ML(nn.Module):
    def __init__(self, opt):
        super(ML, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.out_dim)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x):
        hidden = self.lrelu(self.fc1(x))
        pred1 = self.fc2(hidden)
        return pred1