'''
Now its attrs_best performance is unseen=57.87,seen=68.86,h=62.89
encoder and decoder jointly optimize
'''
import argparse
import random
import torch
import util
from models import VAE
import classifier
import pre_classifier
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA', help='AWA,AWA2,APY,CUB,SUN')
parser.add_argument('--syn_num', type=int, default=1800, help='number features to generate per class')
parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--gzsl',action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--attSize', type=int, default=85, help='size of semantic features')
parser.add_argument('--z_dim', type=int, default=85, help='size of the latent z vector')
parser.add_argument('--nz', type=int, default=85, help='size of the noise')
parser.add_argument('--cls_weight', type=float, default=0.01, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--nclass_all', type=int, default=50, help='number of all classes')
parser.add_argument('--nclass_seen', type=int, default=40, help='number of seen classes')
parser.add_argument('--final_classifier', default='softmax', help='the classifier for final classification. softmax or knn')
parser.add_argument('--REG_W_LAMBDA',type=float,default=0.0004,help = 'the regularization for generator')

parser.add_argument('--dataroot', default='../datasets', help='path to dataset')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, default=None,help='manual seed')#3483
opt = parser.parse_args()

#init random seeds for every package
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

#init torch settings
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# initialize networks' structure
vae = VAE(opt)

#init parameters and loss function
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)

cls_criterion = nn.NLLLoss()

best_H=0
best_unseen = 0

# setup optimizer
optimizer = optim.Adam(vae.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))

if opt.cuda:
    vae.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def generate_syn_feature(vae, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    with torch.no_grad():
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            output, mu, log_var= vae(syn_att, syn_noise)
            syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i*num, num).fill_(iclass)
    return syn_feature, syn_label

def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        if torch.sum(idx)==0:
            acc_per_class +=0
        else:
            acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    acc_per_class /= float(target_classes.size(0))
    return acc_per_class

# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = pre_classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100)

for p in pretrain_cls.model.parameters(): # set requires_grad to False
    p.requires_grad = False

for epoch in range(opt.nepoch):
    for i in range(0, data.ntrain, opt.batch_size):
        vae.zero_grad()

        sample()
        input_resv = Variable(input_res)
        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)

        mu, log_var = vae.encoder(input_attv, noisev)
        recon_att_loss = torch.sum((mu - input_attv) ** 2)

        # kl_diverge = torch.sum(0.5 * (mu ** 2 + torch.exp(log_var) - log_var - 1))

        epsilon = torch.randn(mu.shape).cuda()
        z = mu + epsilon * torch.exp(log_var / 2)
        recon_visual = vae.decoder(z)
        c_errG_fake = cls_criterion(pretrain_cls.model(recon_visual), input_label)
        recon_visual_loss = torch.sum((recon_visual - input_resv) ** 2)

        reg_loss = Variable(torch.Tensor([0.0])).cuda()
        if opt.REG_W_LAMBDA != 0:
            for name, p in vae.named_parameters():
                if 'weight' in name:
                    reg_loss += p.pow(2).sum()
            reg_loss.mul_(opt.REG_W_LAMBDA)

        total_loss = recon_visual_loss + recon_att_loss + opt.cls_weight * c_errG_fake + reg_loss
        total_loss.backward()
        optimizer.step()
    print('[%d/%d] visual_loss: %.4f, att_loss: %.4f, total_loss: %.4f' % (epoch, opt.nepoch,recon_visual_loss.item(), recon_att_loss.item(), total_loss.item()))
    vae.eval()

    # Generalized zero-shot learning
    syn_unseen_feature, syn_unseen_label = generate_syn_feature(vae, data.unseenclasses, data.attribute, opt.syn_num)
    train_X = torch.cat((data.train_feature, syn_unseen_feature), 0)
    train_Y = torch.cat((data.train_label, syn_unseen_label), 0)
    nclass = opt.nclass_all
    if opt.gzsl == True:
        cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass, True, 0.001, 0.5, 50, 2 * opt.syn_num,True)
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
        if cls.H > best_H:
            best_H = cls.H
            torch.save(vae.state_dict(),'../saved_models/seen{0}_unseen{1}_H{2}.pkl'.format(cls.acc_seen, cls.acc_unseen, cls.H))
            print('attrs_best models saved!')
    else:
        syn_feature, syn_label = generate_syn_feature(vae, data.unseenclasses, data.attribute, opt.syn_num)
        cls = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data,data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 50, 2 * opt.syn_num, False, epoch)
        if cls.acc > best_unseen:
            best_unseen = cls.acc
            print('attrs_best unseen acc is:', cls.acc)
    vae.train()