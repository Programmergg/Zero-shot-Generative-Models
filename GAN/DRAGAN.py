'''
Now its attrs_best performance is unseen=56.66,seen=69.24,h=62.32
DRAGAN: Deep Regret Analytic GAN
The output of DRAGAN's D can be interpretted as a probability, similarly to
MMGAN and NSGAN. DRAGAN is similar to WGANGP, but seems less stable.
Proposes to study GANs from a regret minimization perspective. This
model is very similar to WGAN GP, in that it is applying a gradient penalty to
try and get at an improved training objective based on how D and G would
optimally perform. They apply the gradient penalty only close to the real data
manifold (whereas WGAN GP picks the gradient location on a random line between
a real and randomly generated fake sample). For further details, see
Section 2.5 of the paper.
'''
import argparse
import random
import torch
import util
from models import MLP_G,MLP_D
import classifier
import pre_classifier
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA', help='AWA,AWA2,APY,CUB,SUN')
parser.add_argument('--syn_num', type=int, default=2000, help='number features to generate per class')
parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--gzsl',action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--attSize', type=int, default=85, help='size of semantic features')
parser.add_argument('--nz', type=int, default=85, help='size of the latent z vector')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
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
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
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
netG = MLP_G(opt)
netD = MLP_D(opt)

#init parameters and loss function
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)

cls_criterion = nn.NLLLoss()

best_H=0
best_unseen = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def generate_syn_feature(netG, classes, attribute, num):
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
            output = netG(syn_att, syn_noise)
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

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * (fake_data + fake_data.std().cuda() * torch.rand(fake_data.size()).cuda()))
    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates,_= netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = pre_classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100)

for p in pretrain_cls.model.parameters(): # set requires_grad to False
    p.requires_grad = False

for epoch in range(opt.nepoch):
    for i in range(0, data.ntrain, opt.batch_size):
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()

            input_resv = Variable(input_res)
            input_attv = Variable(input_att)

            criticD_real,pred_real = netD(input_resv, input_attv)
            criticD_real = criticD_real.mean() + 1e-8

            # train with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise)

            fake = netG(input_attv, noisev)
            criticD_fake, pred_fake = netD(fake.detach(), input_attv)
            criticD_fake = criticD_fake.mean() + 1e-8

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_resv, fake.data,input_attv)

            D_cost = criticD_fake - criticD_real  + gradient_penalty
            D_cost.backward()
            optimizerD.step()

        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation

        netG.zero_grad()
        input_resv = Variable(input_res)
        input_attv = Variable(input_att)

        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(input_attv, noisev)
        criticG_fake, pred_fake=netD(fake, input_attv)
        G_cost = -criticG_fake.mean() + 1e-8

        c_errG_fake = cls_criterion(pretrain_cls.model(fake), input_label)

        reg_loss = Variable(torch.Tensor([0.0])).cuda()
        if opt.REG_W_LAMBDA != 0:
            for name, p in netG.named_parameters():
                if 'weight' in name:
                    reg_loss += p.pow(2).sum()
            reg_loss.mul_(opt.REG_W_LAMBDA)
        errG = G_cost + opt.cls_weight * c_errG_fake + reg_loss

        errG.backward()
        optimizerG.step()

    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, c_errG_fake:%.4f' % (epoch, opt.nepoch, D_cost.item(), G_cost.item(), c_errG_fake.item()))
    # evaluate the model, set G to evaluation mode
    netG.eval()
    netD.eval()

    # Generalized zero-shot learning
    syn_unseen_feature, syn_unseen_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
    train_X = torch.cat((data.train_feature, syn_unseen_feature), 0)
    train_Y = torch.cat((data.train_label, syn_unseen_label), 0)
    nclass = opt.nclass_all

    if opt.gzsl == True:
        cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass, True, 0.001, 0.5, 50, 2 * opt.syn_num,True)
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
        if cls.H > best_H:
            best_H = cls.H
            torch.save(netG.state_dict(),'../saved_models/seen{0}_unseen{1}_H{2}.pkl'.format(cls.acc_seen, cls.acc_unseen, cls.H))
            print('attrs_best models saved!')
    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        cls = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data,data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 50, 2 * opt.syn_num, False, epoch)
        if cls.acc > best_unseen:
            best_unseen = cls.acc
            print('attrs_best unseen acc is:', cls.acc)
    netG.train()
    netD.train()
