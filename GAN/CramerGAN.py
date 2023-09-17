'''
Now its attrs_best performance is unseen=60.62,seen=70.36,h=65.13
'''
import argparse
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import util
import pre_classifier
import classifier
from models import MLP_G,MLP_D
from sklearn.neighbors import KNeighborsClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA', help='dataset for zsl dataset')
parser.add_argument('--syn_num', type=int, default=2000, help='number features to generate per class')
parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--attSize', type=int, default=85, help='size of semantic features')
parser.add_argument('--nz', type=int, default=85, help='size of the latent z vector')
parser.add_argument('--lambda1', type=float, default=2, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.01, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--nclass_all', type=int, default=50, help='number of all classes')
parser.add_argument('--gen_param', type=float, default=1.0, help='proto param 1')
parser.add_argument('--REG_W_LAMBDA',type=float,default=0.0004,help='regularization param')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--final_classifier',default='softmax',help='softmax or knn')
parser.add_argument('--manualSeed', type=int,default=None,help='manual seed')

parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

parser.add_argument('--dataroot', default='../datasets', help='path to dataset')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")

#param init
opt = parser.parse_args()
torch.cuda.set_device(opt.ngpu)
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("Training samples: ", data.ntrain)#19832

# initialize generator and discriminator
netG = MLP_G(opt)
netD = MLP_D(opt)

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)#[64,2048]
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)#[64,85]
noise = torch.FloatTensor(opt.batch_size, opt.nz)#[64,85]
noise2 = torch.FloatTensor(opt.batch_size, opt.nz)#[64,85]

one = torch.FloatTensor([1])
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)#[64,]

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input_res = input_res.cuda()
    input_att = input_att.cuda()
    noise,noise2 = noise.cuda(),noise2.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)#s label is normal label based 0
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))#map normal label into 0-39

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_att), Variable(syn_noise))
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i
    return mapped_label

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    # print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates,_ = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1)) ** 6).mean() * opt.lambda1
    return gradient_penalty

def Critic(netD, real, fake2,att):
    net_real,_ = netD(real,att)
    return torch.norm(net_real - netD(fake2,att)[0], p=2, dim=1) - \
           torch.norm(net_real, p =2,  dim=1)

def compute_per_class_acc_gzsl(predicted_label, test_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        if torch.sum(idx).float()==0:
            continue
        else:
            acc_per_class += torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
    acc_per_class /= target_classes.size(0)
    return acc_per_class

# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = pre_classifier.CLASSIFIER(_train_X=data.train_feature, _train_Y=util.map_label(data.train_label, data.seenclasses),
                                     _nclass=data.seenclasses.size(0), _input_dim=opt.resSize, _cuda=opt.cuda, _lr=0.001, _beta1=0.5, _nepoch=100, _batch_size=100,
                                     pretrain_classifer=opt.pretrain_classifier)

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters():  # set requires_grad to False
    p.requires_grad = False

best_H=0
best_unseen=0

for epoch in range(opt.nepoch):
    for i in range(0, data.ntrain, opt.batch_size):
        for p in netD.parameters():
            p.requires_grad = True

        #optimize discriminator
        for iter_d in range(opt.critic_iter):#5
            sample()#samples for input_res,input_att,input_label
            netD.zero_grad()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            input_labelv=Variable(input_label)

            criticD_real,pred_real = netD(input_resv, input_attv)
            criticD_real_loss = criticD_real.mean()
            # criticD_real_loss.backward(mone,retain_graph=True)

            noise.normal_(0, 1)
            noise2.normal_(0,1)
            noisev = Variable(noise)
            noisev2=Variable(noise2)

            fake = netG(input_attv, noisev)
            fake2 = netG(input_attv, noisev2)

            criticD_fake,pred_fake = netD(fake.detach(), input_attv)
            criticD_fake2,pred_fake2 = netD(fake2.detach(), input_attv)

            criticD_fake_loss = criticD_fake.mean()
            # criticD_fake_loss.backward(one,retain_graph=True)

            gen_loss = torch.mean(
                torch.norm(criticD_real - criticD_fake, p=2, dim=1)
                + torch.norm(criticD_real - criticD_fake2, p=2, dim=1)
                - torch.norm(criticD_fake - criticD_fake2, p=2, dim=1)
            )

            surrogate = torch.mean(Critic(netD, input_resv, fake2,input_attv) -Critic(netD, fake, fake2,input_attv))
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)

            disc_loss=-surrogate+gradient_penalty
            disc_loss.backward(retain_graph=True)
            optimizerD.step()

        #stop optimize discriminator
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation

        #begin optimizing generator
        netG.zero_grad()
        input_resv = Variable(input_resv)
        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(input_attv, noisev)
        criticG_fake,pred_fake = netD(fake, input_attv)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake

        # classification loss
        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))

        # ||W||_2 regularization
        reg_loss = Variable(torch.Tensor([0.0])).cuda()
        if opt.REG_W_LAMBDA != 0:
            for name, p in netG.named_parameters():
                if 'weight' in name:
                    reg_loss += p.pow(2).sum()
            reg_loss.mul_(opt.REG_W_LAMBDA)

        errG = G_cost + opt.cls_weight * c_errG + gen_loss * opt.gen_param + reg_loss
        errG.backward(retain_graph=True)
        optimizerG.step()
    print('EP[%d/%d]************************************************************************************' % (epoch, opt.nepoch))

    # evaluate the model, set G to evaluation mode
    netG.eval()
    # Generalized zero-shot learning
    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        syn_seen_feature,syn_seen_label = generate_syn_feature(netG, data.seenclasses, data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all
        if opt.final_classifier == 'softmax':
            cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 50, 2*opt.syn_num,True)
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
            if cls.H>best_H:
                best_H=cls.H
                torch.save(netG.state_dict(),'../saved_models/seen{0}_unseen{1}_H{2}.pkl'.format(cls.acc_seen,cls.acc_unseen,cls.H))
                print('model saved!!!!')
        elif opt.final_classifier == 'knn':
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(X=train_X.cuda().cpu(), y=train_Y)
            pred_Y_s = torch.from_numpy(clf.predict(data.test_seen_feature.cuda().cpu()))
            pred_Y_u = torch.from_numpy(clf.predict(data.test_unseen_feature.cuda().cpu()))
            acc_seen = compute_per_class_acc_gzsl(pred_Y_s, data.test_seen_label, data.seenclasses)
            acc_unseen = compute_per_class_acc_gzsl(pred_Y_u, data.test_unseen_label, data.unseenclasses)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (acc_unseen, acc_seen, H))
            if H>=best_H:
                best_H=H
                print('model saved!!!')
    # Zero-shot learning
    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        cls = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data,
                                     data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 50, 2*opt.syn_num,
                                     False, epoch)
        if cls.acc>best_unseen:
            best_unseen=cls.acc
            print('attrs_best unseen acc is:',cls.acc)
    cls = None
    netG.train()