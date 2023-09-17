import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import util
import sys
import models
import numpy as np

class CLASSIFIER:
    # train_Y is interger
    # CLASSIFIER(syn_feature,util.map_label(syn_label,data.unseenclasses),data,data.unseenclasses.size(0),opt.cuda,opt.classifier_lr, 0.5, 25, opt.syn_num, False)
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20,
                 _batch_size=100, generalized=True, epoch=20):
        self.train_X = _train_X
        self.train_Y = _train_Y

        self.test_seen_label = data_loader.test_seen_label
        self.test_seen_feature = data_loader.test_seen_feature
        # self.test_seen_feature = torch.from_numpy(np.power(self.test_seen_feature[:, ].float().numpy(), 0.5))
        # mu_seen, var_seen = vae.encoder(data_loader.test_seen_feature.cuda(), data_loader.attribute[self.test_seen_label].cuda())
        # self.test_seen_feature = self.reparameterize(mu_seen, var_seen)

        self.test_unseen_label = data_loader.test_unseen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        # self.test_unseen_feature = torch.from_numpy(np.power(self.test_unseen_feature[:, ].float().numpy(), 0.5))
        # mu_unseen, var_unseen = vae.encoder(data_loader.test_unseen_feature.cuda(), data_loader.attribute[self.test_unseen_label].cuda())
        # self.test_unseen_feature = self.reparameterize(mu_unseen, var_unseen)

        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses

        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(models.weights_init)
        self.criterion = nn.NLLLoss()

        self.data = data_loader

        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)

        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        self.epoch = epoch

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        self.backup_X = _train_X
        self.backup_Y = _train_Y

        if generalized:
            self.acc_seen, self.acc_unseen, self.H = self.fit()
        else:
            self.acc = self.fit_zsl()

    def reparameterize(self, mu, log_var):
        epsilon = torch.randn(mu.shape).cuda()
        z = mu + epsilon * torch.exp(log_var / 2)
        return z

    def fit_zsl(self):
        first_acc = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                inputv = Variable(self.input)  # fake_feature
                labelv = Variable(self.label)  # fake_labels
                output = self.model(inputv)
                loss = self.criterion(output, labelv)  # 使用fake_unseen_feature和labels来训练分类器
                loss.backward()
                self.optimizer.step()

            acc, pred, output, all_acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if acc > first_acc:
                first_acc = acc

        print('Unseen Acc: {:.2f}%'.format(first_acc * 100))
        sys.stdout.flush()
        return first_acc

    def val(self, test_X, test_label, target_classes, second=False):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        all_output = None
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            if self.cuda:
                output = self.model(Variable(test_X[start:end].cuda()))
            else:
                output = self.model(Variable(test_X[start:end]))
            if all_output is None:
                all_output = output
            else:
                all_output = torch.cat((all_output, output), 0)
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end
        acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label,
                                         target_classes.size(0))
        acc_all = self.compute_every_class_acc(util.map_label(test_label, target_classes), predicted_label,
                                               target_classes.size(0))
        return acc, predicted_label, all_output, acc_all

    def val_gzsl(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        all_output = None
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            if self.cuda:
                output = self.model(Variable(test_X[start:end].cuda()))
            else:
                output = self.model(Variable(test_X[start:end]))

            if all_output is None:
                all_output = output
            else:
                all_output = torch.cat((all_output, output), 0)
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end
        # acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc, predicted_label, all_output

    def split_pred(self, all_pred, real_label):
        seen_pred = None
        seen_label = None
        unseen_pred = None
        unseen_label = None
        for i in self.seenclasses:
            idx = (real_label == i)
            if seen_pred is None:
                seen_pred = all_pred[idx]
                seen_label = real_label[idx]
            else:
                seen_pred = torch.cat((seen_pred, all_pred[idx]), 0)
                seen_label = torch.cat((seen_label, real_label[idx]))

        for i in self.unseenclasses:
            idx = (real_label == i)
            if unseen_pred is None:
                unseen_pred = all_pred[idx]
                unseen_label = real_label[idx]
            else:
                unseen_pred = torch.cat((unseen_pred, all_pred[idx]), 0)
                unseen_label = torch.cat((unseen_label, real_label[idx]), 0)

        return seen_pred, seen_label, unseen_pred, unseen_label

    # for gzsl
    def fit(self):
        all_test_label = torch.cat((self.test_seen_label, self.test_unseen_label), 0)  # [4958+5685]
        first_all_pred = None

        best_H = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):  # self.ntrain=22057, self.batch_size=300
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()

            acc_seen, pred_seen, output_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen, pred_unseen, output_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            if H >= best_H:
                best_H = H
                first_all_pred = torch.cat((pred_seen, pred_unseen), 0)
        first_seen_pred, first_seen_label, first_unseen_pred, first_unseen_label = self.split_pred(first_all_pred, all_test_label)
        acc_first_seen = self.compute_per_class_acc_gzsl(first_seen_label, first_seen_pred, self.seenclasses)
        acc_first_unseen = self.compute_per_class_acc_gzsl(first_unseen_label, first_unseen_pred, self.unseenclasses)
        acc_first_H = 2 * acc_first_seen * acc_first_unseen / (acc_first_seen + acc_first_unseen)
        sys.stdout.flush()
        return acc_first_seen, acc_first_unseen, acc_first_H

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.train_X[start:end], self.train_Y[start:end]

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            if torch.sum(idx).float() != 0:
                acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
        return acc_per_class.mean()

    def compute_every_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            if torch.sum(idx).float() != 0:
                acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
        return acc_per_class

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            if torch.sum(idx).float() == 0:
                continue
            else:
                acc_per_class += torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
        acc_per_class /= target_classes.size(0)
        return acc_per_class


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o