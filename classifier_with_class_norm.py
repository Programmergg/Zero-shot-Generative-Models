import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class CLASSIFIER:
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True, epoch=20):
        self.data_loader = data_loader
        self.train_X = _train_X
        self.train_Y = _train_Y

        self.test_seen_label = data_loader.test_seen_label
        self.test_seen_feature = data_loader.test_seen_feature

        self.test_unseen_label = data_loader.test_unseen_label
        self.test_unseen_feature = data_loader.test_unseen_feature

        self.USE_CLASS_STANDARTIZATION = True
        self.USE_PROPER_INIT = True

        self.test_idx = data_loader.test_idx
        self.seen_classes = data_loader.seenclasses.numpy().tolist()
        self.unseen_classes = data_loader.unseenclasses.numpy().tolist()
        self.seen_mask = np.array([(c in self.seen_classes) for c in range(_nclass)])
        self.unseen_mask = np.array([(c in self.unseen_classes) for c in range(_nclass)])

        self.all_feats = data_loader.all_feature
        self.all_labels = data_loader.all_labels
        self.attrs = data_loader.attribute.cuda()
        self.attrs_seen = self.attrs[self.seen_mask]
        self.attrs_unseen = self.attrs[self.unseen_mask]

        self.labels = self.all_labels.numpy()
        self.train_labels = self.train_Y
        self.test_labels = self.all_labels[self.test_idx].numpy()
        self.test_seen_idx = [i for i, y in enumerate(self.test_labels) if y in self.seen_classes]
        self.test_unseen_idx = [i for i, y in enumerate(self.test_labels) if y in self.unseen_classes]
        self.test_labels_remapped_seen = [(self.seen_classes.index(t) if t in self.seen_classes else -1) for t in self.test_labels]
        self.test_labels_remapped_unseen = [(self.unseen_classes.index(t) if t in self.unseen_classes else -1) for t in self.test_labels]

        self.ds_test = [(self.all_feats[i], int(self.all_labels[i])) for i in self.test_idx]
        self.ds_train = [(self.train_X[i], self.train_Y[i]) for i in range(self.train_X.shape[0])]
        self.train_dataloader = DataLoader(self.ds_train, batch_size=256, shuffle=True)
        self.test_dataloader = DataLoader(self.ds_test, batch_size=2048)
        self.class_indices_inside_test = {c: [i for i in range(len(self.test_idx)) if self.labels[self.test_idx[i]] == c] for c in range(_nclass)}

        self.classifier = Normalized_Classifier(self.attrs.shape[1], 1024, self.all_feats.shape[1]).cuda()
        self.optimizer_cls = optim.Adam(self.classifier.parameters(), lr=0.0005, weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_cls, gamma=0.1, step_size=25)

        self.zsl_unseen, self.gzsl_seen, self.gzsl_unseen, self.gzsl_H = self.train_softmax_classfier()

    def train_softmax_classfier(self):
        for _ in tqdm(range(50)):
            self.classifier.train()
            for i, batch in enumerate(self.train_dataloader):
                feats = torch.from_numpy(np.array(batch[0])).cuda()
                targets = torch.from_numpy(np.array(batch[1])).cuda()
                logits = self.classifier(feats, self.attrs)
                loss = F.cross_entropy(logits, targets)
                self.classifier.zero_grad()
                loss.backward()
                self.optimizer_cls.step()
            self.scheduler.step()
        self.classifier.eval()

        logits = [self.classifier(x.cuda(), self.attrs).cpu() for x, _ in self.test_dataloader]
        logits = torch.cat(logits, dim=0)
        logits[:, self.seen_mask] *= (0.95 if self.data_loader.name != "CUB" or "CUB2" else 1.0)  # Trading a bit of gzsl-s for a bit of gzsl-u
        preds_gzsl = logits.argmax(dim=1).numpy()  # predict test labels
        preds_zsl_s = logits[:, self.seen_mask].argmax(dim=1).numpy()
        preds_zsl_u = logits[:, ~self.seen_mask].argmax(dim=1).numpy()
        guessed_zsl_u = (preds_zsl_u == self.test_labels_remapped_unseen)
        guessed_gzsl = (preds_gzsl == self.test_labels)

        zsl_unseen_acc = np.mean([guessed_zsl_u[cls_idx].mean().item() for cls_idx in [self.class_indices_inside_test[c] for c in self.unseen_classes]])
        gzsl_seen_acc = np.mean([guessed_gzsl[cls_idx].mean().item() for cls_idx in [self.class_indices_inside_test[c] for c in self.seen_classes]])
        gzsl_unseen_acc = np.mean([guessed_gzsl[cls_idx].mean().item() for cls_idx in [self.class_indices_inside_test[c] for c in self.unseen_classes]])
        gzsl_harmonic = 2 * (gzsl_seen_acc * gzsl_unseen_acc) / (gzsl_seen_acc + gzsl_unseen_acc)
        return zsl_unseen_acc, gzsl_seen_acc, gzsl_unseen_acc, gzsl_harmonic

class Normalized_Classifier(nn.Module):
    def __init__(self, attr_dim: int, hid_dim: int, proto_dim: int):
        super(Normalized_Classifier, self).__init__()
        self.fc1 = nn.Linear(attr_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, proto_dim)
        self.relu = nn.ReLU(True)

        weight_var = 1 / (hid_dim * proto_dim)
        b = np.sqrt(3 * weight_var)
        self.fc3.weight.data.uniform_(-b, b)

    def forward(self, x ,attrs):
        #generate soul samples
        res = self.relu(self.fc1(attrs))
        res = self.fc2(res)
        protos = self.relu(self.fc3(res))

        # x_ns = x
        # protos_ns = protos
        # x_ns = 5 * F.normalize(x, p=2, dim=x.dim() - 1, eps=1e-12)
        # protos_ns = 5 * F.normalize(protos, p=2, dim=protos.dim() - 1, eps=1e-12)
        x_ns = 5 * x / x.norm(dim=1, keepdim=True)  # [batch_size, x_dim]
        protos_ns = 5 * protos / protos.norm(dim=1, keepdim=True)  # [num_classes, x_dim]
        logits = x_ns @ protos_ns.t()  # [batch_size, num_classes]
        return logits