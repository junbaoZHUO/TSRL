import torch
import torch
import torch.nn as nn
import numpy as np
import ot

def mask_classification_softmax_loss(output_all, output_att, train_mask, attention_mask, labels, FLAGS):

    label_num = train_mask.shape[0]
    classifiers = output_all[0:label_num, :]
    output_att = output_att.view(labels.shape[0], -1, FLAGS.output_dim)
    labels=torch.zeros(labels.shape[0], label_num).scatter_(1, torch.LongTensor(labels).view(-1, 1), 1).cuda() 
    losses = torch.zeros(1, label_num).cuda()

    for i in range(labels.shape[0]):
        output_att_one = output_att[i, :, :]
        label = labels[i]
        attend_features = output_att_one[attention_mask.to_dense().long()].sum(dim=1)
        raw_score_per_action = classifiers * attend_features
        raw_score_per_action = torch.sum(raw_score_per_action, dim=1)
        mask = torch.Tensor(train_mask.astype(np.int32)).float().cuda()

        mask /= torch.mean(mask)
        raw_score_per_action *= mask
        raw_score_per_action = torch.nn.functional.log_softmax(raw_score_per_action)
        loss = -raw_score_per_action.view(1, -1) * label
        losses += loss

    return torch.sum(losses)/labels.shape[0]

def mask_classification_softmax_accuracy(output_all, output_att, train_mask, attention_mask, labels, FLAGS):

    label_num = train_mask.shape[0]
    classifiers = output_all[0:label_num, :]
    output_att = output_att.view(labels.shape[0], -1, FLAGS.output_dim)
    mask = torch.Tensor(train_mask.astype(np.int32)).float().cuda()
 
    accuracy=0
    for i in range(labels.shape[0]):
        output_att_one = output_att[i, :, :]
        label = labels[i]
        attend_features = output_att_one[attention_mask.to_dense().long()].sum(dim=1)
        raw_score_per_action = classifiers * attend_features
        raw_score_per_action = torch.sum(raw_score_per_action,dim=1)
        raw_score_per_action *= mask

        top1 = torch.argmax(raw_score_per_action, 0)
        accuracy += (top1==torch.Tensor([label]).long().cuda()).float()


    return accuracy

def mask_TD_classification_softmax_loss(output_all, output_att, train_mask, attention_mask, labels, FLAGS):

    label_num = train_mask.shape[0]
    classifiers = output_all[0:label_num, :]
    output_att = output_att.view(labels.shape[0], -1, FLAGS.output_dim)

    labels=torch.zeros(labels.shape[0], label_num).scatter_(1, torch.LongTensor(labels).view(-1, 1), 1).cuda()
    losses = torch.zeros(1, label_num).cuda()

    unseen = []

    for i in range(labels.shape[0]):
        output_att_one = output_att[i, :, :]
        label = labels[i]
        attend_features = output_att_one[attention_mask.to_dense().long()].sum(dim=1)
        raw_score_per_action = classifiers * attend_features
        raw_score_per_action = torch.sum(raw_score_per_action, dim=1)
        mask = torch.Tensor(train_mask.astype(np.int32)).float().cuda()

        if (mask * label).sum() > 0.2:
            mask /= torch.mean(mask)
            raw_score_per_action *= mask
            raw_score_per_action = torch.nn.functional.log_softmax(raw_score_per_action)
            loss = -raw_score_per_action.view(1, -1) * label
            losses+=loss
        else:
            mask = torch.Tensor(train_mask.astype(np.int32)).float().cuda()
            raw_score_per_action *= (1 - mask)
            raw_score_per_action = nn.Softmax(dim=1)(raw_score_per_action.view(1, -1))
            unseen.append(raw_score_per_action)

    loss =  torch.mean(losses)

    if len(unseen) > 0:
        unseen_stack = torch.stack(unseen)
        softmax_unseen = unseen_stack.view(len(unseen), -1)
        _, V, _ = torch.svd(softmax_unseen)
        loss -= torch.mean(V) * FLAGS.bnm

    return loss


def mask_TDGZL_classification_softmax_loss(output_all, output_att, train_mask, attention_mask, labels, FLAGS, sim, att_feat):

    label_num = train_mask.shape[0]
    classifiers = output_all[0:label_num, :]
    output_att = output_att.view(labels.shape[0], -1, FLAGS.output_dim)
    att_feat = att_feat.view(labels.shape[0], -1, FLAGS.output_dim)
    labels=torch.zeros(labels.shape[0], label_num).scatter_(1, torch.LongTensor(labels).view(-1, 1), 1).cuda()
    losses = torch.zeros(1, label_num).cuda()

    unseen=[]

    for i in range(labels.shape[0]):
        output_att_one = output_att[i, :, :]
        label = labels[i]
        mask = torch.Tensor(train_mask.astype(np.int32)).float().cuda()


        if (mask * label).sum() > 0.2:
            output_att_one = (output_att + att_feat * sim)[i, :, :]
            attend_features = output_att_one[attention_mask.to_dense().long()].sum(dim=1)
            raw_score_per_action = classifiers * attend_features
            raw_score_per_action = torch.sum(raw_score_per_action, dim=1)
            mask /= torch.mean(mask)
            raw_score_per_action *= mask
            raw_score_per_action = torch.nn.functional.log_softmax(raw_score_per_action)
            loss = -raw_score_per_action.view(1, -1) * label
            losses+=loss
        else:
            output_att_one = (output_att + att_feat)[i, :, :]
            attend_features = output_att_one[attention_mask.to_dense().long()].sum(dim=1)
            raw_score_per_action = classifiers * attend_features
            raw_score_per_action = torch.sum(raw_score_per_action, dim=1)
            unseen.append(raw_score_per_action)

    loss =  torch.mean(losses)

    if len(unseen) > 0:
        unseen_stack = torch.stack(unseen)

        unseen_stack = nn.Softmax(dim=1)(unseen_stack)
        softmax_unseen = unseen_stack.view(len(unseen), -1)
        _, V, _ = torch.svd(softmax_unseen)
        loss -= torch.mean(V) * FLAGS.bnm

    return loss


def mask_TDGZL_classification_softmax_accuracy(output_all, output_att, train_mask, attention_mask, labels, FLAGS):

    label_num = train_mask.shape[0]
    classifiers = output_all[0:label_num, :]
    output_att = output_att.view(labels.shape[0], -1, FLAGS.output_dim)
 
    accuracy=0
    for i in range(labels.shape[0]):
        output_att_one = output_att[i, :, :]
        label = labels[i]
        attend_features = output_att_one[attention_mask.to_dense().long()].sum(dim=1)
        raw_score_per_action = classifiers * attend_features
        raw_score_per_action = torch.sum(raw_score_per_action, dim=1)
        mask = torch.Tensor(train_mask.astype(np.int32)).float().cuda()

        top1 = torch.argmax(raw_score_per_action, 0)
        accuracy += (top1==torch.Tensor([label]).long().cuda()).float()

    return accuracy

