import time
import datetime
import os
import torch
import torch.optim as optim


from utils import *
from models import GCN_dense_mse_2s

import argparse
import pickle as pkl
import networkx as nx


parser = argparse.ArgumentParser(description='Transfer Learning')
parser.add_argument('--dataset', type=str, nargs='?', default='fcvid', help='Dataset string.') #ucf101, hmdb51, olympic_sports
parser.add_argument('--datadir', type=str, nargs='?', default='fcvid', help='Dataset string.') #ucf101, hmdb51, olympic_sports
parser.add_argument('--w2v_type', type=str, nargs='?', default='Yahoo_100m', help='Word2Vec Type.')# Google_News_w2v, Yahoo_100m
parser.add_argument('--w2v_dim', type=int, nargs='?', default=500, help='dimension of the word2vec.')
parser.add_argument('--time_interval', type=int, nargs='?', default=2, help='Number of time interval for a shot.')
parser.add_argument('--ini_seg_num', type=int, nargs='?', default=32, help='Number of initial number of segments.')
parser.add_argument('--num_class', type=int, nargs='?', default=1320, help='Number of chossen imageNet classes.')# 2343, 1397, 373 847 163 
parser.add_argument('--output_dim', type=int, nargs='?', default=300, help='Number of units in the last layer (output the classifier).')
parser.add_argument('--split_ind', type=int, nargs='?', default=0, help='current zero-shot split.')
parser.add_argument('--topK', type=int, nargs='?', default=30, help='we choose topK objects for each segment.')
parser.add_argument('--label_num', type=int, nargs='?', default=101, help='number of actions.')
parser.add_argument('--batch_size', type=int, nargs='?', default=48, help='batch size.')
parser.add_argument('--result_save_path', type=str, nargs='?', default='./results/', help='results save dir')
parser.add_argument('--mode', type=str, nargs='?', default='GCN+INIT+SIM', help='Ablation')
parser.add_argument('--setting', type=str, nargs='?', default='inductive', help='inductive, transductive, generalized_transductive')

parser.add_argument('--bnm', type=float, nargs='?', default=50, help='BNM')


parser.add_argument('--model', type=str, nargs='?', default='dense', help='Model string.')
parser.add_argument('--learning_rate', type=float, nargs='?', default=0.0001, help='Initial learning rate.')
parser.add_argument('--save_path', type=str, nargs='?', default='./output_models/', help='save dir')
parser.add_argument('--epochs', type=int, nargs='?', default=5, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, nargs='?', default=2048, help='Number of units in hidden layer 1.')
parser.add_argument('--dropout', type=float, nargs='?', default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, nargs='?', default=5e-4, help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--gpu', type=str, nargs='?', default='0', help='gpu id')
FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


now_time = datetime.datetime.now().strftime('%Y-%m-%d-%T')

# Load data
data_path = 'data_'+FLAGS.datadir
all_att_inds, all_att_scores = get_imageNet_input_data(FLAGS.dataset, FLAGS.time_interval, FLAGS.ini_seg_num, FLAGS.num_class, root=data_path)
adj_all, adj_att, features, y_train, y_val, idx_train, idx_val, train_mask, test_mask, lookup_table = \
        load_data_action_zero_shot(FLAGS.dataset, FLAGS.w2v_type, FLAGS.split_ind, data_path = data_path, FLAGS=FLAGS)

label_num = len(train_mask)
FLAGS.label_num = label_num
if FLAGS.w2v_type == 'Yahoo_100m':
    FLAGS.w2v_dim = 300

features_all = features
features_att = features[label_num:,:]

if FLAGS.model == 'dense':

    support_all = [preprocess_adj(adj_all)]
    support_att = [preprocess_adj(adj_att)]

    support_att_batch = [preprocess_adj(adj_att)]
    for s in range(len(support_att_batch)):
        support_att_batch[s] = list(support_att_batch[s])
        for i in range(FLAGS.batch_size-1):
            support_att_batch[s][0] = np.concatenate((support_att_batch[s][0], support_att[s][0]+(i+1)*FLAGS.num_class))
            support_att_batch[s][1] = np.concatenate((support_att_batch[s][1], support_att[s][1]))
        support_att_batch[s][2] = tuple(np.array(support_att[s][2])*FLAGS.batch_size*1)
    num_supports = len(support_att)
    model_func = GCN_dense_mse_2s

    support_att_batch_test = [preprocess_adj(adj_att)]
    for s in range(len(support_att_batch_test)):
        support_att_batch_test[s] = list(support_att_batch_test[s])
        for i in range(7):
            support_att_batch_test[s][0] = np.concatenate((support_att_batch_test[s][0], support_att[s][0]+(i+1)*FLAGS.num_class))
            support_att_batch_test[s][1] = np.concatenate((support_att_batch_test[s][1], support_att[s][1]))
        support_att_batch_test[s][2] = tuple(np.array(support_att[s][2])*8*1)
 
seg_number = int(FLAGS.ini_seg_num / FLAGS.time_interval)
topK = FLAGS.topK
print('topK = %d' %topK)
tmp_row_index = np.arange(0, seg_number)
tmp_row_index = np.expand_dims(tmp_row_index,1)
tmp_row_index = np.expand_dims(tmp_row_index, 0)
tmp_row_index = np.tile(tmp_row_index,(FLAGS.batch_size,1,topK))
tmp_batch_index = np.arange(0, FLAGS.batch_size)
tmp_batch_index = np.expand_dims(tmp_batch_index, 1)
tmp_batch_index = np.expand_dims(tmp_batch_index, 1)
tmp_batch_index = np.tile(tmp_batch_index, (1, seg_number, topK))
input_dict = {}

tmp_row_index_test = np.arange(0, seg_number)
tmp_row_index_test = np.expand_dims(tmp_row_index_test,1)
tmp_row_index_test = np.expand_dims(tmp_row_index_test, 0)
tmp_row_index_test = np.tile(tmp_row_index_test,(8,1,topK))
tmp_batch_index_test = np.arange(0, 8)
tmp_batch_index_test = np.expand_dims(tmp_batch_index_test, 1)
tmp_batch_index_test = np.expand_dims(tmp_batch_index_test, 1)
tmp_batch_index_test = np.tile(tmp_batch_index_test, (1, seg_number, topK))
 
lookup_table_act_att = torch.sparse.FloatTensor(torch.LongTensor(lookup_table[0]).t(), torch.FloatTensor(lookup_table[1]), torch.Size(lookup_table[2]))
model = model_func(input_dict, lookup_table_act_att, features.shape[1], FLAGS)
model=model.cuda()

savepath = FLAGS.save_path
exp_name = os.path.basename(FLAGS.dataset)
savepath = os.path.join(savepath, exp_name)
if not os.path.exists(savepath):
    os.makedirs(savepath)
    print('!!! Make directory %s' % savepath)
else:
    print('### save to: %s' % savepath)

result_save_path = FLAGS.result_save_path + FLAGS.dataset + '/'
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)
    print('!!! Make directory %s' % result_save_path)
else:
    print('### save to: %s' % result_save_path)
result_file_name = result_save_path + FLAGS.dataset + '_MM_' + FLAGS.w2v_type + '_' \
                    + str(FLAGS.time_interval) + '_' + str(FLAGS.ini_seg_num) \
                   + '_' + str(FLAGS.num_class) + '_' + 'True'  \
                   + str(FLAGS.learning_rate *100000)+ '_' + str(FLAGS.hidden1) + '_' \
                   + str(FLAGS.output_dim) + '_' \
                   + str(FLAGS.split_ind) + '_' \
                   + '_' + str(FLAGS.batch_size) + '.txt'


# Train model
now_lr = FLAGS.learning_rate
y_train = np.array(y_train)
idx_train = np.array(idx_train)
y_val = np.array(y_val)
idx_val = np.array(idx_val)
all_att_inds = np.array(all_att_inds)
all_att_scores = np.array(all_att_scores)
optimizer = optim.Adam([{"params":model.parameters(), "lr":now_lr}], lr=now_lr,weight_decay=FLAGS.weight_decay)


if FLAGS.mode == "INIT":
    FLAGS.epochs = 1

if FLAGS.setting == "generalized_transductive":
    if os.path.exists(data_path+'/permutation_'+str(FLAGS.split_ind)+'.npy'):
        IND = np.load(data_path+'/permutation_'+str(FLAGS.split_ind)+'.npy')
    else:
        IND = np.random.permutation(len(y_train))
        np.save(data_path+'/permutation_'+str(FLAGS.split_ind)+'.npy', IND)
    For_test = IND[:int(len(IND)*0.2)]
    For_train = IND[int(len(IND)*0.2):]
    y_val_tmp = np.concatenate([y_train[For_test].copy(), y_val])
    y_train[For_test] = y_val[:For_test.shape[0]].copy()
    y_train = np.concatenate([y_train,y_val])
    y_val = y_val_tmp
    
    idx_val_tmp = np.concatenate([idx_train[For_test],idx_val])
    idx_train = np.concatenate([idx_train,idx_val])
    idx_val = idx_val_tmp


for epoch in range(FLAGS.epochs):
    if not FLAGS.mode == "INIT":
        model.train()
        count = 0
        rand_inds = np.random.permutation(len(y_train))
        rand_inds = rand_inds[:int(len(rand_inds)/FLAGS.batch_size)*FLAGS.batch_size]
        rand_inds = np.reshape(rand_inds,[-1, FLAGS.batch_size])

        for inds in rand_inds[:int(len(rand_inds))]:
            # Construct feed dictionary
            optimizer.zero_grad()
            label = y_train[inds]
            video_idx = idx_train[inds]
            features_att_this_sample = np.zeros([FLAGS.batch_size,seg_number,FLAGS.num_class])
            att_ind = all_att_inds[video_idx]
            att_score = all_att_scores[video_idx]
            att_ind = att_ind[:,:, :topK]
            att_score = att_score[:,:, :topK]
            features_att_this_sample[tmp_batch_index,tmp_row_index, att_ind] = att_score
            features_att_this_sample = np.expand_dims(features_att_this_sample, 2)

            input_dict['support_all']=support_all
            input_dict['support_att']=support_att_batch
            input_dict['support_att1']=support_att_batch
            input_dict['features_all']=features_all
            input_dict['features_att']=features_att_this_sample
            input_dict['labels']=label
            input_dict['train_mask']=train_mask
            input_dict['dropout']=0
            input_dict['learning_rate']=now_lr
            input_dict['label_num']=label_num


            loss, accuracy = model(input_dict)
            loss.backward()
            optimizer.step()

            if count % 1 == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "sample_batch:", '%04d' % (count + 1), "train_loss=", "{:.5f}".format(loss.item()), flush=True)
                count += 1
    test_accuracy = 0
    test_inds = np.arange(len(y_val))
    test_inds = test_inds[:int(len(test_inds) / 8) * 8]
    test_inds = np.reshape(test_inds, [-1, 8])
    count_test = 0
    model.eval()
    with torch.no_grad():
        for inds in test_inds:
        #     # Construct feed dictionary
            label = y_val[inds]
            video_idx = idx_val[inds]
            features_att_this_sample = np.zeros([8, seg_number, FLAGS.num_class])
            att_ind = all_att_inds[video_idx]
            att_score = all_att_scores[video_idx]
            att_ind = att_ind[:, :, :topK]
            att_score = att_score[:, :, :topK]
            features_att_this_sample[tmp_batch_index_test, tmp_row_index_test, att_ind] = att_score
            features_att_this_sample = np.expand_dims(features_att_this_sample, 2)

            input_dict['support_all']=support_all
            input_dict['support_att']=support_att_batch_test
            input_dict['support_att1']=support_att_batch_test
            input_dict['features_all']=features_all
            input_dict['features_att']=features_att_this_sample
            input_dict['labels']=label
            input_dict['train_mask']=test_mask
            input_dict['dropout']=0
            input_dict['learning_rate']=now_lr
            input_dict['label_num']=label_num
            _, accuracy = model(input_dict)
            test_accuracy += accuracy.item()
            count_test += 1
            if count_test % 10 == 0:
                print('%04d baches are processed for testing' % (count_test ))
        test_accuracy /= (len(test_inds)*8)
        print("Epoch:", '%04d' % (epoch + 1),
              "accuracy=", "{:.5f}".format(float(test_accuracy)), flush=True
              )
    with open(result_file_name, 'a') as f:
        f.write(str(test_accuracy)+'\n')


print("Optimization Finished!")

