import sys

import scipy.io
import torch
import numpy as np
# import time
import os


#######################################################################
# Evaluate
def evaluate(qf, ql, gf, gl, self_index):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # print(index)
    # index = index[0:2000]
    # good index
    # print("self_index", self_index)
    query_index = np.argwhere(gl == ql)
    # print("index", index)
    # print("query_index", query_index)
    camera_index = [self_index]
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # junk_index1 = np.argwhere(gl == -1)
    # junk_index2 = np.intersect1d(query_index, camera_index)
    # junk_index = np.append(junk_index2, junk_index1)  # .flatten())
    junk_index = [self_index]

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    # print("good_index", good_index.size)
    # print("index", index.size)
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    # print("mask", mask)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    # sys.exit()
    return ap, cmc


######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]
dirs = result['dirs'][0]
print()
print(dirs)
multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature.shape)
CMC = torch.IntTensor(len(gallery_label)).zero_()
# individual_CMC = torch.IntTensor(10, len(gallery_label)).zero_()
# individual_ap = torch.FloatTensor(10).zero_()
# count = [0] * 10
# print(individual_CMC.shape)
# print(individual_ap.shape)
# sys.exit()
ap = 0.0
# print(query_label)
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label, i)
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    # individual_CMC[query_label[i]-1] += CMC_tmp
    # individual_ap[query_label[i]-1] += ap_tmp
    # count[query_label[i]-1] += 1
    ap += ap_tmp
    # print(i, CMC_tmp[0])

CMC = CMC.float()
# individual_CMC = individual_CMC.float()
CMC = CMC / len(query_label)  # average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
# print()
# for i in range(10):
#     individual_CMC[i] = individual_CMC[i] / float(count[i])
#     print('index:%d——Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (i+1, individual_CMC[i][0], individual_CMC[i][4], individual_CMC[i][9], individual_ap[i] / count[i]))

# multiple-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
if multi:
    for i in range(len(query_label)):
        mquery_index1 = np.argwhere(mquery_label == query_label[i])
        mquery_index2 = np.argwhere(mquery_cam == query_cam[i])
        mquery_index = np.intersect1d(mquery_index1, mquery_index2)
        mq = torch.mean(mquery_feature[mquery_index, :], dim=0)
        ap_tmp, CMC_tmp = evaluate(mq, query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        # print(i, CMC_tmp[0])
    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))