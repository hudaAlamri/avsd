import torch
import os

def get_gt_ranks(ranks, ans_ind):
    ans_ind = ans_ind.view(-1)
    gt_ranks = torch.LongTensor(ans_ind.size(0))
    for i in range(ans_ind.size(0)):
        gt_ranks[i] = int(ranks[i, ans_ind[i]])
    return gt_ranks

def process_ranks(ranks, save_path, epoch):
    num_ques = ranks.size(0)
    num_opts = 100

    # none of the values should be 0, there is gt in options
    if torch.sum(ranks.le(0)) > 0:
        num_zero = torch.sum(ranks.le(0))
        print("Warning: some of ranks are zero: {}".format(num_zero))
        ranks = ranks[ranks.gt(0)]

    # rank should not exceed the number of options
    if torch.sum(ranks.ge(num_opts + 1)) > 0:
        num_ge = torch.sum(ranks.ge(num_opts + 1))
        print("Warning: some of ranks > 100: {}".format(num_ge))
        ranks = ranks[ranks.le(num_opts + 1)]
  

    metrics = {}

    ranks = ranks.float()
    num_r1 = float(torch.sum(torch.le(ranks, 1)))
    num_r5 = float(torch.sum(torch.le(ranks, 5)))
    num_r10 = float(torch.sum(torch.le(ranks, 10)))
   
    r1 = num_r1 / num_ques
    r5 = num_r5 / num_ques
    r10 = num_r10 / num_ques
    meanR = torch.mean(ranks)
    meanRR = torch.mean(ranks.reciprocal())

    metrics['r1'] = r1
    metrics['r5'] = r5
    metrics['r10'] = r10
    metrics['meanR'] = meanR
    metrics['meanRR'] = meanRR

    with open(os.path.join(save_path, "ranks_resutls_valSet_Feb11.txt"), "a+") as f:
        f.write("Epoch: {}".format(epoch))
        f.write("\tNo. questions: {}\n".format(num_ques))
        f.write("\tr@1: {}\n".format(num_r1 / num_ques))
        f.write("\tr@5: {}\n".format(num_r5 / num_ques))
        f.write("\tr@10: {}\n".format(num_r10 / num_ques))
        f.write("\tmeanR: {}\n".format(torch.mean(ranks)))
        f.write("\tmeanRR: {}\n".format(torch.mean(ranks.reciprocal())))
        f.write('\n')
    
    f.close()
    
    print("\tNo. questions: {}".format(num_ques))
    print("\tr@1: {}".format(num_r1 / num_ques))
    print("\tr@5: {}".format(num_r5 / num_ques))
    print("\tr@10: {}".format(num_r10 / num_ques))
    print("\tmeanR: {}".format(torch.mean(ranks)))
    print("\tmeanRR: {}".format(torch.mean(ranks.reciprocal())))

    return metrics 

def scores_to_ranks(scores):
    # sort in descending order - largest score gets highest rank
    sorted_ranks, ranked_idx = scores.sort(1, descending=True)

    # convert from ranked_idx to ranks
    ranks = ranked_idx.clone().fill_(0)
    for i in range(ranked_idx.size(0)):
        for j in range(100):
            ranks[i][ranked_idx[i][j]] = j
    ranks += 1
    return ranks
