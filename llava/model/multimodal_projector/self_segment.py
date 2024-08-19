import torch

def cal_depth_score(sim_scores):
    n = sim_scores.shape[0]
    depth_scores = torch.zeros(sim_scores.size(), dtype=sim_scores.dtype, device=sim_scores.device)
    # clip = min(max(n//10, 2), 5) # adopt clip to improve efficiency
    for i in range(n):
        lpeak = sim_scores[i]
        for li in range(i-1, -1, -1):
            if sim_scores[li] >= lpeak:
                lpeak = sim_scores[li]
            else:
                break
        rpeak = sim_scores[i]
        for ri in range(i+1, n):
            if sim_scores[ri] >= rpeak:
                rpeak = sim_scores[ri]
            else:
                break
        depth_scores[i] = lpeak + rpeak - 2 * sim_scores[i]
    return depth_scores


def segment(features, alpha=0.5, k=None):
    # input shape: t, d
    sim_scores = torch.cosine_similarity(features[:-1, :], features[1:, :])
    depth_scores = cal_depth_score(sim_scores)

    if k is not None:
        # select by top k
        boundaries = torch.topk(depth_scores, k).indices.sort()[0]
    else:
        # select by threshold (original)
        std, mean = torch.std_mean(depth_scores)
        thresh = mean + alpha * std
        condition = depth_scores > thresh
        boundaries = condition.nonzero().squeeze(-1)
        if len(boundaries) > 15: #TODO: limit max segments to prevent from OOM: 7 comes from RMT paper
            boundaries = torch.topk(depth_scores, 15).indices.sort()[0]
    
    boundaries = boundaries.tolist()
    
    # print("boudaries: ", boundaries)
    # print("features: ", features)
    
    if type(boundaries) == int or boundaries == [] or boundaries[-1] != features.shape[0]-1:
        boundaries.append(features.shape[0]-1)

    # # average segment
    # l = features.shape[0]
    # boundaires = list(range(l//(k+1)-1, l, l//(k+1)))
    
    # segments = []
    # index = 0
    # for bi in boundaries:
    #     segments.append(features[index: bi+1])
    #     index = bi + 1
    # if index < features.shape[0]: segments.append(features[index:])

    return boundaries

def cal_left_depth_score(sim_scores):
    n = sim_scores.shape[0]
    depth_scores = torch.zeros(sim_scores.size(), dtype=sim_scores.dtype, device=sim_scores.device)
    # clip = min(max(n//10, 2), 5) # adopt clip to improve efficiency
    for i in range(n):
        lpeak = sim_scores[i]
        for li in range(i-1, -1, -1):
            if sim_scores[li] >= lpeak:
                lpeak = sim_scores[li]
            else:
                break
        depth_scores[i] = lpeak - sim_scores[i]
    return depth_scores


def segment_left(features, alpha=0.5, k=None):
    # input shape: t, d
    sim_scores = torch.cosine_similarity(features[:-1, :], features[1:, :])
    depth_scores = cal_left_depth_score(sim_scores)
    
    # print(depth_scores)

    if k is not None:
        # select by top k
        boundaries = torch.topk(depth_scores, k).indices.sort()[0]
    else:
        # select by threshold (original)
        std, mean = torch.std_mean(depth_scores)
        thresh = mean + alpha * std
        condition = depth_scores > thresh
        boundaries = condition.nonzero().squeeze(-1)
        # if len(boundaries) > 15: #TODO: limit max segments to prevent from OOM: 7 comes from RMT paper
        #     boundaries = torch.topk(depth_scores, 15).indices.sort()[0]
    
    boundaries = boundaries.tolist()
    
    # print("boudaries: ", boundaries)
    # print("features: ", features)
    
    # if type(boundaries) == int or boundaries == [] or boundaries[-1] != features.shape[0]-1:
    #     boundaries.append(features.shape[0]-1)
    if type(boundaries) == int or boundaries == []:
        boundaries.append(features.shape[0]-1)
        
        
    # # average segment
    # l = features.shape[0]
    # boundaires = list(range(l//(k+1)-1, l, l//(k+1)))
    
    # segments = []
    # index = 0
    # for bi in boundaries:
    #     segments.append(features[index: bi+1])
    #     index = bi + 1
    # if index < features.shape[0]: segments.append(features[index:])

    return boundaries

