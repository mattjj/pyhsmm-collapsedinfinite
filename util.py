from __future__ import division
import numpy as np

def ndargmax(arr):
    return np.unravel_index(np.argmax(np.ravel(arr)),arr.shape)

def match_by_overlap(a,b):
    assert a.ndim == b.ndim == 1 and a.shape[0] == b.shape[0]
    ais, bjs = list(set(a)), list(set(b))
    scores = np.zeros((len(ais),len(bjs)))
    for i,ai in enumerate(ais):
        for j,bj in enumerate(bjs):
            scores[i,j] = np.dot(a==ai,b==bj)

    flip = len(bjs) > len(ais)

    if flip:
        ais, bjs = bjs, ais
        scores = scores.T

    matching = []
    while scores.size > 0:
        i,j = ndargmax(scores)
        matching.append((ais[i],bjs[j]))
        scores = np.delete(np.delete(scores,i,0),j,1)
        ais = np.delete(ais,i)
        bjs = np.delete(bjs,j)

    return matching if not flip else [(x,y) for y,x in matching]

def hamming_error(a,b):
    return (a!=b).sum()

def stateseq_hamming_error(sampledstates,truestates):
    sampledstates = np.array(sampledstates,ndmin=2).copy()

    errors = np.zeros(sampledstates.shape[0])
    for idx,s in enumerate(sampledstates):
        # match labels by maximum overlap
        matching = match_by_overlap(s,truestates)
        s2 = s.copy()
        for i,j in matching:
            s2[s==i] = j
        errors[idx] = hamming_error(s2,truestates)

    return errors if errors.shape[0] > 1 else errors[0]

def scoreatpercentile(data,per,axis):
    '''
    like the function in scipy.stats but with an axis argument, and works on
    arrays.
    '''
    a = np.sort(data,axis=axis)
    idx = per/100. * (data.shape[axis]-1)

    if (idx % 1 == 0):
        return a[[slice(None) if ii != axis else idx for ii in range(a.ndim)]]
    else:
        lowerweight = 1-(idx % 1)
        upperweight = (idx % 1)
        idx = int(np.floor(idx))
        return lowerweight * a[[slice(None) if ii != axis else idx for ii in range(a.ndim)]] \
                + upperweight * a[[slice(None) if ii != axis else idx+1 for ii in range(a.ndim)]]

def get_autocorr(chains):
    '''
    component-by-component
    '''
    chains = np.array(chains)
    results = np.zeros(chains.shape)
    for chainidx, chain in enumerate(chains):
        for idx in range(chain.shape[1]):
            temp = chain[:,idx] - chain[:,idx].mean(0)
            temp = np.correlate(temp,temp,'full')
            results[chainidx,:,idx] = temp[temp.shape[0]//2:]
    return results
