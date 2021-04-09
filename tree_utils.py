def is_list(v):
  return isinstance(v, list)

def is_list_of_type(v, t):
  return is_list(v) and all(isinstance(e, t) for e in v)

def flatten_indices(indices):
    # indices could be nested nested list, we convert them to nested list
    # if indices is not a list, then there is something wrong
    if not is_list(indices): raise ValueError('indices is not a list')
    # if indices is a list of integer, then we are done
    if is_list_of_type(indices, int): return indices
    # if indices is a list of list
    flat_indices = []
    for inds in indices:
        if is_list_of_type(inds, int): flat_indices.append(inds)
        else: flat_indices.extend(flatten_indices(inds))
    return flat_indices

def flatten_scores(scores):
    # scores could be nested list, we convert them to list of floats
    if isinstance(scores, float): return scores
    # instead of using deepflatten, i will write my own flatten function
    # because i am not sure how deepflatten iterates (depth-first or breadth-first)
    flat_scores = []
    for score in scores:
        if isinstance(score, float): flat_scores.append(score)
        else: flat_scores.extend(flatten_scores(score))
    return flat_scores
