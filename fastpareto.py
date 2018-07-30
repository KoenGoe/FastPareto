import numpy as np

def single_block_pareto(points):
    """Finds pareto optimal points (points of which no other point has some smaller cost without any larger other cost)  in the given points. 
    Works with O(#points^2 . #costs) memory and is thus not suitable for large amounts of points
    
    :param points: (#points, #costs)-shaped numpy array
    :returns: (#points,)-shaped boolean array numpy indicating the pareto points with True (1)"""
    
    #Determine for all point pairs if one dominates the other. 
    #Done by making two axes representing the points by adding singleton dimensions that will be broadcasted.
    dominates = np.logical_and(
                 np.all(points[:, np.newaxis,:] <= points[np.newaxis, :,:],2) #all costs better or equal
                ,np.any(points[:, np.newaxis,:] <  points[np.newaxis, :,:],2)) #at least one better
    #Determine equal points
    same = np.all(points[:, np.newaxis, :] == points[np.newaxis, :,:],2)

    is_dominated = np.any(dominates,0); #a point is dominated if it is dominated by any other
    is_dominated = np.logical_or(is_dominated, np.any(np.triu(same,1),1))#only one point is dominating if all are the same
    return np.logical_not(is_dominated)


def blocked_pareto(points, initial_blocksize=50):
    """Finds pareto optimal (points of which no other point has some smaller cost without any larger other cost) points in the given points. 
    Works with O(max(#initial_blocksize, #pareto_points)^2 . #costs) memory,
    
    :param points: (#points, #costs)-shaped numpy array
    :returns: (#points,)-shaped boolean array numpy indicating the pareto points with True (1)"""
   
    #processes in blocks to diminish quadratic memory usage issue of single block version
    #After processing in blocks, the remaining pareto candidates are recursively processed
    #This stops when the remaining pareto candidates are smaller fit in one block size and are processed at once
    
    
    blocksize = initial_blocksize #tradeoff code overhead vs overly large matrices overhead.

    nb_points = points.shape[0]
    
    is_not_dominated = np.zeros((nb_points,), dtype = np.bool)
    
    nb_blocks = nb_points//blocksize if (nb_points//blocksize)*blocksize == nb_points else (nb_points//blocksize + 1)
    
    #process blocks
    for i in range(nb_blocks):
        start = i*blocksize
        end_ = min(points.shape[0], (i+1)*blocksize)
        is_not_dominated[start:end_] = single_block_pareto(points[start:end_, :])
        

    #if still more than a blocksize candidates remain, recursively call this function.
    #if on top of that, the number of candidates did not drop, it is necessary to compare more points at once,
    #thus increase the blocksize in that case.
    #If less than a blocksize candidates remain: do a final single block pareto on the remaining ones
    #(unless the input was already smaller than blocksize, in which case done already)
    amount = np.sum(is_not_dominated)
    if amount > blocksize:
        if amount < nb_points:
            is_not_dominated[is_not_dominated] = blocked_pareto(points[is_not_dominated], blocksize)
        else: #nb of possible pareto points no longer dropping with current block size ==> increase
            is_not_dominated[is_not_dominated] = blocked_pareto(points[is_not_dominated], blocksize*2)
    else:
        if nb_points <= blocksize: #there was no actual blocking ==> done already
            return is_not_dominated
        else:#there was blocking, but the remaining ones can be done as one block
            is_not_dominated[is_not_dominated] = single_block_pareto(points[is_not_dominated])
    
    return is_not_dominated

def lex_pareto(points):
    """Finds pareto optimal points (points of which no other point has some smaller cost without any larger other cost) in the given points with 2 costs. 
    Vastly quicker than blocked_pareto, but limited to 2 costs.
    
    :param points: (#points,2)-shaped numpy array
    :returns: (#pareto_points,)-shaped numpy array with the indices of the pareto points"""

    #trivial cases for which below code fails
    if points.shape[0] == 0:
        return np.array([],dtype=np.uint8)
    if points.shape[0] == 1:
        return np.array([0],dtype=np.uint8)

    ind = np.lexsort(points.transpose())

    points = points[ind, :]
    
    #points are now sorted by 2nd column first and to 1st column second
    
    #As the 2nd column has ever increasing cost, only points that further decrease the 1st cost can be pareto optimal
    #Thus, we check the running minimum. Where it decreases, we have a pareto optimal point
    running_minimum = np.minimum.accumulate(points[:,0])
    
    indicators = np.zeros((running_minimum.shape[0],),dtype=np.bool)
    #first point is always optimal (due to the lexsort)
    indicators[0]=True
    #find where second to last cost reaches a new minimum
    indicators[1:] = running_minimum[1:]<running_minimum[:-1]
        
    return ind[indicators]

def pareto(points):
    """Finds pareto optimal points (points of which no other point has some smaller cost without any larger other cost) in the given points with a suitable algorithm. 
    
    :param points: (#points,#costs)-shaped numpy array
    :returns: (#pareto_points,)-shaped numpy array with the indices of the pareto points"""
    if points.shape[1] == 2:
        return lex_pareto(points)
    else:
        return blocked_pareto(points).nonzero()[0]
