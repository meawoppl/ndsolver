from numpy import arange, array, c_, dot, ndenumerate, newaxis, mgrid, zeros, zeros_like

# Return an iterator of tuples of points in the grid
def iter_grid(grid, strip_non_dof = True, my_min_dof=0, my_max_dof=-1):
    if my_max_dof == -1: my_max_dof = array(grid.shape).prod
    slice_count = len(grid.shape)
    slice_tup = ()
    
    for s in range(slice_count):
        slice_tup += tuple([slice(0,grid.shape[s])])

    trick = []
    the_grid = mgrid[ slice_tup ]
    for indx in range(slice_count):
        trick += [the_grid[indx].flat]

    # (grid.size X grix.ndim) array
    iterable_array = array(c_[tuple(trick)])
    
    # Strip non dof points from the list
    if strip_non_dof:
        iterable_array = iterable_array[grid.flatten() >= 0]

    iterable_tuples = [ tuple(point) for point in iterable_array ]

    return iter(iterable_tuples[my_min_dof:my_max_dof + 1])

def full_iter_grid(grid):
    shape = grid.shape
    upper = array(shape).prod()
    
    list_of_points = []
    for x in range(upper):
        list_of_points.append( num_to_point(x, shape) )

    return list_of_points
    
def pruned_iterator(grid, my_min_dof, my_max_dof):
    for pt, val in ndenumerate(grid):
        if val < my_min_dof: continue
        if val > my_max_dof: raise StopIteration()
        yield pt

# def pruned_iterator(grid, my_min_dof, my_max_dof):
#     shape = grid.shape
#     upper = array(shape).prod()

#     for x in range(upper):
#         point = num_to_point(x, shape)
#         if grid[point] < my_min_dof: continue
#         if grid[point] > my_max_dof: raise StopIteration()
#         yield point



def num_to_point(num, shape):
    rev_point = []
    for axis_dim in shape[::-1]:
        mod = num % axis_dim
        num = num / axis_dim    # int
        rev_point.append(mod)
        
    return tuple(rev_point[::-1])

def point_to_num(point, shape):
    return dot(([1] + list(array(shape).cumprod()[:-1]))[::-1], array(point))

    
        
    

def perturb(point):
    ndim = len(point)
    index_array = array(point)

    for dim in range( ndim ):
        poke = zeros_like(index_array)
        for inc in [-1,1]:
            poke[dim] = inc
            yield tuple(index_array + poke)

def roller(point, shape):
    parray = array(point)
    sarray = array(shape)
    return tuple( parray % sarray )

    
if __name__ == "__main__":
    a = arange(25).reshape((5,5))
    for x in iter_grid(a):
        print x, "-", a[x]

    for point in perturb((3,3,3)):
        print point
        

    
    
    
