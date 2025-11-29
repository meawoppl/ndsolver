from numpy import array, zeros, roll, \
                  logical_and, logical_or, cumsum, \
                  logical_not, int32, int64, ones, \
                  ndenumerate, zeros_like

from .equation import Equation
from . import ndimed

def ndcodex(dim):
    if dim in codex:
        return codex[dim]
    
    shape  = tuple([3]) * dim
    temp   = ones(shape, dtype=int64)
    
    center = tuple([1]) * dim
    temp[center] = 0
    temp = temp.cumsum().reshape(shape)
    temp = temp - 1
    temp[center] = 0

    temp = 2**temp
    temp[center] = 0

    codex[dim] = temp
    
    return temp

# Pre-populate the codex . . . why not . . .
global codex
codex = {}
for x in [2,3]:
    codex[x] = ndcodex(x)


def p_dof(domain):
    # Calculate pressure dof numbers for each cell
    p_dof = (((1-domain).cumsum().reshape(domain.shape) * (1-domain))) - 1
    return p_dof.astype(int32)
  
def velocity_dof(domain, ax):
    # Calculate velocity dof numbers for each cell
    rm = roll( domain, 1, axis=ax )
    type_3 = logical_and( domain, rm )
    type_2 = logical_or(  domain, rm )
    
    dof = cumsum( logical_not( logical_or( type_3, type_2 ) ) ).reshape( domain.shape ) - 1
    # Do logic to figure out type 2 and 3
    dof[type_2 == 1] = -2
    dof[type_3 == 1] = -3

    return dof.astype(int64)

def velocity_dofs(domain):
    # Number of dimensions
    ndim = domain.ndim

    # List to put DOF grids in
    v_dofs = []

    # Populate above list
    for dim in range(ndim):
        v_dofs.append(velocity_dof(domain, dim))

    return v_dofs

def solid_adjacent(solid):
    ndim = solid.ndim

    solid_sum = zeros_like(solid)

    for dim in range(ndim):
        solid_sum += roll(solid,  1, axis=dim)
        solid_sum += roll(solid, -1, axis=dim)

    return solid_sum


def momentum_eq(dof_grid, point):
    # Correct for pbc's
    shape = dof_grid.shape
    ndim = len(shape)

    point = ndimed.roller(point, shape)
    
    # If solid . . . there is no momentum equation
    if dof_grid[ point ] < 0:
        return Equation()

    # DOF at the center point
    center_dof_num = dof_grid[ point ]

    m_eq = Equation()
    m_eq[ center_dof_num ]  = -2 * ndim

    # Move one step in each direction
    for p in ndimed.perturb( point ):
        # Wrap where necessary
        p = ndimed.roller(p, shape)

        # Get the current dof number
        current_dof = dof_grid[p]

        # If moving in one direction is a dof
        if   current_dof >=  0:
            # This could glitch if the domain is small enough
            # For the laplacian est. to touch itself (3x3 or smaller . . .)
            m_eq[ current_dof ] = 1

        # If it is solid
        elif current_dof == -2:
            continue

        # If it is completely enclosed solid
        elif current_dof == -3:
            m_eq[ center_dof_num ] -= 1
            
    return  m_eq

def comp_mom_eq(grids, point):
    # Roll coordinate system if necessary
    s = grids[0].shape
    point = ndimed.roller(point, s)
    ndim = len(grids)

    # Stupid Check!
    for grid in grids:
        if grid.shape != s or len(grid.shape) != len(point):
            print("Shape mismatch!")
            raise TypeError

    # Initialize the equations
    mom_eqs = []
    for x in range(ndim):
        mom_eqs.append(Equation())

    # Iterate through the dimensions
    for dim in range(ndim):
        # Make a point
        # Shifted by 1
        # In the dimension we are working in
        shifted_point = array(point)
        shifted_point[dim] += 1
        shifted_point = tuple(shifted_point)
        shifted_point = ndimed.roller(shifted_point, s)

        # Think M(right) - M(left) (pg.84 in Dr. E. Thesis)
        # TODO: implement __iadd__ in Equation
        mom_eqs[dim] = mom_eqs[dim] + momentum_eq( grids[dim], shifted_point )
        mom_eqs[dim] = mom_eqs[dim] - momentum_eq( grids[dim], point )

    # Not obvious but these are the U, V, (W) components . . .
    return mom_eqs


def div_eq(grids, point):
    # Because any time this is called on a solid centered cell
    # it will return all zeros, we don't worry about that
    s = grids[0].shape
    ndim = len(grids)

    # Stupid Check!
    for grid in grids:
        if grid.shape != s or len(grid.shape) != len(point):
            print("Shape mismatch!")
            raise TypeError

    # PCB correction
    point = ndimed.roller(point, s)

    # Initialize the equations
    div_eqs = []
    for x in range( ndim ):
        div_eqs.append( Equation() )

    #iterate through the dimensions, grids, and corresponding momentum equations
    for dim in range(ndim):
        # Shifted by 1 in the n-th axis
        shifted_point = array(point)
        shifted_point[dim] += 1
        shifted_point = tuple(shifted_point)
        shifted_point = ndimed.roller(shifted_point, s)

        # Inflow
        dof = grids[dim][point]
        if dof >= 0:
            div_eqs[dim][dof] = -1

        # outflow
        dof = grids[dim][shifted_point]
        if dof >= 0:
            div_eqs[dim][dof] = 1

    return div_eqs


def lap_div(pdof, grids, point):
    s = grids[0].shape
    ndim = len(grids)

    # Center div equation
    center_eq = div_eq( grids, point )

    # Initialize the equations
    div_lap_eqs = []
    for x in range( ndim ):
        div_lap_eqs.append( Equation() )

    for pp in ndimed.perturb(point):
        # PBC correction
        pp = ndimed.roller(pp, s)

        # If there is a single -3 dof, it must be a solid square
        if pdof[pp] < 0:
            continue
        
        pert_eq = div_eq(grids, pp)

        for dim in range(ndim):
            div_lap_eqs[dim] = div_lap_eqs[dim] + ( pert_eq[dim] - center_eq[dim] )

    return div_lap_eqs

def old_config_code(solid, point):
    # Number of dimensions
    ndim = len(solid.shape)

    # Iterate through and bring the point we want to (1,1,1 . . .)
    for dim in range(ndim):
        solid = roll(solid, -point[dim] + 1, axis=dim )

    # Generate the number of slices for the number of dimensions
    slices = [slice(0,3)] * ndim

    # Multiply that point by the codex of correct dimensions
    key = solid[slices] * codex[ndim]
    
    return key.sum() # The Sum is the configuration code


def config_code(solid, point):
    # Number of dimensions, DONE THIS WAY FOR HDF5 COMPLIANCE
    ndim = len(solid.shape)

    # Were gonna populate a solid section into this
    sect = zeros(tuple([3]) * ndim)

    # Populate it
    for pt, val in ndenumerate(sect):
        arr_pt = tuple( (array(pt) + array(point) - 1) % array(solid.shape) ) 
        sect[pt] = solid[arr_pt]

    # Multiply that section by the codex corresponding to its dimensionality
    key = sect * codex[ndim]
    
    return key.sum() # The Sum is the configuration code

def all_configs(solid):
    config_list = []

    # Iterate over the grid
    for point in ndimed.iter_grid(solid):
        # Add the points config to the list
        config_list.append( config_code(solid, point) )

    # Return a unique set of numbers
    return set(config_list)


def make_config(cfg):
    bin_list = [ 1 if (cfg & (1 << i)) else 0 for i in range(8) ]
    bin_list.insert(4,0)
    return array(bin_list).reshape((3,3))

def make_config_test(cfg):
    three = make_config(cfg)
    five = zeros((5,5))
    five[1:4,1:4] = three
    return five

def make_safe_config_test(cfg):
    three = make_config(cfg)

    if three[0,1] == 1 and three[1,0] == 1 and three[2,1] == 1 and three[1,2] == 1:
        three[1,1] = 1

    five = zeros((7,7))
    five[2:5,2:5] = three
    return five.astype(int32)


def s_term(pdof, grids, point):
    '''This generates n-dictionaries of equations corresponding to
    the dof's of the n'th dimension'''

    # Not valid for solid squares (should return nothing anyway?)
    # for grid in grids:
    #     if grid[point] == -3:
    #         print "Solid square!!!"
    #         raise ValueError

    ndim = len(grids)
    
    mom_eqs = comp_mom_eq( grids, point )
    div_eqs = lap_div( pdof, grids, point )

    terms = [None] * ndim
    for dim in range(ndim):
        terms[dim] = mom_eqs[dim] - div_eqs[dim]
    
    return terms


# def s_term_matricies(pdof, grids, zero_rows = [0]):
#     ndim = pdof.ndim

#     # Backout solid so we can avoid doing work where necessary
#     solid = where(pdof>=0, 0,1)

#     # Get degrees of freedom for the matrices
#     pdofs = pdof.max() + 1
#     dof_counts = []
#     for dim in range(ndim):
#         dof_counts.append( grids[dim].max() + 1 )

#     # Make the matrix's
#     # Yes I do realize that the spellingis different
#     # it inst repeatable for me so freiking replace-string if you care
#     matrixs = []
#     for dim in range(ndim):
#         matrixs.append( sparse.lil_matrix( (dof_counts[dim], pdofs) ) )
    
#     # Iterate over the grid
#     for point in ndimed.iter_grid(pdof):
#         # Find the DOF for the current pressure cell
#         pd = pdof[point]

#         # Skip non-pressure dof's
#         if pd < 0:
#             continue

#         # This is here to prevent writing and equation for a pinned cell
#         if pd in zero_rows:
#             continue

#         # If completely liquid . . . lapace equation . . . no source
#         cfg_code = config_code(solid, point)
#         if cfg_code  == 0:
#             continue

#         # This will return a list of n equations
#         point_equations = s_term(pdof, grids, point)

#         # For each dimension (equation) populate the corresponding matrix row
#         for dim, eq in enumerate(point_equations):
#             for dofn, coeff in eq.iteritems():
#                 matrixs[dim][ dofn, pd ] = coeff

#     for matrix in matrixs:
#         matrix = matrix.tocsc()

#     return matrixs

# def div_matrix(p_dof_grid, v_dof_grid, axis):
#     # shape and locus correction
#     s = p_dof_grid.shape

#     # Pressure degree of freedom count
#     pre_dof = p_dof_grid.max() + 1
#     vel_dof = v_dof_grid.max() + 1

#     # Matrix definition
#     div_mat = sparse.lil_matrix( ( pre_dof, vel_dof) )

#     rolled_v_dof_grid = roll(v_dof_grid, -1, axis=axis)
        
#     # Iterate over the grid
#     for point in ndimed.iter_grid(p_dof_grid):
#         # P Degree of freedom
#         pd = p_dof_grid[point]
#         if pd < 0:
#             continue
        
#         # Neg is flowing in
#         dof = v_dof_grid[point]
#         if dof >= 0:
#             div_mat[pd, dof] = -1
        
#         # Positive is flowing out
#         dof = rolled_v_dof_grid[point]
#         if dof >= 0:
#             div_mat[pd, dof] = 1

#     return div_mat.tocsr()

# def div_matrices(p_dof_grid, v_dof_grids):
#     ndim = p_dof_grid.ndim

#     div_mats = []

#     for dim in range(ndim):
#         div_mats.append( div_matrix( p_dof_grid, v_dof_grids[dim], dim ) )

#     return div_mats


# # Gradient of pressure
# def vel_rhs_matrix(p_dof_grid, vel_dof_grid, axis):
#     # shape and locus correction
#     s = p_dof_grid.shape

#     pdofs = p_dof_grid.max() + 1
#     vel_dofs = vel_dof_grid.max() + 1

#     # Eq definition
#     vel_rhs_mat = sparse.lil_matrix( ( pdofs, vel_dofs ) )

#     rolled_p = roll(p_dof_grid, 1, axis=axis)

#     for point in ndimed.iter_grid(p_dof_grid):
#         vel_dof = vel_dof_grid[point]

#         if vel_dof < 0:
#             continue 
       
#         p1 = p_dof_grid[point]
#         if p1 >= 0:
#             vel_rhs_mat[p1, vel_dof] = 1

#         p2 = rolled_p[point]
#         if p2 >= 0:
#             vel_rhs_mat[p2, vel_dof] = -1

#     return vel_rhs_mat.tocsc()

# def vel_rhs_matrices(p_dof_grid, vel_dof_grids):
#     ndim = p_dof_grid.ndim
#     matrices = []
    
#     for dim in range(ndim):
#         matrices.append( vel_rhs_matrix(p_dof_grid, vel_dof_grids[dim], dim) )

#     return matrices


if __name__ == "__main__":
    from .eq_solver import div_matrices
    sf = zeros((3,3))
    pdof  = p_dof(sf)
    grids = velocity_dofs(sf)

    for x, grid in enumerate(div_matrices(pdof, grids)):
        print(x, grids[x])
        print(grid.todense())
