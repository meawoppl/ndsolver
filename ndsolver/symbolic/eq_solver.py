import numpy as np
from numpy import array, zeros, roll, logical_and, logical_or, cumsum, logical_not, int64, ones
from scipy import sparse

from .Equation import Equation

global codex
codex = array([[  0,  1,  2 ],
               [  3,  0,  4 ],
               [  5,  6,  7 ]])

codex = 2**codex
codex[1,1] = 0

solved_configs = set([])
solved_u = {}
solved_v = {}

udlr = array([[0,1,0],
              [1,0,1],
              [0,1,0]])



def p_dof(domain):
    # Calculate pressure dof numbers for each cell
    p_dof = (((1-domain).cumsum().reshape(domain.shape) * (1-domain))) - 1
    return p_dof.astype(int64)
  
def velocity_dof(domain, ax):
    # Calculate velocity dof numbers forr each cell
    rm = roll( domain, 1, axis=ax )
    type_3 = logical_and( domain, rm )
    type_2 = logical_or(  domain, rm )
    
    dof = cumsum( logical_not( logical_or( type_3, type_2 ) ) ).reshape( domain.shape ) - 1
    # Do logic to figure out type 2 and 3
    dof[type_2 == 1] = -2
    dof[type_3 == 1] = -3

    return dof.astype(int64)



def momentum_eq(dof_grid, x, y):
    # Correct for pbc's
    s = dof_grid.shape
    x, y = x % s[0], y % s[1]
    
    # If solid . . . there is no momentum equation
    if dof_grid[x,y] < 0:
        return Equation()

    m_eq = Equation()

    center_dof_num = dof_grid[ x, y ]
    m_eq[ center_dof_num ]  = -4

    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    for d in directions:
        tx, ty = (x + d[0]) % s[0], (y + d[1]) % s[1]
        current_dof = dof_grid[tx, ty]
        
        if   current_dof == -2:
            pass
        elif current_dof == -3:
            m_eq[ center_dof_num ] -= 1
        else:
            if current_dof in m_eq:
                m_eq[ current_dof ] += 1
            else:
                m_eq[ current_dof ] = 1

    return  m_eq

def comp_mom_eq(u_dof_grid, v_dof_grid, x, y):
    # Roll coord system if necessary
    if u_dof_grid.shape != v_dof_grid.shape:
        print("Shape mismatch!")
        raise TypeError
    s = u_dof_grid.shape

    x, y = x % s[0], y % s[1]

    # Stupid check!
    if s != u_dof_grid.shape or s != v_dof_grid.shape:
        raise ValueError("Shapes of all DOF grids must match!")

    # six momentum equations here
    m_left    = momentum_eq( u_dof_grid, x,   y   )
    m_bottom  = momentum_eq( v_dof_grid, x,   y   )

    # Shift cells in appropiate direction and repeat
    m_right   = momentum_eq( u_dof_grid, x+1, y   )
    m_top     = momentum_eq( v_dof_grid, x,   y+1 )

    print("ML", m_left)
    print("MR", m_right)
    print("MT", m_top)
    print("MB", m_bottom)

    # Not odbvious but these are the U, Vcomponents . . .
    return (m_right-m_left), (m_top-m_bottom)


def div_eq(u_dof_grid, v_dof_grid, x, y):
    # Because any time this is called on a soid centered cell
    # it will retrun all zeros, we dont worry about that

    # shape and locus correction
    s = u_dof_grid.shape
    x, y = x % s[0], y % s[1]

    # Eq definition
    ut = Equation()
    vt = Equation()

    # U Degrees of freedom
    ud = u_dof_grid[x, y]
    if ud >= 0:
        ut[ ud ] = -1
    ud = u_dof_grid[(x+1)%s[0], y]
    if ud >= 0:
        ut[ ud ] = 1

    # V Degrees of freedom
    vd = v_dof_grid[x, y]
    if vd >= 0:
        vt[ vd ] = -1
    vd = v_dof_grid[x, (y+1)%s[1]]
    if vd >= 0:
        vt[ vd ] = 1

    return ut, vt


def lap_div(u_dof_grid, v_dof_grid, x, y):
    # Center

    s = u_dof_grid.shape
    
    u_d_ce, v_d_ce = div_eq(u_dof_grid, v_dof_grid, x, y)

    u_lap = Equation()
    v_lap = Equation()

    for px, py in [(-1,0),(1,0),(0,-1),(0,1)]:
        ax, ay = (x+px)%s[0], (y+py)%s[1]
        if u_dof_grid[ax, ay] == -3 or v_dof_grid[ax, ay] == -3:
            continue

        u_t, v_t = div_eq(u_dof_grid, v_dof_grid, ax, ay)

        u_lap =  u_lap + (u_t - u_d_ce)
        v_lap =  v_lap + (v_t - v_d_ce)

    return u_lap, v_lap

def config_code(solid, x, y):
    solid = roll( solid, -x + 1, axis = 0 )
    solid = roll( solid, -y + 1, axis = 1 )

    key = solid[0:3,0:3] * codex

    return key.sum()

def all_configs(solid):
    config_list = []
    for x in range(solid.shape[0]):
      for y in range(solid.shape[1]):
        config_list.append( config_code(solid, x, y) )

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

    five = zeros((5,5))
    five[1:4,1:4] = three
    return five



def index_coeff(equation, dofs):
    index_list2 = []
    coeff_list2 = []

    indexs = (ones(dofs.shape,dtype=int64).cumsum() - 1).flatten()

    # Kinda hackedy
    for dof_num, coeff in equation.items():
        flat_index_number = indexs[(dofs==dof_num).flatten()]
        index_list2.append(flat_index_number[0])
        coeff_list2.append( coeff )

    return index_list2, coeff_list2





# def index_coeff(equation, dofs):
#     flats = list(dofs.flat)

#     index_list = []
#     coeff_list = []

#     for dof_num, coeff in equation.items():
#         index_list.append( flats.index( dof_num ) )
#         coeff_list.append( coeff )

#     return index_list, coeff_list




def s_term(udof, vdof, x, y):
    if udof[x,y] == -3 or vdof[x,y] == -3:
        print("Solid square!!!")
        raise ValueError
    
    u_c, v_c = comp_mom_eq( udof, vdof, x, y )
    u_l, v_l = lap_div(     udof, vdof, x, y )
    
    uterms = u_c - u_l
    vterms = v_c - v_l

#     print "u_c:%s\nu_l:%s\nv_c:%s\nv_l:%s\n" % (u_c, u_l, v_c, v_l)

    return uterms, vterms


def s_term_matricies(pdof, udof, vdof, zero_rows = [0]):
    
    pdofs = pdof.max() + 1
    udofs = udof.max() + 1
    vdofs = vdof.max() + 1

    U_RHS_M = sparse.lil_matrix( (udofs, pdofs) )
    V_RHS_M = sparse.lil_matrix( (vdofs, pdofs) )

    for x in range(pdof.shape[0]):
      for y in range(pdof.shape[1]):
        # Skip non-pressure dof's
        pd = pdof[x,y]
        if pd < 0:
            continue

        # This is here to prevent writing and equation for a pinned cell
        if pd in zero_rows:
            continue

        ud = udof[x,y]
        vd = vdof[x,y]

        u_eq, v_eq = s_term(udof, vdof, x, y)

        for dofn, coeff in u_eq.items():
            U_RHS_M[ dofn, pd ] = coeff

        for dofn, coeff in v_eq.items():
            V_RHS_M[ dofn, pd ] = coeff

    return U_RHS_M.tocsc(), V_RHS_M.tocsc()

def div_matrices(p_dof_grid, u_dof_grid, v_dof_grid):
    # shape and locus correction
    s = p_dof_grid.shape

    # Counts for matrix sizes
    pdofs = p_dof_grid.max() + 1
    udofs = u_dof_grid.max() + 1
    vdofs = v_dof_grid.max() + 1

    # Matrix definition
    u_div_mat = sparse.lil_matrix((udofs, pdofs))
    v_div_mat = sparse.lil_matrix((vdofs, pdofs))

    # Figger out heach row's val
    for x in range(s[0]):
      for y in range(s[1]):
        # P Degree of freedom
        pd = p_dof_grid[x, y]
        if pd < 0:
            continue
        
        # U Degrees of freedom
        ud = u_dof_grid[x, y]
        if ud >= 0:
            u_div_mat[ ud, pd ] = -1

        ud = u_dof_grid[(x+1)%s[0], y]
        if ud >= 0:
            u_div_mat[ ud, pd ] = 1

        # V Degrees of freedom
        vd = v_dof_grid[x, y]
        if vd >= 0:
            v_div_mat[ vd, pd ] = -1

        vd = v_dof_grid[x, (y+1)%s[1]]
        if vd >= 0:
            v_div_mat[ vd, pd ] = 1

    return u_div_mat.tocsc(), v_div_mat.tocsc()

def vel_rhs_matrices(p_dof_grid, vel_dof_grid, axis):
    # shape and locus correction
    s = p_dof_grid.shape

    pdofs = p_dof_grid.max() + 1
    vel_dofs = vel_dof_grid.max() + 1

    # Eq definition
    vel_rhs_mat = sparse.lil_matrix( ( pdofs, vel_dofs ) )

    rolled_p = roll(p_dof_grid, 1, axis=axis)
    
    for x in range(s[0]):
      for y in range(s[1]):
        point = (x,y)

        if vel_dof_grid[point] < 0:
            continue 
        vel_dof = vel_dof_grid[point]
       
        p1 = p_dof_grid[point]
        if p1 >= 0:
            vel_rhs_mat[p1, vel_dof] = 1

        p2 = rolled_p[point]
        if p2 >= 0:
            vel_rhs_mat[p2, vel_dof] = -1

    return vel_rhs_mat.tocsc()




def grad_matrices(pdof, axis):
    pass
    




if __name__ == "__main__":
    sf = make_config_test(10)
        
    ud = velocity_dof(sf,0)
    vd = velocity_dof(sf,1)

    print("SOLID")
    print(sf)

    print("UDOF")
    print(ud)

    print("VDOF")
    print(vd)

    print(s_term(ud,vd,2,2))


    
        
        




