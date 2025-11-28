import numpy as np
from numpy import (allclose, all, array, arctan2, around, byte, c_, dot,
                   linspace, mgrid, ones, outer, pi, roll, sin, sqrt, zeros)
from scipy import linalg
from ndsolver.core import Solver
from ndsolver.symbolic import ndim_eq
from ndsolver import hdf5
import tables
import time

test_matrix = zeros((5,5))
test_matrix[2,2] = 1

three_test = zeros((5,5,5))
three_test[:,2,2] = 1

correct_P = array([[ 1.  ,  1.05,  1.15,  1.05,  1.  ],
                   [ 0.85,  1.  ,  1.55,  1.  ,  0.85],
                   [ 0.55,  0.55,  0.  ,  0.55,  0.55],
                   [ 0.25,  0.1 , -0.45,  0.1 ,  0.25],
                   [ 0.1 ,  0.05, -0.05,  0.05,  0.1 ]])

correct_u = array([[ 0.144,  0.11 ,  0.072,  0.11 ,  0.144],
                   [ 0.151,  0.112,  0.054,  0.112,  0.151],
                   [ 0.167,  0.123,  0.   ,  0.123,  0.167],
                   [ 0.167,  0.123,  0.   ,  0.123,  0.167],
                   [ 0.151,  0.112,  0.054,  0.112,  0.151]])

correct_v = array([[ 0.   , -0.007, -0.009,  0.009,  0.007],
                   [ 0.   , -0.016, -0.027,  0.027,  0.016],
                   [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
                   [ 0.   ,  0.016,  0.027, -0.027, -0.016],
                   [ 0.   ,  0.007,  0.009, -0.009, -0.007]])

def proj(u, v):
   '''Return the vector projection of u onto v.
   '''
   return dot(u, outer(v, v))

def perp(u, v):
   '''Return the perpendicular component of the projection of u onto v.
   '''
   return u - proj(u, v)

def oblique_111cylinder():
    axis = array([1., 1., 1.])  # cylinder axis
    axis /= linalg.norm(axis)

    res = 10
    radius = 0.3

    z, y, x = mgrid[0:1:res*1j, 0:1:res*1j, 0:1:res*1j]
    coords = c_[x.flat, y.flat, z.flat]

    s = zeros(x.shape, dtype=byte)

    for xi in [0., 1.]:
        for yi in [0., 1.]:
            for zi in [0., 1.]:
                if xi + yi + zi == 3:
                    continue
                dc = coords - array([xi, yi, zi])
                norms = array([linalg.norm(p) for p in perp(dc, axis)])
                s += 1 * (norms.reshape((res, res, res)) <= radius)
    return s

def clear_autotest_h5():
   from os import system
   system("rm autotest.h5")

def test_2d(sol_method='default'):
   print("Testing 2d iterative")
   sol = Solver(test_matrix, (1,0), sol_method=sol_method)
   sol.converge(max_iter=10)

   clear_autotest_h5()
   hdf5.write_solver_to_h5("autotest.h5", sol)

   # Reading is threadsafe!
   h5 = tables.open_file("autotest.h5")
   test_P = h5.root.simulations.x_sim.P[:]
   h5.close()

   if not allclose(sol.P, correct_P):
      print("Incorrect:")
      print(sol.P)
      print()
      print("Correct:")
      print(correct_P)
      raise ValueError("Incorrect answer for 2-d test case\n See 'autotest.h5'")
   else:
      print("Test Successful!!!")
      clear_autotest_h5()

   return sol

def shift_test(sol_method='default'):
   print("Testing 2d shifting")

   for x in range(5):
      for y in range(5):
         rolled_solid = roll(test_matrix, x, axis=0)
         rolled_solid = roll(rolled_solid, y, axis=1)

         rolled_u = roll(correct_u, x, axis=0)
         rolled_u = roll(rolled_u, y, axis=1)

         rolled_v = roll(correct_v, x, axis=0)
         rolled_v = roll(rolled_v, y, axis=1)

         sol = Solver(rolled_solid, (1,0), sol_method=sol_method)
         sol.converge(max_iter=5)

         clear_autotest_h5()
         sol.sync()
         hdf5.write_solver_to_h5("autotest.h5", sol)
         sol.sync()
         # Reading is threadsafe!
         h5 = tables.open_file("autotest.h5")
         test_P = h5.root.simulations.x_sim.P[:]
         test_u = h5.root.simulations.x_sim.u[:]
         test_v = h5.root.simulations.x_sim.v[:]
         h5.close()
         sol.sync()

         # Phrasing the subtraction this way makes
         du = (rolled_u - test_u)
         dv = (rolled_v - test_v)

         print(f"Offset test ({x}, {y}) Mean du: {du.mean()} Mean dv: {dv.mean()}")

         assert all(du < 1e-8)
         assert all(dv < 1e-8)

         print("Success . . ")
         clear_autotest_h5()      

   

def test_3d(sol_method='default'):
    sol = Solver(three_test, (0,1,0), sol_method=sol_method)
    sol.converge()
    sol.regrid()

    if not allclose(sol.P[2,:,:], correct_P):
        raise ValueError("Incorrect answer for 3-d test case")
    else:
        print("Test Successful!!!")

    return sol

def test_tube(sol_method="spsolve"):
    liquid = oblique_111cylinder() 
    solid = 1 - liquid
    sol = Solver(1-solid, (1,0,0), sol_method=sol_method)
    # sol.monolithic_solve()
    sol.converge()

    hdf5.write_solver_to_h5("3d-tube.h5", sol)

def test_helix(sol_method="default"):
    liquid = helix() 
    solid = 1 - liquid
    sol = Solver(solid, (0.,0.,1.), sol_method=sol_method)
    sol.converge()

    hdf5.write_S("semi-helix.h5", solid)
    hdf5.write_solver_to_h5("semi-helix.h5", sol)

def test_tables(sol_method='default'):
   print("Testing hdf5 saving capabilities:")
   sol = Solver(test_matrix, (1,0), sol_method=sol_method)
   sol.converge(1e-10)

   hdf5.write_solver_to_h5("autotest.h5", sol)
   print("Test Sucessful!")

def helix():
   x, y, z = mgrid[-1:1:31j,-1:1:31j,-1:1:31j]

   r = sqrt(x**2 + y**2)
   theta = arctan2(y, x)

   # theta[(z>=-1  ) & (z<0.5) ] = -abs(theta[(z>=-1  ) & (z<0.5) ])
   # theta[(z>=-0.5) & (z<0  ) ] =  abs(theta[(z>=-0.5) & (z<0  ) ])
   # theta[(z>=0   ) & (z<0.5) ] =  abs(theta[(z>=0   ) & (z<0.5) ])
   # theta[(z>=0.5 ) & (z<1  ) ] = -abs(theta[(z>=0.5 ) & (z<1  ) ])
   
   # print theta.min(), theta.max()

   # 1/0

   tube_tube = (r > 0.5) & (r < 0.8)
   theta_hi = sin(pi * z) - (pi/8)
   theta_lo = sin(pi * z) + (pi/8)

   sli = (theta > theta_hi) & (theta < theta_lo)
   # sli = around(theta - (pi/4), 2) == 0   
   tube = tube_tube & sli

   return tube

def test_all_2d_config(sol_method='default'):
    print("Now running all cell configurations:")
    cfg_iter = []
    for x in range(1, 256):
        solid = ndim_eq.make_safe_config_test(x)
        print(f"Starting Config {x}")
        print("Solid--")
        print(solid)

        start_time = time.time()
        a = Solver(solid, (1,0), sol_method=sol_method)
        setup_time = time.time()
        a.converge()
        finish_time = time.time()

        print(f"Config {x} - ")
        print(f"\t{a.I} iterations. ")
        print(f"\tSetup Time:{setup_time - start_time}")
        print(f"\tConverge Time:{finish_time - setup_time}")

    print("Test Successful!!! Solver converged for all 255 configurations!")


def test_monolithic_2d(sol_method='default'):
    s = Solver(test_matrix, (1,0))
    s.monolithic_solve()
    s.regrid()
    if not allclose(s.P, correct_P):
        print("Answer Different!")
        print("Correct:")
        print(correct_P)
        print("Wrng!:")
        print(s.P)
        raise ValueError("Unittest failure")
    print("Test Sucessful!")


def test_all(sol_method='default'):
    test_tables(sol_method)
    shift_test(sol_method)
    test_all_2d_config(sol_method)
    test_monolithic_2d()

def do_validation_runs(domain_width=50., count=10, filename="validation.h5"):
    # TODO: Out of date?
    if count < 1:
        raise ValueError("Invalid count.")

    domain_shape = (int(domain_width), int(domain_width))
    max_radius = domain_width / 2.
    x, y = mgrid[-1:1:1j*domain_width, -1:1:1j*domain_width]

    radaii = linspace(0, max_radius, count + 2) / max_radius
    radaii = radaii[1:-1]

    h5 = tables.open_file(filename, "w")
    for n, radius in enumerate(radaii):

        solid = x**2 + y**2 < radius**2
        s = Solver(solid, (1,0))
        s.converge()
        s.regrid()

        # Create the groups we need.
        table_title = f"flow around cylinder with non-dimensional radius of {radius:06f}"
        table_name = f"run_{n}"
        r_group = h5.create_group("/", table_name, title=table_title)
        h5.create_carray(r_group, "S", tables.Int8Atom(), domain_shape)
        for name in ["P", "u", "v"]:
            tab_atom = tables.Atom.from_dtype(s.P.dtype)
            h5.create_carray(r_group, name, tab_atom, domain_shape)

        r_group.S[:] = solid
        r_group.P[:] = s.P
        r_group.u[:] = s.V_GRIDS[0]
        r_group.v[:] = s.V_GRIDS[1]

        meta = r_group._v_attrs

        meta.nondimensional_radius = radius
    print("Validation Runs Completed")




if __name__ == "__main__":
   pass

   # unittest.main()
   # test_all()
   # hdf5.write_S("oc.h5", oblique_111cylinder())
   # print test_helix()
   # test_helix()
   # test_tube()
   # s = test_3d()
   # test_tables(test_3d())
   # do_validation_runs()

   # shift_test(sol_method="trilinos")
