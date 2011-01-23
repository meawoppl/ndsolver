from scipy import *
from core import Solver
import hdf5

test = zeros((5,5))

test[2,2] = 1

sol = Solver(test, (1,0), sol_method="trilinos", printing=3)

sol.converge()

sol.sync("Extern")

hdf5.write_solver_to_h5("mpitest-results.h5", sol)

print sol.P
