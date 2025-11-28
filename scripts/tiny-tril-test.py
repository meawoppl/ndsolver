import numpy as np
from ndsolver.core import Solver

test = np.zeros((5,5))
test[2,2] = 1


S = Solver(test, (1,0), sol_method="spsolve")
S.converge()

S.regrid()


S._fill_DM(0)
for x in range(S.ndim):
    print(np.all(S.GM[x].todense() == S.VEL_RHS[x].todense()))
