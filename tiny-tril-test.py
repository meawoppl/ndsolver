from scipy import *
from core import Solver

test = zeros((5,5))
test[2,2] = 1


S = Solver(test, (1,0), sol_method = "trilinos")
S.converge()

S.regrid()


S._fill_DM(0)
for x in range(S.ndim):
    print all(S.GM[x].todense() == S.VEL_RHS[x].todense())
