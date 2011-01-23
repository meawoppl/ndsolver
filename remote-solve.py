from scipy import *
import ndsolver
import os, time

def debug_output(something):
    f = open("debug.log",'a')
    f.write("[%s] - " % time.ctime() + something + "\n")
    f.close()

def clear_debug():
    open("debug.log",'w')

debug_output( "task started for %s %s" % (task_path, str(dP) ) )

debug_output( "ndsolver in locals(): " + str("ndsolver" in locals()) )

# When we use this module the following must be pushed:
# S, max_div, dP
have_necessary_varibles = ("S"           in locals()
                           and "max_div" in locals()
                           and "dP"      in locals())

debug_output("Have necessary varibles: " + str(have_necessary_varibles) ) 

if not have_necessary_varibles:
    raise ValueError("The variables S, max_div, and dP must be 'pushed' to this script.")


# when we use this module the following may be pushed:
# P, u, v
debug_output( "Setting up Solver . . ." )

sol = ndsolver.Solver(S, dP, dbcallback = debug_output)

debug_output( "Solver Instanciated: " + str("sol" in locals()) )
debug_output("Starting solver")

sol.converge(stopping_div = max_div)
sol.regrid()

P = array( sol.P )
vgrids = [ array(v) for v in sol.V_GRIDS]
meta = sol.getMetaDict()

sol.nuke_all_trilinos()

del sol

meta['Solver'] = "ip-parallel"

return_vals_in_locals = ( "P"      in locals() and
                          "vgrids" in locals() and
                          "meta"   in locals() )

debug_output("Return vals in locals:" +  str(return_vals_in_locals) )
debug_output("Finished!")
debug_output("*" * 10)

