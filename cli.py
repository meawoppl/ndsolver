#!/usr/bin/env python
from core import Solver
from benchmark import converge_150

import hdf5
import sys, os
from optparse import OptionParser
from scipy import misc

# Pretty self explanatory really . . .
parser = OptionParser()

parser.add_option( "-f", "--h5file", dest="filename",
                   help="hdf5 file to operate on.  If given image files, it will create an h5 file in-place with the same filename other than the extension." )

parser.add_option( "-o", "--overwrite", dest="force",
                   action="store_true", default=False,
                   help="Run the simulation even if results exist in the h5." )

parser.add_option( "-d", "--h5dir", dest="directory",
                   help="gives a directory (recursive) to converge all files ending '.h5'.  If given image files, it will create an h5 file in-place with the same filename other than the extension." )

parser.add_option( "-s", "--solver", dest="solver",
                   default="default",
                   help="Sets the matrix solver. Options are: spsolve (default), nobi, splu, ruge, bicgstab, trilinos" )

parser.add_option( "-v", "--verbose", dest="verb",
                   default=2,
                   help="set the outputting of the solver (0-3) 2 is default.")

parser.add_option( "-c", "--converge", type=float,
                   dest="converge", default = 1e-8,
                   help="Use the iterative solver(default). If convergence criteria is unspecified the default is 1e-8" )

parser.add_option( "-m", "--monolithic",
                   default = False,
                   dest="mono",
                   action="store_true",
                   help="Use the direct monolithic solve function" )

parser.add_option( "--random", dest="rand",
                   default=0, type=int,
                   help="Give a seed to randomize the order the files are converged in." )

parser.add_option( "-r", "--reverse", dest="reverse",
                   action="store_true", default=False,
                   help="Reverse the order the files are solved in." )

# Add the three simulation direction options
for sim_dir in ["x", "y", "z"]:
    parser.add_option( "-%s" % sim_dir,"--%s_sim" % sim_dir,
                       default = False,
                       dest= "%s" % sim_dir,
                       action="store_true",
                       help="Converge and save a simulation in the %s direction" % sim_dir )

parser.add_option( "-t","--test",
                   default = False,
                   dest="test",
                   action="store_true",
                   help="Test all 255 configurations")

parser.add_option( "-b","--big",
                   default = False,
                   dest="bigmode",
                   action="store_true",
                   help="Enable Big Mode")

parser.add_option( "-n","--nobiot",
                   default = False,
                   dest="nobi",
                   action="store_true",
                   help="Disable Biot Number Acceleration")

(options, args) = parser.parse_args()

if options.test:
    from unittest_validate import test_all
    test_all(sol_method=options.solver)
    sys.exit()

# One or the other or both . . . cmon guys
if options.filename == None and options.directory == None and not options.test:
    parser.print_help()
    raise ValueError("You must specify either a directory (-d) or file (-f).")

# you _can_ specify both a -d and -f!
targets = []

# Populate list of targets from -f
if options.filename != None:
    targets.append(options.filename)

# Populate list of targets from -d
if options.directory != None:
    for path, fold, fils in os.walk(options.directory):
        for f in fils:
            # skip svn folders.
            if not ".svn" in path:
                targets.append( os.path.join(path, f) )

targets.sort()

if options.rand:
    import random
    random.seed(options.rand)
    random.shuffle(targets)

if options.reverse:
    targets.reverse()

# Print the targets
print "The following files will be operated on:"
for f in targets:
    print "\t", f

# Populate the various pressure drops we will use 
print "Using %s solver." % options.solver

# For each file targeted . . .
for f in targets:

    # Determine where to save (this accommodates h5's and image format)
    # Saving to h5's
    save_path = os.path.splitext(f)[0] + ".h5"


    if hdf5.isHDF5File(f):
        print "Loading", f, "as hdf5 file . . ."
        # open the h5 and get the dimensionality
        h5 = hdf5.openFile(f)
        ndim = len(h5.root.geometry.S.shape)
        h5.close()
    else:
        # Default to trying an image . . .
        print f, "is not and HDF5 file . . . trying image?"
        S = 1. * ( misc.imread(f, flatten=True) < (255/2.) )

        # Appropriate image transformation to make x/y convention meaningful
        S = S[::-1].T

        ndim = 2

        # Make a h5 file for the results
        hdf5.write_S(save_path, S)

    # Get the domain dimensionality 
    print "Dimension:", ndim

    # Populate the pressure drops based on user input
    # Done here and not earlier b/c we need to know the dimensionality
    dPs = []
    # Have dp/dx
    if   options.x and ndim == 2:
        dPs.append((1,0))
    elif options.x and ndim == 3:
        dPs.append((1,0,0))

    # have dp/dy 
    if   options.y and ndim == 2:
        dPs.append((0,1))
    elif options.y and ndim == 3:
        dPs.append((0,1,0))

    # dp/dz (3d only) 
    if   options.z and ndim == 3:
        dPs.append((0,0,1))

    if options.x and ndim == 4:
        dPs.append((1,0,0,0))
    if options.y and ndim == 4:
        dPs.append((0,1,0,0))
    if options.z and ndim == 4:
        dPs.append((0,0,1,0))

    # For each pressure drop
    for dP in dPs:
        # Print which dP were using currently
        print "Doing %s-sim" % str(dP)

        # Check to see if a simulation has been done
        if hdf5.has_dP_sim(save_path, dP) and options.force:
            print "\tSimulation Detected! Results will be overwritten!"
        elif hdf5.has_dP_sim(save_path, dP) and not options.force:
            print "\tSimulation Detected! Skipping!"
            continue
        
        # If bigmode, just pass the filename instead of the solid
        if options.bigmode:
            S = save_path
        else:
            S = hdf5.get_S(save_path) 

        # Setup solver, printing full debug info, using the solver specified
        sol = Solver(S, dP, printing=int(options.verb), sol_method=options.solver)

        # setup is now implicit
        # sol.setup()

        #if we want to solve monolithically . . . or use the iterative solver
        if options.mono:
            sol.monolithic_solve()
        else:
            sol.converge(options.converge)

        # Save dis shit.
        print "Saving . . ."
        hdf5.write_solver_to_h5(save_path, sol)

        del sol
