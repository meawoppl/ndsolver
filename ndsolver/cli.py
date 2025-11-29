#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np
from PIL import Image
from tables import is_pytables_file, open_file

from .core import Solver
from . import hdf5


def build_parser():
    parser = argparse.ArgumentParser(
        description="ndsolver CLI - Solve Stokes flow in periodic porous media"
    )

    parser.add_argument("-f", "--h5file", dest="filename",
                        help="HDF5 file to operate on. If given image files, creates an h5 file with same name.")

    parser.add_argument("-o", "--overwrite", dest="force",
                        action="store_true", default=False,
                        help="Run the simulation even if results exist in the h5.")

    parser.add_argument("-d", "--h5dir", dest="directory",
                        help="Directory (recursive) to converge all '.h5' files.")

    parser.add_argument("-s", "--solver", dest="solver",
                        default="default",
                        help="Matrix solver: spsolve (default), splu, bicgstab, trilinos")

    parser.add_argument("-c", "--converge", type=float,
                        dest="converge", default=1e-8,
                        help="Convergence criteria for iterative solver (default: 1e-8)")

    parser.add_argument("-m", "--monolithic", dest="mono",
                        action="store_true", default=False,
                        help="Use the direct monolithic solve function")

    parser.add_argument("--random", dest="rand",
                        default=0, type=int,
                        help="Seed to randomize file processing order.")

    parser.add_argument("-r", "--reverse", dest="reverse",
                        action="store_true", default=False,
                        help="Reverse the order files are solved in.")

    parser.add_argument("-x", "--x_sim", dest="x",
                        action="store_true", default=False,
                        help="Converge and save a simulation in the x direction")

    parser.add_argument("-y", "--y_sim", dest="y",
                        action="store_true", default=False,
                        help="Converge and save a simulation in the y direction")

    parser.add_argument("-z", "--z_sim", dest="z",
                        action="store_true", default=False,
                        help="Converge and save a simulation in the z direction")

    parser.add_argument("-t", "--test", dest="test",
                        action="store_true", default=False,
                        help="Test all 255 configurations")

    parser.add_argument("-b", "--big", dest="bigmode",
                        action="store_true", default=False,
                        help="Enable Big Mode (memory-mapped arrays)")

    parser.add_argument("-n", "--nobiot", dest="nobi",
                        action="store_true", default=False,
                        help="Disable Biot Number Acceleration")

    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if options.test:
        from tests.test_validate import test_all
        test_all(sol_method=options.solver)
        sys.exit()

    # One or the other or both
    if options.filename is None and options.directory is None:
        parser.print_help()
        raise ValueError("You must specify either a directory (-d) or file (-f).")

    # You can specify both -d and -f
    targets = []

    if options.filename is not None:
        targets.append(options.filename)

    if options.directory is not None:
        for path, fold, fils in os.walk(options.directory):
            for f in fils:
                if ".svn" not in path and ".git" not in path:
                    targets.append(os.path.join(path, f))

    targets.sort()

    if options.rand:
        import random
        random.seed(options.rand)
        random.shuffle(targets)

    if options.reverse:
        targets.reverse()

    print("The following files will be operated on:")
    for f in targets:
        print(f"  {f}")

    print(f"Using {options.solver} solver.")

    for f in targets:
        save_path = os.path.splitext(f)[0] + ".h5"

        if is_pytables_file(f):
            print(f"Loading {f} as HDF5 file...")
            h5 = open_file(f)
            ndim = len(h5.root.geometry.S.shape)
            h5.close()
        else:
            print(f"{f} is not an HDF5 file, trying as image...")
            img = Image.open(f).convert('L')
            solid = (np.array(img) < 128).astype(np.int8)
            solid = solid[::-1].T
            ndim = 2
            hdf5.write_s(save_path, solid)

        print(f"Dimension: {ndim}")

        # Build pressure drop list based on dimensionality
        pressure_drops = []
        if options.x:
            if ndim == 2:
                pressure_drops.append((1, 0))
            elif ndim == 3:
                pressure_drops.append((1, 0, 0))
            elif ndim == 4:
                pressure_drops.append((1, 0, 0, 0))

        if options.y:
            if ndim == 2:
                pressure_drops.append((0, 1))
            elif ndim == 3:
                pressure_drops.append((0, 1, 0))
            elif ndim == 4:
                pressure_drops.append((0, 1, 0, 0))

        if options.z:
            if ndim == 3:
                pressure_drops.append((0, 0, 1))
            elif ndim == 4:
                pressure_drops.append((0, 0, 1, 0))

        for dp in pressure_drops:
            print(f"Doing {dp}-sim")

            if hdf5.has_dp_sim(save_path, dp):
                if options.force:
                    print("  Simulation detected! Results will be overwritten.")
                else:
                    print("  Simulation detected! Skipping.")
                    continue

            if options.bigmode:
                solid = save_path
            else:
                solid = hdf5.get_s(save_path)

            sol = Solver(solid, dp, sol_method=options.solver)

            if options.mono:
                sol.monolithic_solve()
            else:
                sol.converge(options.converge)

            print("Saving...")
            hdf5.write_solver_to_h5(save_path, sol)
            del sol


if __name__ == "__main__":
    main()
