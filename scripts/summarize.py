from scipy import *
from pylab import *
from tables import *
import matplotlib

import os


def h5s_in_dir(directory):
    files_to_process = []

    # Spider the h5 directory and generate a list of files to process
    for der, fold, files in os.walk(directory):
        for h5_file in files:
            if h5_file.endswith(".h5"):
                files_to_process.append( os.path.join(der, h5_file) )
                
    return files_to_process


def process_2d_dir(h5_path, summary_file):

    files_to_process = h5s_in_dir(h5_path)
    # For testing
    # files_to_process = files_to_process[::100]

    # Calculate the number of points
    data_count = len(files_to_process)
    print(f"{data_count} files to process.")

    # Allocate a bigish array
    big_array = recarray((data_count,),
                         dtype=[ ("path","S128"),
                                 ("f_s","<f8"),
                                 ("width","<i8"), ("height","<i8"), 
                                 ("Kxx","<f8"), ("Kxy","<f8"),
                                 ("Kyx","<f8"), ("Kyy","<f8"),
                                 ("obs_count" ,"<i8"),
                                 ("obs_radius","<f8") ])

    # Keep a list of files that errors occur during
    corrupt = []

    # For each file populate a row of the big array
    for file_num, f in enumerate(files_to_process):
        # Print debugging infoz
        print(f"{file_num} {f}")

        # This block can fail if there is a corrupt file or incomplete simulation data
        try:
            # open the h5 and extract S
            h5 = open_file(f)
            s = h5.root.geometry.S[:]

            # Permebility tensor bits.
            Kxx = h5.root.simulations.x_sim.u[:].mean()
            Kxy = h5.root.simulations.x_sim.v[:].mean()
            Kyx = h5.root.simulations.y_sim.u[:].mean()
            Kyy = h5.root.simulations.y_sim.v[:].mean()

            # Obstacle information
            nd_radii = h5.root.geometry.obstacles[:]["radius"]
            obs_count = h5.root.geometry.obstacles.nrows

            # Close the h5
            h5.close()
        
        except (IOError, NoSuchNodeError, AttributeError) as inst:
            # Barring failure, add it to the failed list
            print(f"Skipping {f}")
            print("In process or corrupted?")
            print(inst)
            corrupt.append(f)
            h5.close()
            continue

        # Populate the array row with the info out of the try block
        width, height = s.shape
        f_s = s.mean()
        radius = nd_radii[0]
        big_array[file_num] = (f, f_s,
                               width, height,
                               Kxx, Kxy,
                               Kyx, Kyy,
                               obs_count,
                               radius)
        
    # Open the summary h5
    results_h5 = open_file(summary_file, 'w')
        
    # Create the table and throw it all in there
    res_tab = results_h5.createTable("/", "results", big_array)

    # Close the summary h5
    results_h5.close()


def process_3d_dir(h5_path, summary_file):

    files_to_process = h5s_in_dir(h5_path)
    # For testing
    # files_to_process = files_to_process[::100]

    # Calculate the number of points
    data_count = len(files_to_process)
    print(f"{data_count} files to process.")

    # Allocate a bigish array
    big_array = recarray((data_count,),
                         dtype=[ ("path","S128"),
                                 ("f_s","<f8"),
                                 ("width","<i8"), ("height","<i8"), ("depth","<i8"), 
                                 ("Kxx","<f8"), ("Kxy","<f8"), ("Kxz","<f8"), 
                                 ("Kyx","<f8"), ("Kyy","<f8"), ("Kyz","<f8"),
                                 ("Kzx","<f8"), ("Kzy","<f8"), ("Kzz","<f8"),
                                 ("obs_count" ,"<i8"),
                                 ("obs_radius","<f8") ])

    # Keep a list of files that errors occur during
    corrupt = []

    # For each file populate a row of the big array
    for file_num, f in enumerate(files_to_process):
        # Print debugging infoz
        print(f"{file_num} {f}")

        # This block can fail if there is a corrupt file or incomplete simulation data
        try:
            # open the h5 and extract S
            h5 = open_file(f)
            s = h5.root.geometry.S[:]

            # Permebility tensor bits.
            Kxx = h5.root.simulations.x_sim.u[:].mean()
            Kxy = h5.root.simulations.x_sim.v[:].mean()
            Kxz = h5.root.simulations.x_sim.w[:].mean()
            
            Kyx = h5.root.simulations.y_sim.u[:].mean()
            Kyy = h5.root.simulations.y_sim.v[:].mean()
            Kyz = h5.root.simulations.y_sim.w[:].mean()

            Kzx = h5.root.simulations.z_sim.u[:].mean()
            Kzy = h5.root.simulations.z_sim.v[:].mean()
            Kzz = h5.root.simulations.z_sim.w[:].mean()

            # Obstacle information
            nd_radii = h5.root.geometry.obstacles[:]["radius"]
            obs_count = h5.root.geometry.obstacles.nrows

            # Close the h5
            h5.close()
        
        except (IOError, NoSuchNodeError, AttributeError) as inst:
            # Barring failure, add it to the failed list
            print(f"Skipping {f}")
            print("In process or corrupted?")
            print(inst)
            corrupt.append(f)
            h5.close()
            continue

        # Populate the array row with the info out of the try block
        width, height, depth = s.shape
        f_s = s.mean()
        radius = nd_radii[0]
        big_array[file_num] = (f, f_s,
                               width, height, depth,
                               Kxx, Kxy, Kxz,
                               Kyx, Kyy, Kyz,
                               Kzx, Kzy, Kzz,
                               obs_count, radius)

    big_array = big_array[big_array['path'] != '']
        
    # Open the summary h5
    results_h5 = open_file(summary_file, 'w')
        
    # Create the table and throw it all in there
    res_tab = results_h5.createTable("/", "results", big_array)

    for c in corrupt:
        print(c)
    print(f"Problems encountered in {len(corrupt)} files")

    # Close the summary h5
    results_h5.close()

if __name__ == "__main__":
    pass
