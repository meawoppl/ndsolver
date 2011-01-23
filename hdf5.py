import time
try:
    from tables import *
except ImportError:
    print "pytables is not installed or LD_PATH does not have the libraries necessary"
    import sys
    sys.exit()


# Keeping things from getting sloppy 
allowed_meta_tags = ["P_and_V_DOF_Number",
                     "Setup_Time",
                     "Converge_Time",
                     "Solver",
                     "Identity",
                     "Matrix_Solver",
                     "Solve_Time",
                     "Iteration_Count",
                     "Time_Recorded",
                     "dP",
                     "Total_Time"]

def forceGroup(h5, where, name, title):
    try:
        h5.getNode(where, name=name)
    except NoSuchNodeError:
        h5.createGroup(where, name, title)
    return h5.getNode(where, name=name)

def forceCArray(h5, where, name, atom, shape, title):
    try:
        h5.getNode(where, name=name)
    except NoSuchNodeError:
        h5.createCArray(where, name, atom=atom, shape=shape, title=title)
    return h5.getNode(where, name=name)
    

def get_S(filename):
    '''Quick and dirty way to get the solid array from a h5 file given a h5 object or a filename'''
    h5 = openFile(filename, "a")
    S = h5.root.geometry.S[:]
    h5.close()
    return S

def write_S(path, S):
    # Open h5
    h5 = openFile(path,'a')
    # Create groups if necessary
    geom = forceGroup(h5, "/", "geometry", 'details of the input geometry')

    # Setup CArray
    solid_atom = UInt8Atom()            # Binary, but whatever.
    solid_title = 'solid/liquid indicator array'
    solidCarray = forceCArray(h5, geom, "S", atom = solid_atom, shape=S.shape, title=solid_title)
    
    # Copy the data into the CArray
    solidCarray[:] = S[:]

    # Flush and Close
    h5.flush()
    h5.close()

def write_geometry(filename, points, radaii):
    # STUPID CHECK
    print len(radaii), points.shape[0]
    if len(radaii) != points.shape[0]:
        raise ValueError("Mismatch between points and radaii!")

    # Open the file
    h5 = openFile(filename, 'a')
    geom = forceGroup(h5, "/",  "geometry", 'details of the input geometry')
    ndim = points.shape[1]

    # Cases for different dimensionality
    if   ndim == 2:
        class Obstacle(IsDescription):
            radius = Float32Col()
            x = Float32Col()
            y = Float32Col()
    elif ndim == 3:
        class Obstacle(IsDescription):
            radius = Float32Col()
            x = Float32Col()
            y = Float32Col()
            z = Float32Col()
    else:
        # B0RK!
        raise ValueError("3d is highest supported hy hdf5 writing module!")

    # create the geometry group if necessary
    obstacles = h5.createTable(geom, 'obstacles', Obstacle, 'radii and centers of solid obstacles')
    obstacle = obstacles.row

    # Populate the points table
    for point, radius in zip(points, radaii):
        obstacle['radius'] =  radius # actual dimensionless radius of obstacle goes here
        if   ndim == 2:
            obstacle['x'], obstacle['y'] = point.T 
        elif ndim == 3:
            obstacle['x'], obstacle['y'], obstacle['z'] = point.T
        else:
            raise ValueError("3d is highest supported hy hdf5 writing module!")
        obstacle.append()

    # flush and close
    h5.flush()
    h5.close()



def has_dP_sim(filepath, dP):
    '''Check to see if a h5 file has a simulation for a given pressure drop . . .'''
    h5 = openFile(filepath, "a")
    sim_dim = dP_to_dim(dP)
    sim_name = "%s_sim" % sim_dim
    try:
        h5.getNode("/simulations/%s_sim" % sim_dim)
        has_group = True
    except NoSuchNodeError:
        has_group = False

    # Flush and Close 
    h5.flush()
    h5.close()
    return has_group


def get_dP_sim(h5, dP):
    sim_dim  = dP_to_dim(dP)
    sim_name = "%s_sim" % sim_dim

    return h5.getNode("/simulations/%s_sim" % sim_dim)

def dP_to_dim(dP):
    # Ugleeeee!
    ndim = len(dP)
    if   ndim == 2 and dP == (1,0):   sim_dim  = "x"
    elif ndim == 2 and dP == (0,1):   sim_dim  = "y"
    elif ndim == 3 and dP == (1,0,0): sim_dim  = "x"
    elif ndim == 3 and dP == (0,1,0): sim_dim  = "y"
    elif ndim == 3 and dP == (0,0,1): sim_dim  = "z"
    else:                             sim_dim  = str(dP)

    return sim_dim

# This does _NOT_ write S to the h5 file.  This is becuase this same routine is called by
# Both the parallel solver and the serial one
# The parallel solver already has S and geometry written . . . use the "ipython_task_to_h5" function

def get_sim_p_vs(h5, dP, shape):
    # Get the simulation group, make a simulation group if necessary
    sims = forceGroup(h5, "/", "simulations", 'Simulations details')

    # the code below works . . . . urgle . . .
    sim_dim = dP_to_dim(dP)
    sim_name = "%s_sim" % sim_dim
    
    ###########################
    # Table constructing tedium
    ###########################
    # For the simulation, make a group if necessary
    
    sim_title = 'periodic Stokes flow simulation for pressure drop in %s-direction' % sim_dim
    this_sim = forceGroup(h5, sims, sim_name, sim_title)
    
    # Make the P array if necessary
    p = forceCArray(h5, this_sim, "P", Float64Atom(), shape, 'cell-centered pressure')
    
    # Collect the velocity carrays in this
    vs = []

    # Record the velocity results
    dim_names = ["x","y","z"]
    vel_names = ["u","v","w"]
    for dim_num in range(len(shape)):
        # the varible name
        v_name = vel_names[dim_num]
        
        #dimension string
        dn = dim_names[dim_num]

        # if the array does not exist, make it
        v_title = title='%s-face-centered %s-component of velocity' % (dn, dn)
        v_Carray = forceCArray(h5, this_sim, v_name, Float64Atom(), shape, v_title)
        vs.append(v_Carray)

    return this_sim, p, vs

def set_save_token(h5_filename, number):
    h5 = openFile(h5_filename, 'a')
    h5.root._v_attrs.last_cpu = number
    h5.close()

def get_save_token(h5_filename):
    h5 = openFile(h5_filename, 'r')
    num = h5.root._v_attrs.last_cpu
    h5.close()
    return num

def del_save_token(h5_filename):
    h5 = openFile(h5_filename, 'a')
    del h5.root._v_attrs.last_cpu
    h5.close()
 
def h5_ready_for_me(h5_filename, myID):
    if myID == 0:
        return True
    # If the results from the cpu before me are visable
    if get_save_token(h5_filename) == (myID - 1):
        sleep(1)
        return True
    
    # Otherwise
    return False

def wait_for_token(h5_filename, solver):
    # Thread 0 waits for no man!
    if solver.myID == 0:
        return

    # Others wait until token is visable
    while(True):
        number_up = get_save_token(h5_filename)
        if number_up != solver.myID:
            solver.dbprint("Waiting my Turn! Current number up = %i" % number_up)
            time.sleep(0.5)
        if number_up != solver.myID:
            solver.dbprint("Its My Turn!")
            time.sleep(0.5)
            break
            
# This one should work for both, but will be slower!
def trilinos_h5_writer(savepath, solver):
    # Write the sim results structure
    # 0th cpu writes the initial table structure:
    for cpu in range(solver.cpuCount):
        solver.sync("Saving Loop Sync")
        if (solver.myID != cpu):
            continue

        wait_for_token(savepath, solver)

        # Foe each thread
        solver.dbprint("Writing Results for thread", 2)

        # Open h5 file
        h5 = openFile(savepath, mode="a")

        # This creates the results path, or grabs it if already created
        sim, p, vs = get_sim_p_vs(h5, solver.dP, solver.shape)

        solver.dbprint("Saving My Pressure", 2)
        solver.assign_P_to_obj(p)

        for n, v in enumerate(vs):
            solver.dbprint("Saving My Velocity (%i)" % n)
            solver.assign_V_to_obj(n, v)
        
        h5.flush()
        h5.close()

        set_save_token(savepath, solver.myID)

        # For each thread
        solver.dbprint("File Closed by thread", 2)        

    solver.sync("Parallel save completed")
    # Nuke the save token to avoid confusing future instances
    if solver.myID == 0: 
        del_save_token(savepath)
        

def wmd(sim_obj, meta_data):
    '''Not Designed to be Human Called'''
    meta = sim_obj._v_attrs
    meta.Time_Recorded = time.time()
    #TODO: SVN version number?
    
    # Put the passed metadata into the h5 object
    for key, val in meta_data.iteritems():
        # Make sure its valid
        if key not in allowed_meta_tags:
            raise ValueError("%s is not a valid metadata tag!" % key)
        # Add the metadata bit
        setattr(meta, key, val)
    

def write_meta(save_path, solver):    
    h5 = openFile(save_path, "a")
    sim, p, vs = get_sim_p_vs(h5, solver.dP, solver.shape)

    wmd(sim, solver.getMetaDict())

    h5.flush()
    h5.close()


def h5_writer(path, dP, P, Vs):
    ndim = len(Vs)
    if ndim > 3:
        raise ValueError("Greater than 3 dimensions not supported by h5 writer")

    # Input dimension check
    dP = tuple(dP)
    if len(dP) != ndim:
        raise ValueError("Pressure drop and simulation dimension mismatch!")

    # Open h5 file
    h5 = openFile(path, mode="a")

    # h5 C Array objects
    sim, p, vs = get_sim_p_vs(h5, dP, P.shape)

    # Copy in the Pressure results 
    p[:] = P

    # Record the velocity results
    for dim in range(ndim):        
        # Assign the fucking array already
        vs[dim][:] = Vs[dim]

    # Flush and close the h5
    h5.flush()
    h5.close()

def write_solver_to_h5(file_path, solver):
    # Only main thread has results  . . .
    # Convert the internals back into grids
    solver.dbprint("Saving to hdf5 file", 1)

    # This is true for single thread execution (non-trilinos)
    # AND the 0th thread of trilinos
    if solver.myID == 0:
        # write sim results structure
        write_meta(file_path, solver)

        # Write the S and geometry etc. and close
        solver.dbprint("Thread writing S and metadata")
        write_S(file_path, solver.S)

    # Make sure above is complete before moving on    
    # AKA make sure thread one has written S, and metadata first
    solver.sync()

    # Write the simulation results
    if solver.using_trilinos:
        trilinos_h5_writer(file_path, solver)
    else:
        solver.regrid()        
        h5_writer(file_path, solver.dP, solver.P, solver.V_GRIDS)

    solver.dbprint("Completed Save", 1)
    

def ipython_task_to_h5(task):
    # Grab the results
    r = task.results

    # Make Solver reflect the task-client etc.
    meta_stuff = r['meta']
    meta_stuff["Solver"] = "ip-finalconverge"
    v_grids = r["vgrids"]

    # write metadata
    h5 = openFile(r["task_path"], 'a')
    sim, p, vs = get_sim_p_vs(h5, r["dP"], r["P"].shape)

    wmd(sim, r["meta"])    
    h5.flush()
    h5.close()

    # Pass it off to the hdf5 writer
    h5_writer(r["task_path"], r["dP"], r["P"], v_grids)
