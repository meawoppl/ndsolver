import warnings
from numpy import ( all, allclose, arange, array, average, concatenate, 
                    c_, cumsum, dot, float64, inf,  int64, logical_and,
                    logical_not, logical_or, mean, memmap, ones, r_, roll, sqrt, 
                    take, where, zeros, zeros_like )
import time
from socket import gethostname

# libs that do the symbolic S terms etc.
from symbolic import ndim_eq, ndimed

# Scipy and Solvers are imported as needed to keep memory profile low!

class Solver():
    '''This is the setup method of the solver.  It instantiates a class
    with methods central to solving low Reynolds number flows.

    Required Arguments:
    'solid' is an n-dimensional array describing geometry of the problem
    'dP' is list/tuple of pressure difference across
    the n-th dimension of the domain"
    '''
    def __init__(self, solid_or_filename, dP, sol_method = "default", printing = False, dbcallback = None ):
        # Log the starting time
        self.start_time = time.time()

	# Set debugging flag
	self.printlevel = printing
        self.dbcallback = dbcallback

        # Trilinos setup and iteration requires very different programmatic flow . . . 
        self.using_trilinos = ( sol_method == "trilinos")

        # Trilinos communicator between threads . . .
        if self.using_trilinos:
            # Epetra Imported Here and all vectors defined
            from PyTrilinos import Epetra 
            self.Comm = Epetra.PyComm()    
            self.myID = self.Comm.MyPID()
            self.cpuCount = self.Comm.NumProc()
            self.dbprint("Trilinos Inititated: %s" % gethostname())
        else:
            # Only one thread
            self.myID = 0
            self.cpuCount = 1

        self.dbprint("Solver Instantiated", level = 2)
        
        # Default direction of pressure drop
	self.dP = dP

        # I left this in here so you can manually 
        # disable Biot number based acceleration if desired
        self.useBi = True  

        if sol_method == "default":
            self.method = "spsolve"
        else:
            self.method = sol_method

        # Iteration count
        self.I = 0

        ################################################################
        # Solver Internal Stuff ## Degree of Freedom Grids and Numbers #
        ################################################################
        
        # the ndarray of solid
        if hasattr(solid_or_filename, "shape"):
            self.S = solid_or_filename
            self.shape = self.S.shape
            self.ndim  = len(self.S.shape)

            # Pressure (cell centered)
            # Get the P degrees of freedom
            self.P_dof_grid = ndim_eq.p_dof( self.S )

            # Velocities (face centered)
            self.vel_dof_grids = ndim_eq.velocity_dofs( self.S )
            self.bigMode = False

        elif type(solid_or_filename) == str:
            if self.myID == 0:
                self.setup_dof_cache_faster(solid_or_filename)
            
            self.sync()
            self.dbprint("Waiting for cached file to flush to disk.")
            time.sleep(1)
            self.sync()

            self.import_dof_cache()
            self.bigMode = True

        self.P_dof_num  = self.P_dof_grid.max() + 1
        self.vel_dof_nums  = [int(grid.max() + 1) for grid in self.vel_dof_grids]
        self.sync("DOF Config Completion Sync")

        self.la_is_setup = False
        self.bc_is_setup = False

    def force_la_setup(self):
        '''Force the solver to set-up the linear algebra bits (vectors, matrices etc.)'''
        if not self.la_is_setup:
            self.setup_la()

    def force_bc_setup(self):
        '''Force the solver to set-up the boundary conditions (P_CORR and V_CORRS etc.)'''
        if not self.bc_is_setup:
            self.setup_bc()
    
    def setup_la(self):
        self.dbprint("Starting Setup Routine", 2)
        # Everything is ND wrt.  the 0th axis
        self.h = 1./self.shape[0]

        ################# 
        # Stupid checks #
        #################        
        if self.ndim != len(self.dP):
            raise ValueError("Solid Array and Pressure Drop do not have matching dimensions:\n\tself.ndim:%i\n\t%s" % (self.ndim, dP))

        #################
        # FYI Printouts #
        #################
        # DOF Number total
        self.dof_number = sum(self.vel_dof_nums)  + self.P_dof_num
       
        # print some useful DOF debugging information:
        self.dbprint( "Degree of freedom count: %i" % self.dof_number )
        self.dbprint( "\tPressure: %i" % self.P_dof_num )
        for dim in range(self.ndim):
            self.dbprint( "\tVelocity %i: %i" % (dim, self.vel_dof_nums[dim]) )

        ##########################
        # DOF Grabbing Functions #
        ##########################
        # This returns the pressure DOF for a given point
        self.pdp = lambda point: int( self.P_dof_grid[tuple(array(point) % array(self.shape))] )
        
        # This returns the velocity DOF for a given dimension and point
        self.nvd = lambda dim, point: int( self.vel_dof_grids[dim][tuple(array(point) % array(self.shape))] )        
        ##########################
        # Linear Algebra Related #
        ##########################

        # The maximum number of non-zero entries on a Poisson matrix of dimension self.ndim
        self.max_row_nz = 1 + (2 * self.ndim)
            
        ###########
        # Vectors #
        ###########
        # Vector shaped pressure and correction
        if self.using_trilinos:
            from PyTrilinos import Epetra 
            self.PMap = Epetra.Map(self.P_dof_num, 0, self.Comm)

            self.myP_dof_min = self.PMap.MinMyGID()
            self.myP_dof_max = self.PMap.MaxMyGID()

            self.P_LHS = Epetra.Vector(self.PMap)
            self.P_RHS = Epetra.Vector(self.PMap)
            self.P_COR = Epetra.Vector(self.PMap)
            self.PTEMP = Epetra.Vector(self.PMap)

            # Biot number and Divergence have the same vector size/Map
            self.Bi       = Epetra.Vector(self.PMap)
            self.DIV_MULT = Epetra.Vector(self.PMap)
        
            # Numbers related to the divergence
            self.last_abs_div= Epetra.Vector(self.PMap)
            self.D_LIN       = Epetra.Vector(self.PMap)
            self.ABS_D_LIN   = Epetra.Vector(self.PMap)
        else:
            self.myP_dof_min = 0
            self.myP_dof_max = self.P_dof_num - 1

            # Pressure
            self.P_LHS = zeros( self.P_dof_num )
            self.P_RHS = zeros( self.P_dof_num )
            self.P_COR = zeros( self.P_dof_num )
            self.PTEMP = zeros( self.P_dof_num )
            
            # Biot number and Divergence have the same vector size/Map
            self.Bi          = zeros( self.P_dof_num )
            self.DIV_MULT    = zeros( self.P_dof_num )
        
            # Numbers related to the divergence
            self.last_abs_div= zeros( self.P_dof_num )
            self.D_LIN       = zeros( self.P_dof_num )
            self.ABS_D_LIN   = zeros( self.P_dof_num )

        # So the convergence loop runs once before assessing that its done
        self.max_D       = inf
        self.last_max_D  = inf
        
        # All of these variables are per velocity axis (aka per dimension)
        # So, these Lists contain one element for each dimension.
        # self.VEL_EDGE  - The static adjustment that come from PBCS to the dv/dx
        # self.V_LHS - The linear representation of velocity (index is dof #)
        # self.VEL_RHS - The matrices that generate the RHS to the velocity equation

        # Vector shaped Velocities and 
        if self.using_trilinos:            
            self.VMaps = [ Epetra.Map(dof_count, 0, self.Comm) for dof_count in self.vel_dof_nums ]

            self.myV_dof_min = [ vmap.MinMyGID() for vmap in self.VMaps ]
            self.myV_dof_max = [ vmap.MaxMyGID() for vmap in self.VMaps ]

            self.V_LHS = [ Epetra.Vector(vmap) for vmap in self.VMaps ]
            self.V_RHS = [ Epetra.Vector(vmap) for vmap in self.VMaps ]
            self.V_COR = [ Epetra.Vector(vmap) for vmap in self.VMaps ]
            self.VTEMP = [ Epetra.Vector(vmap) for vmap in self.VMaps ]

        else:
            self.myV_dof_min = [ 0 for dofs in self.vel_dof_nums ]
            self.myV_dof_max = [ dofs for dofs in self.vel_dof_nums ]

            self.V_LHS = [ zeros( dof_count ) for dof_count in self.vel_dof_nums ]
            self.V_RHS = [ zeros( dof_count ) for dof_count in self.vel_dof_nums ]
            self.V_COR = [ zeros( dof_count ) for dof_count in self.vel_dof_nums ]
            self.VTEMP = [ zeros( dof_count ) for dof_count in self.vel_dof_nums ]

        # Print pressure DOFs
        for cpu in range(self.cpuCount):
            if cpu == self.myID:
                self.dbprint("My Pressure Degrees of Freedom.  Min:%i Max:%i" 
                             % (self.myP_dof_min, self.myP_dof_max) )
                self.sync()
        
        # Print the velocity degrees of freedom
        # In order!
        for x in range(self.ndim):
            for cpu in range(self.cpuCount):
                if cpu != self.myID:
                    continue
                self.dbprint("My Velocity (%i) DOFs.  Min:%i Max:%i" 
                             % (x, self.myV_dof_min[x], self.myV_dof_max[x]) )
                self.sync()

        # These lists are the ones you have to step through to cover
        # Each cpu's degrees of freedom 
        # DOF Points for P
        self.Get_P_Iterator = lambda : ndimed.pruned_iterator(self.P_dof_grid, 
                                                              self.myP_dof_min, 
                                                              self.myP_dof_max)

        self.Get_V_Iterator = lambda axis: ndimed.pruned_iterator(self.vel_dof_grids[axis], 
                                                                  self.myV_dof_min[axis], 
                                                                  self.myV_dof_max[axis])
        
        # Setup the matrices!
        self.setup_matrices()

        # Setup Matrix Solver
        self.setup_matrix_solver()

        # Get your coffee, were ready to go!
        self.setup_time = time.time() - self.start_time

        # Mark the linear algebra as setup
        self.la_is_setup = True

    def setup_dof_cache_faster(self, s_filename):
        from tables import openFile
        self.dbprint("BIG MODE! Setting up DOF Cache")
        # Should only be run by cpu 0!
        if self.myID != 0: 
            raise RuntimeError("Only thread 0 should setup dof cache!")

        # Open the file to copy S from
        self.dbprint("Opening h5 file to read solid")
        source_h5 = openFile(s_filename)
        S = source_h5.root.geometry.S[:]
        self.dbprint("Success!", 3)
        shape = S.shape
        ndim = len(shape)

        # Write Memory Maps
        self.dbprint("Writing memmap files.")
        shape_map = memmap("shape.mem",    dtype="int64", mode='w+', shape=tuple([ndim]))
        s_memmap  =  memmap("S.mem",       dtype="int64", mode='w+', shape=shape)
        p_memmap  =  memmap("P.mem",       dtype="int64", mode='w+', shape=shape)
        v_memmaps = [memmap("V%i.mem" % x, dtype="int64", mode='w+', shape=shape) for x in range(ndim)]

        self.dbprint("Assigning Values.")

        # Assign the shape
        shape_map[:] = array(shape).astype(int64)[:]
        self.dbprint("Shape Done.")
        
        # Assign S
        s_memmap[:] = S[:].astype(int64)
        self.dbprint("S Done.")

        # Assign P's
        p_memmap[:] = ndim_eq.p_dof(S).astype(int64)
        self.dbprint("P Done.")

        # Assign V's
        for axis, v_mmap in enumerate(v_memmaps):
            v_mmap[:] = ndim_eq.velocity_dof(S, axis).astype(int64)
        self.dbprint("VS Done.")
        self.dbprint("Loaded Maps . . .  Flushing.")
        # Flush All to disk.
        shape_map.flush()
        s_memmap.flush()
        p_memmap.flush()
        [v_mmap.flush() for v_mmap in v_memmaps]

        source_h5.close()
        self.dbprint("\tDone Constructing memory mapped files.")
        
    def import_dof_cache(self):
        self.dbprint("Opening DOF Cache")
        sm = memmap("shape.mem", dtype="int64", mode='r')
        self.dbprint("\tShape Done.", 3)
        self.shape = tuple(sm[:])
        self.ndim = len(self.shape)

        self.S  = memmap("S.mem", dtype="int64", mode='r', shape=self.shape)
        self.dbprint("\tSolid Done.", 3)


        self.P_dof_grid  = memmap("P.mem", dtype="int64", mode='r', shape=self.shape)
        self.dbprint("\tPressure Done.", 3)
        self.vel_dof_grids = [ memmap("V%i.mem" % x, dtype="int64", mode='r', shape=self.shape) for x in range(self.ndim) ]
        self.dbprint("\tVelocities Done.", 3)

    # Get the matrix product between where Mx = b
    # Agnostic to whether we are using Trilinos/scipy
    def mat_mult(self, M, x, b):
        self.dbprint("Matrix Multiply Called", 2)
        if ("scipy" in str(type(M))):
            b[:] = M * x
        elif ("Epetra" in str(type(M))):
            self.sync("Pre MM")
            self.dbprint("Trilinos MM return: %i" % M.Multiply(False, x, b), 3)
            self.sync("Post MM")
        else:
            raise ValueError("Huh?")
 
    # Debug printer . . . kinda neat to watch
    def dbprint(self, string, level = 1):
        '''This is a debug printing routine.
        Level Indicates Urgency:\n
        0:Non-recoverable errors
        1:Information
        2:Details'''
        
        # Call whatever debug printing function you desire!
        time_diff = time.time() - self.start_time
        string = "[%02f][%i/%i] %s" % (time_diff, self.myID + 1, self.cpuCount, string)

        # This is used to output debugging info locally on headless nodes, etc.
        # i.e make a function that cats the strings to a file, or html stream etc.
        if self.dbcallback != None:
            self.dbcallback(string)

        # Screen output regulated on priority, callback not
        if level <= self.printlevel:
            print string

    def setup_matrices(self):
        ###################
        # Create Matrices #
        ###################
        # PM - Pressure Poisson Matrix
        # VM - Pressure Poisson Matrix
        # DM - Velocity Divergence Matrix
        # SM - S-Terms Matrix (RHS to Pressure Poisson)
        # GM - Gradient of P Matrix

        if self.using_trilinos:
            # Square Matrices
            from PyTrilinos.Epetra import CrsMatrix, Copy
            self.PM =   CrsMatrix(Copy, self.PMap, self.max_row_nz)
            self.VM = [ CrsMatrix(Copy, vmap, self.max_row_nz) for vmap in self.VMaps ]
            # Rectangulars
            self.DM = [ CrsMatrix(Copy, self.PMap, 2 ) for vmap in self.VMaps ]
            self.ST = [ CrsMatrix(Copy, self.PMap, 2 ) for vmap in self.VMaps ]
            self.GM = [ CrsMatrix(Copy,      vmap, 2 ) for vmap in self.VMaps ]
        else:
            # Scipy imported here now
            from scipy.sparse import lil_matrix
            # Square Matrices
            self.PM =   lil_matrix( (self.P_dof_num, self.P_dof_num ) )
            self.VM = [ lil_matrix( (dof_count,      dof_count) ) for dof_count in self.vel_dof_nums ]
            # Rectangulars
            self.DM = [ lil_matrix( (self.P_dof_num, dof_count) ) for dof_count in self.vel_dof_nums ]
            self.ST = [ lil_matrix( (self.P_dof_num, dof_count) ) for dof_count in self.vel_dof_nums ]
            self.GM = [ lil_matrix( (dof_count, self.P_dof_num) ) for dof_count in self.vel_dof_nums ]
            
        #################
        # Fill Matrices #
        #################

        # Setup the Pressure Matrix
        self.dbprint("Filling the Pressure Poisson Matrix")
        self._fill_PM()

        # Setup the n Velocity, divergence, and gradient matrices
        for dim in range(self.ndim):
            self.dbprint( "Setting up Velocity Poisson Matrix (%i)" % dim )
            self._fill_VM(dim)

            self.dbprint( "Calculating Divergence/Biot Matrix (%i)" % dim )
            self._fill_DM(dim)

            self.dbprint( "Calculating Gradient Matrix (%i)" % dim )
            self._fill_GM(dim)

        # Due to the symbolic nature of these, they are done
        # All at once
        self.dbprint("Calculating S-Terms")
        self._fill_S()

        ##############################
        # Convert/Finialize Matrices #
        ##############################

        if self.using_trilinos:
            self.dbprint("FillComplete on all Matrices (Waiting For other Threads)")
            self.sync("Pre-FillComplete() of matrices")

            # Square Matrix Implicit in FillComplete (PM, VM's)
            self.PM.FillComplete()
            [ m.FillComplete() for m in self.VM ]

            # Rectagular need maps defined
            [ m.FillComplete(vmap, self.PMap) for m, vmap in zip(self.DM, self.VMaps) ]
            [ m.FillComplete(vmap, self.PMap) for m, vmap in zip(self.ST, self.VMaps) ]
            [ m.FillComplete(self.PMap, vmap) for m, vmap in zip(self.GM, self.VMaps) ]

        else:
            self.dbprint("Converting All To CSR")
            self.PM = self.PM.tocsr()
            self.VM = [m.tocsr() for m in self.VM]
            self.DM = [m.tocsr() for m in self.DM]
            self.ST = [m.tocsr() for m in self.ST]
            self.GM = [m.tocsr() for m in self.GM]



    # scipy spsolve
    def _spsolve_P(self, *args, **kwargs):
        self.P_LHS = self.spsolve( self.PM, self.P_RHS )
    def _spsolve_V(self, dim, *args, **kwargs):
        self.V_LHS[dim] = self.spsolve( self.VM[dim], self.V_RHS[dim] )
        
    # scipy splu
    def _splu_P(self, *args, **kwargs):
        self.P_LHS = self.PM_LU.solve(self.P_RHS)
    def _splu_V(self, dim, *args, **kwargs):
        self.V_LHS[dim] = self.VM_LU[dim].solve(self.V_RHS[dim])

    # pyamg
    def _pyamg_P(self, *args, **kwargs):
        self.P_LHS = self.PM_RUBE.solve(self.P_RHS, tol=1e-10)
    def _pyamg_V(self, dim, *args, **kwargs):
        self.V_LHS[dim] = self.VM_RUBE[dim].solve(rhs, tol=1e-10)

    # pytrilinos
    # I expected it to converge more quickly using the
    # Previoud LHS as the first guess, but its def. 
    # seems to cause problems (zeroing out is the fastest I can find)
    def _trilinos_P(self, *args, **kwargs):
        self.sync("Pre P Solve" )
        self.P_LHS[:] = 0
        tril_return = self.t_PSol.Iterate(5000, 1e-12)
        self.sync("Post P Solve")
        return_string = "Trilinos solve return code: %i" % tril_return
        self.dbprint(return_string, 3)
        if tril_return > 0:
            warnings.warn(return_string)
        elif tril_return < 0:
            self.dbprint("Non-Zero Trilinos Return.  This is BAD!", 0)
            # raise RuntimeError(return_string)

    def _trilinos_V(self, dim, *args, **kwargs):
        self.sync("Pre V Solve (%i)" % dim)
        self.V_LHS[dim][:] = 0
        tril_return = self.t_VSol[dim].Iterate(5000, 1e-12)
        self.sync("Post V Solve (%i)" % dim)
        return_string = "Trilinos solve return code: %i" % tril_return
        self.dbprint(return_string, 3)
        if tril_return > 0:
            warnings.warn(return_string)
        elif tril_return < 0:
            self.dbprint("Non-Zero Trilinos Return.  This is BAD!", 0)
            # raise RuntimeError(return_string)

    def test_matrices(self):
        def vec_nnz(vec):
            if "Trilinos" in str(type(vec)):
                return vec.Norm1()             
            else:
                return abs(vec).sum()

        def mat_nnz(mat):
            if hasattr(mat, "getnnz"):
                return mat.getnnz()
            elif hasattr(mat, "NumGlobalNonzeros"):
                return mat.NumGlobalNonzeros()
            else:
                raise ValueError("Not a recognized Matrix Format")

        self.dbprint("Matrix Non-Zero Check")
        self.dbprint("PM: %i" % mat_nnz(self.PM) )
        for dim in range(self.ndim):
            self.dbprint("VM (%i): %i" % (dim, mat_nnz(self.VM[dim])) )
            self.dbprint("GM (%i): %i" % (dim, mat_nnz(self.GM[dim])) )
            self.dbprint("DM (%i): %i" % (dim, mat_nnz(self.DM[dim])) )
            self.dbprint("ST (%i): %i" % (dim, mat_nnz(self.ST[dim])) )

        self.dbprint("Vector Norm Check")
        self.dbprint("P_COR: %i" % vec_nnz(self.P_COR))
        for dim in range(self.ndim):
            self.dbprint("V_COR (%i): %i" % (dim, vec_nnz(self.V_COR[dim])) )


    def setup_matrix_solver(self):
        ########################    
        # setup solution methods
        ########################    

        # This converts the matrices to a format appropriate to the solution method
        # It presents exposes the functions: SOLVE_P(rhs) and SOLVE_V(dim, rhs)
        # so solution method is transparent to the main iteration loop

        # No Biot number . . . purely derived from
        # Dr. Erdmann's thesis  (for validation/benchmarking)
        if self.method == "nobi":
            self.dbprint("Bi Number disabled.", level=2)
            # Set the ignore Bi, flag and use spsolve
            self.useBi = False
            self.method = 'spsolve'

        # spsolve is the slowest but has no additional memory overhead 
        # (kept around for my shitty laptop) Also good for large domains where
        # splu blows up memory wise (200x200)+
        if self.method == 'spsolve':
            from scipy.sparse.linalg import spsolve
            self.spsolve = spsolve
            self.dbprint("spsolve selected . . . doing nothing", level=2)            
            self.PM    = self.PM.tocsr()

            for dim in range(self.ndim):
                self.VM[dim] = self.VM[dim].tocsr()

            self.SOLVE_P = self._spsolve_P
            self.SOLVE_V = self._spsolve_V

            try:
                from scipy.sparse.linalg import splu
            except:
                warnings.warn("You do not seem to have UMFpack installed; the use of spsolve will be _very_ slow!")

        # This sparse LU decomposition. Good for systems with any version of scipy
        # Generally faster for anything above 100x100
        # Major memory hog for large systems (200x200 >200Mb)
        # Don't even try this on a 3d . . .
        elif self.method == 'splu':
            from scipy.sparse.linalg import splu
            self.dbprint("SPLU'ing Matrices", level=2)
            self.dbprint("\t Pressure.tocsc()", level=2)
            self.PM    = self.PM.tocsc()
            self.dbprint("\t Pressure", level=2)
            self.PM_LU = splu(self.PM)

            self.VM_LU = [None] * self.ndim
            for dim in range(self.ndim):
                self.dbprint("\t Velocity %i tocsc()" % dim, level=2)
                self.VM[dim] = self.VM[dim].tocsc()
                self.dbprint("\t Velocity %i -splu" % dim, level=2)
                self.VM_LU[dim] = splu(self.VM[dim])

            self.SOLVE_P = self._splu_P            
            self.SOLVE_V = self._splu_V

        # Relies on pyamg for AMG solvers. Should be wicked fast breaks for large grids?
        # Need to add adjustable parameters for large and small systems . . .
        # TODO: Currently crashes for large systems?
        elif self.method == 'ruge':
            import pyamg
            self.dbprint("Setting up ruge_stuben_solver(s)", level=2)
            self.dbprint("\t Pressure.tocsr()", level=2)
            self.PM    = self.PM.tocsr()
            self.dbprint("\t Pressure Stuben", level=2)
            self.PM_RUBE = pyamg.ruge_stuben_solver( self.PM )


            self.VM_RUBE = [None] * self.ndim
            for dim in range(self.ndim):
                self.dbprint("\t Velocity %i tocsr()" % dim, level=2)
                self.VM[dim] = self.VM[dim].tocsr()
                self.dbprint("\t Velocity %i -rube" % dim, level=2)
                self.VM_RUBE[dim] = pyamg.ruge_stuben_solver( self.VM[dim] )
            self.SOLVE_P = _pyamg_P
            self.SOLVE_V = _pyamg_V

        elif self.method == "trilinos":
            self.dbprint("Using Trilinos!", level=2)

            # Trilinos Solving Options:
            MLList = { "max levels" : 10,
                       "output" : 10,
                       "smoother: pre or post" : "both",
                       "smoother: type" : "Chebyshev",
                       "aggregation: type" : "Uncoupled",
                       "coarse: type" : "Amesos-KLU" }
            
            # import ML for the preconditioners
            from PyTrilinos import ML, AztecOO
            # Compute the preconditioner . . .
            self.t_PCond = ML.MultiLevelPreconditioner(self.PM, False)
            self.t_PCond.SetParameterList(MLList)
            self.t_PCond.ComputePreconditioner()

            # Setup the Pressure Poisson solver
            self.t_PSol = AztecOO.AztecOO(self.PM, self.P_LHS, self.P_RHS)
            self.t_PSol.SetPrecOperator(self.t_PCond)
            self.t_PSol.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_gmres)
            self.t_PSol.SetAztecOption(AztecOO.AZ_output, 64)

            self.t_VCon = [ML.MultiLevelPreconditioner(mat, False) for mat in self.VM]
            self.t_VSol = [AztecOO.AztecOO(self.VM[dim], self.V_LHS[dim], self.V_RHS[dim]) for dim in range(self.ndim)]
            for prec, sol in zip(self.t_VCon, self.t_VSol):
                # Compute the preconditioner . . .
                prec.SetParameterList(MLList)
                prec.ComputePreconditioner()

                # Apply it and setup the Velocity Poisson solver for each axis
                sol.SetPrecOperator(prec)
                sol.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_gmres)
                sol.SetAztecOption(AztecOO.AZ_output, 64)

            self.SOLVE_P = self._trilinos_P
            self.SOLVE_V = self._trilinos_V
        elif (self.method == "trilinos") and not have_trilinos:
            self.dbprint("No Trilinos Detected!  Abort!", level = 0)
            raise ValueError("No Trilinos Detected!  Abort!")
        else:
            self.dbprint("Solver type '%s' not recognized!!!!" % self.method, level = 0)
            raise ValueError("Solver type '%s' not recognized!!!!" % self.method)

    def setup_bc(self):
        self.dbprint("Calculating Periodic Correction Vectors", 2)

        self.dbprint("\tV - RHS Correction", 3)
        #############################
        # Velocity RHS's Correction #
        #############################
        for dim in range(self.ndim):
            for point in self.Get_V_Iterator(dim):
                # This decides when the RHS of the velocity equation 
                # requires a addition of the pressure drop
                # This correction comes from the PBC's and the gradient of the pressure (hence the negative)
                if (point[dim] == 0): 
                    vdof = self.nvd(dim, point)
                    # Map to trilinos value
                    if self.using_trilinos:
                        vdof = self.VMaps[dim].LID(vdof)

                    self.V_COR[dim][vdof] -= self.dP[dim]

        ###########################
        # Pressure RHS Correction #
        ###########################
        for point in self.Get_P_Iterator():
            pdof = self.pdp(point)
            
            # This Pins the pressure solution for the 0th DOF
            if pdof == 0:
                if self.using_trilinos:
                    pdof = self.PMap.LID(pdof)

                self.P_COR[pdof] = 1
                continue

            # Now check to see if were at the edge 
            # (have to add a constant to the RHS)
            # Check each dimension
            point_pressure_correction = 0
            for dim in range(self.ndim):
                # Shift the Point Forward in this axis
                back_point = list(point)
                back_point[dim] -= 1
                test_back = tuple(back_point)

                # Looking back yeilds negative dP
                # If point is at the near edge AND
                # Isn't adjacent to solid (normal to that edge)
                if (point[dim] == 0) and (self.pdp(test_back) != -1):
                    point_pressure_correction -= self.dP[dim]

                # Shift the Point Backward in this axis
                shifted_point = list(point)
                shifted_point[dim] += 1
                fore_point = tuple(shifted_point)

                # Looking back yeilds posative dP
                # If point is at the far edge AND
                # Isn't adjacent to solid (normal to that edge)
                if (point[dim] == self.shape[dim] - 1) and (self.pdp(fore_point) != -1):
                    point_pressure_correction += self.dP[dim]

            # If using Trilinos, de-reference to local indexing
            if self.using_trilinos:
                # pdof = self.PMap.LID(pdof)
                self.P_COR.SumIntoGlobalValue(pdof, 0, point_pressure_correction)
            else:
                self.P_COR[pdof] += point_pressure_correction

        # Flag the BC's as setup
        self.bc_is_setup = True
            
    def _fill_PM(self):
        # ndimed.iter_grid only gits points for each proc.
        for point in self.Get_P_Iterator():
            dof = self.pdp(point)
            
            # #ignore non-degrees of freedom (no longer necessary?)
            # if dof < 0:
            #     continue

            # Pin the solution to obtain
            # a unique (non-singular) solution
            if dof == 0:
                self.PM[dof,dof] = 1
                continue
            
            # The trace will be -2 * the number of dimensions
            center_value = -2 * self.ndim

            # The 'dof' is the row, and we gather the rows and cols to facilitate trilinos/scipy
            col = []
            val = []
            
            # Oscillate one in each direction
            for pp in ndimed.perturb(point):
                # Get the DOF number
                test_dof = self.pdp(pp)

                # If it is a dof . . .
                if test_dof != -1:
                    col.append(test_dof)
                    val.append(1)
                else:
                    center_value    += 1                

            # Stupid Warning, I don't think it is necessary any more . . .
            if center_value == 0:
                self.dbprint("Something bad probably just happened! %s" % str(point), 1)

            # Add the center point to the list . . .
            col.append(dof)
            val.append(center_value)

            # Populate the matrix
            for c, v in zip(col, val):
                self.PM[dof,c] = v


    def _fill_VM( self, axis ):
        #Define self.UM
        for point in self.Get_V_Iterator(axis):
            dof = self.nvd(axis, point)
            # #ignore non-degrees of freedom
            # if dof < 0:
	    #     continue
            
            # The trace will be -2 * ndim
            center_val = -2 * self.ndim

            col = []
            val = []
            for pp in ndimed.perturb(point):
                test_dof = self.nvd(axis, pp)
                if   test_dof >= 0:
                    col.append(test_dof)
                    val.append(1)
                elif test_dof ==-3:
                    center_val -= 1

            # Add the center value in the list of stuff to add
            col.append(dof)
            val.append(center_val)

            # Populate the matrix
            for c, v in zip(col, val):
                self.VM[axis][dof,c] = v

    def _fill_DM(self, dim):
        # Matrix associated with this dimension
        this_DM = self.DM[dim]
        rolled_v_dof_grid = roll(self.vel_dof_grids[dim], -1, axis=dim)

        # Iterate over the grid
        for point in self.Get_P_Iterator():
            # P Degree of freedom
            pd = self.pdp(point)
            
            # Neg is flowing in
            dof = self.vel_dof_grids[dim][point]
            if dof >= 0:
                if self.using_trilinos:
                    this_DM.InsertGlobalValues(pd, [-1.], [dof])
                else:
                    this_DM[pd, dof] = -1
        
            # Positive is flowing out
            dof = rolled_v_dof_grid[point]
            if dof >= 0:
                if self.using_trilinos:
                    this_DM.InsertGlobalValues(pd, [1.], [dof])
                else:
                    this_DM[pd, dof] = 1
        
    def _fill_S(self):
        # Iterate over the grid
        for point in self.Get_P_Iterator():
            # Find the DOF for the current pressure cell
            pd = self.pdp(point)

            # No Source Term for pinned point
            if pd == 0:
                continue

            # If completely liquid . . . Laplace equation . . . no source
            cfg_code = ndim_eq.config_code(self.S, point)

            # assert ndim_eq.config_code(self.S, point) == ndim_eq.new_config_code(self.S, point)

            if cfg_code == 0:
                continue
            
            # This will return a list of n equations
            point_equations = ndim_eq.s_term(self.P_dof_grid, self.vel_dof_grids, point)

            # For each dimension (equation) populate the corresponding matrix row
            for dim, eq in enumerate(point_equations):
                for dofn, coeff in eq.iteritems():
                    if self.using_trilinos:
                        self.ST[dim].InsertGlobalValues(pd, [coeff], [dofn])
                    else:
                        self.ST[dim][pd, dofn] = coeff

    def _fill_GM(self, dim):
        rolled_p = roll(self.P_dof_grid, 1, axis=dim)

        vals_added = 0
        for point in ndimed.full_iter_grid(self.P_dof_grid):
            vel_dof = self.vel_dof_grids[dim][point]
       
            # Still Necessary!
            if vel_dof < 0:
                continue

            if self.using_trilinos and (not self.VMaps[dim].MyGID(vel_dof)):
                continue

            p1 = self.P_dof_grid[point]
            if p1 >= 0:
                if self.using_trilinos:
                    self.GM[dim].InsertGlobalValues(vel_dof, [1.], [p1])
                else:
                    self.GM[dim][vel_dof, p1] = 1
                vals_added += 1

            p2 = rolled_p[point]
            if p2 >= 0:
                if self.using_trilinos:
                    self.GM[dim].InsertGlobalValues(vel_dof, [-1.], [p2])
                else:
                    self.GM[dim][vel_dof, p2] = -1
                vals_added += 1

    def update_D(self):
        # Track the previous abs values
        self.last_abs_div[:] = self.ABS_D_LIN

        # If divergence has halted .  . . something borked
        # If this happend 5x in a row . . .
        if self.last_max_D == self.max_D and self.max_D != inf:
            self.bork_count += 1
            if self.bork_count >= 5: raise ValueError("WTF")
        else:
            self.bork_count = 0
        
        self.last_max_D = self.max_D

        # Zero out the divergence
        self.D_LIN *= 0

        # Re-accumulate it from the various flow dimensions
        for d in range(self.ndim):
            self.mat_mult(self.DM[d], self.V_LHS[d], self.PTEMP)
            self.D_LIN[:] += self.PTEMP
        
        # This is ok for all dimensions as 
        #     edge -> sa 
        #     sa   -> vol  
        # so h factor is constant 
        self.D_LIN /= self.h
        # Calculate. the max value (convergence test)
        
        # Abs divergence vector (for bi optimization)
        if self.using_trilinos:
            self.ABS_D_LIN.Abs(self.D_LIN)
            self.max_D = self.ABS_D_LIN.MaxValue()
        else:
            self.ABS_D_LIN = abs(self.D_LIN)
            self.max_D = self.ABS_D_LIN.max()        

    def monolithic_solve(self, method = "default"):
        self.force_la_setup()
        self.force_bc_setup()

        self.dbprint( "Starting Monolithic Solve", 1)
        from scipy.sparse import lil_matrix
        self.MM = lil_matrix((self.dof_number, self.dof_number))

        # TODO: using coo you could just add offsets to all the matrices 
        # involved and cat them together making the setup 
        # faster and trilinos compliant etc . . .

        # DOF numbers
        pdof = self.P_dof_num
        vns = self.vel_dof_nums #[ndim]

        self.dbprint("\tAdding Pressure Laplace", 2)
        # Laplace Matrices
        # PM-L
        xo, yo = (0,0)
        self.PM
        xi, yi = self.PM.nonzero()
        for x, y in zip(xi, yi):
            self.MM[x,y] = self.PM[x, y]

        xo, yo = self.PM.shape
        # VM-L
        self.VM
        for dim in range(self.ndim):
            self.dbprint("\tAdding Velocity Laplace (%i)" % dim, 2)
            vel_matrix = self.VM[dim]
            xi, yi = vel_matrix.nonzero()
            for x, y in zip(xi, yi):
                self.MM[xo + x, yo + y] = vel_matrix[x, y]
            xo += vel_matrix.shape[0]
            yo += vel_matrix.shape[1]

        # # Gradient Matrices
        xo = self.PM.shape[0]
        yo = 0
        for dim in range(self.ndim):
            self.dbprint("\tAdding Gradient (%i)" % dim, 2)
            grad_mat = self.GM[dim]
            xi, yi = grad_mat.nonzero()
            for x, y in zip(xi, yi):
                self.MM[x + xo, y + yo] = - grad_mat[x, y] * self.h
            xo += self.VM[dim].shape[0]

        # # S-term Matrices
        xo = 0
        yo = self.PM.shape[0]
        for dim in range(self.ndim):
            self.dbprint("\tAdding S-terms (%i)"%dim,2)
            s_mat = self.ST[dim]
            xi, yi = s_mat.nonzero()
            for x, y in zip(xi, yi):
                self.MM[x + xo, y + yo] = -s_mat[x, y] / self.h
            yo += self.VM[dim].shape[0]

        self.dbprint("\tAssembling RHS",2)
        self.MM_rhs = zeros(self.P_dof_num)
        self.MM_rhs[0:len(self.P_COR)] = self.P_COR

        for dim in range(self.ndim):
            self.MM_rhs = concatenate( (self.MM_rhs, self.V_COR[dim] * self.h) )

        # self.dbprint("DEBUG:TODO, Committing matrix and rhs to disk")
        # from sparse_to_h5 import storeSparseProblem
        # storeSparseProblem(self.MM, self.MM_rhs, "BigMatrixStorage.h5")

        self.dbprint("Converting to CSR",2)
        self.MM = self.MM.tocsr()
        self.dbprint("Solving . . .", 1)

        self.solve_start = time.time()
        if self.method == "spsolve" or self.method == "nobi":
            from scipy.sparse.linalg import spsolve
            ans = spsolve(self.MM, self.MM_rhs)
        elif self.method == "bicgstab":
            ans = self.bicgstab(self.MM, self.MM_rhs)
        elif self.method == "ruge":
            from pyamg import ruge_stuben_solver
            self.dbprint("Setting up ruge_stuben_solver.", level=2)
            self.rss = self.ruge_stuben_solver( self.MM, max_levels=2)
            self.dbprint(self.rss)
            ans = self.rss.solve(self.MM_rhs, tol=1e-10)
        else:
            dbprint("Solver method not supported!",0)
            raise ValueError
        self.solve_time = time.time() - self.solve_start

        self.P_LHS = ans[0:self.P_dof_num]

        current_offset = len(self.P_LHS)
        for dim in range(self.ndim):
            self.V_LHS[dim] = ans[current_offset:current_offset + self.vel_dof_nums[dim]]
            current_offset += self.vel_dof_nums[dim]

    # The Bi number based acceleration
    # TODO: cleanup this, make it actual Bi instead of multiple of the divergence
    def update_Bi(self, dn_mult = 0.05, up_mult = 1.001, start_Bi=0.0000005, starting_I = 10 ):
        '''This routine does a simple optimization on the Biot number of the Pressure Solution
        i.e. adjusting of the element wise paramaters accelerates convergence by reducing
        the system stiffness.  If the cell-wise divergence goes up, it increases the stringency of BC's at that point
        otherwise it lets it loosen slightly.
        The default paramaters are _extremely_ conservative, but more aggressive setting can vastly accelerate convergence. 
        Selection of understable paramaters can lead to irreversable divergence, so adjust with care'''
        self.dbprint("Starting Bi Optimization")
        mx = 1
        if   self.I  < starting_I:
            self.DIV_MULT[:] = 0
        elif self.I == starting_I:
            self.DIV_MULT[:] = start_Bi
        else:
            # True for dof's who have a lower divergence this round than last
            # Epetra Vectors dont Support fancy slicing, so
            lower = array(self.ABS_D_LIN) < array(self.last_abs_div)

            dm_copy = array(self.DIV_MULT)

            inc_count = sum(1 * lower)
            self.dbprint("Bi DOF Count Increased %i" % inc_count, 2 )
    
            # Some Bi numbers go up
            dm_copy[lower] *= up_mult
            dm_copy[logical_not(lower)] *= dn_mult

            # Bi Capped at some value
            dm_copy[dm_copy > mx] = mx

            self.DIV_MULT[:] = dm_copy


        me = mean(self.DIV_MULT)
        mi = min(self.DIV_MULT)
        mx = max(self.DIV_MULT)
        
        dbinfo = "Bi Optimization Finished - Local Info Mean:%e Min:%e Max:%e" % (me, mi, mx)
        self.dbprint( dbinfo, 2 )

    def sync(self, place=""):
        if self.using_trilinos:
            self.dbprint("Trilinos Thread Sync - %s" % place, 3)
            self.Comm.Barrier()
            self.dbprint("\tDone.", 3)
        
    def iterate(self):
        '''This is the main iteration loop that converges the system.'''
        # Make sure we are setup properly
        self.force_la_setup()
        self.force_bc_setup()
        
        self.sync("Beginning of Iteration")
        ###############################
        # Solve the Pressure Equation #
        ###############################
        self.dbprint("\tCalculating P RHS", level = 2)
        # This is the Bi contribution
        self.P_RHS[:] = self.DIV_MULT * self.D_LIN
        for d in range(self.ndim):
            self.mat_mult(self.ST[d], self.V_LHS[d], self.PTEMP)
            self.P_RHS[:] += self.PTEMP

        # Divide by h and add PBC pressure corrections
        self.P_RHS[:] = (self.P_RHS / self.h) + self.P_COR

        # Solve the pressure Eq.
        self.dbprint("\tSolving P", level = 2)
        self.SOLVE_P()

        ################################
        # Solve the Velocity Equations #
        ################################
        for dim in range(self.ndim):
            # Setup the RHS to the Velocity equation
            self.dbprint("\tCalculating Grad P (V RHS - %i)" % dim , level = 2)
            self.mat_mult(self.GM[dim], self.P_LHS, self.VTEMP[dim])
            self.V_RHS[dim][:] = (self.VTEMP[dim] + self.V_COR[dim]) * self.h

            # Solve the velocity Equation in the n'th dimension
            self.dbprint("\tSolving V - %i" % dim , level = 2)
            self.SOLVE_V(dim)
        # Sync
        self.sync("End of Iteration!")
        # Increment the iteration count
        self.I += 1

    def regrid(self):
        '''Convert the linear-algebra formed vectors back into familiar field varibles'''
        # Make the 2d/3d arrays to hold the results
        self.P = zeros(self.S.shape, dtype=float64)

        # One face centered velocity grid for each Velocity Axis
        self.V_GRIDS = [ zeros( self.shape) for dim in range(self.ndim) ]

        # Convenience handles (P, u, v, w, x)
        for name, dim in zip(['u','v','w','x'], range(self.ndim)):
            setattr(self, name, self.V_GRIDS[dim])

        # Put them back in their respective arrays
        self.assign_P_to_obj(self.P)
        for axis in range(self.ndim):
            self.assign_V_to_obj( axis, self.V_GRIDS[axis] )
    
    def ungrid(self):
        '''Take the grid-wise P, u, v, etc and put into the 
        linear forms used during matrix solution
        Useful if you want to seed a solution into the solver'''

        # There is a classier way to do this with "take" from numpy, but isn't Trilinos safe
        for point in self.P_points_list:
            pdof = self.pdp(point)
            self.P_LHS[pdof] = self.P[point]

        for axis in range(self.ndim):
            for point in self.V_points_list[axis]:
                dof = self.nvd( axis, point )
                self.V_LHS[axis][dof] = self.V_GRIDS[axis][point]

    def assign_P_to_obj(self, obj):
        '''These two functions are used in saving results
        This implementation makes the core not reliant on tables
        i.e. hdf5 can open a file, and assign only DOFs from a cerain
        CPU at a time.'''
        
        for point in self.Get_P_Iterator():
            dof = self.pdp(point)
            if self.using_trilinos:
                dof = self.PMap.LID(dof)
            obj[point] = self.P_LHS[dof]
    def assign_V_to_obj(self, axis, obj):
        '''These two functions are used in saving results
        This implementation makes the core not reliant on tables
        i.e. hdf5 can open a file, and assign only DOFs from a cerain
        CPU at a time.'''

        for point in self.Get_V_Iterator(axis):
            dof = self.nvd(axis, point)
            if self.using_trilinos:
                dof = self.VMaps[axis].LID(dof)
            obj[point] = self.V_LHS[axis][dof]

    def converge(self, stopping_div = 1.0e-8, max_iter = inf, use_biot = True, printing = 0):
        '''Run the iterative version of the solver until the first of input stopping criteria are met.'''
        # Make sure we are setup properly
        self.force_la_setup()
        self.force_bc_setup()

	self.update_D()
	# If the current solution is valid
        # (and it isn't the first iteration), break out
	if self.I != 0 and self.max_D < stopping_div:
            return

        # Start the timer
        self.solve_start = time.time()
	while True:
	    # Perform an iteration
	    self.iterate()

            self.dbprint("\tUpdating Divergence" , level = 2)
            # Update the divergence and max divergence
            self.update_D()
        
            # Update Bi
            if use_biot: self.update_Bi()

	    # Print some status Crap
            self.dbprint("Iteration:%i, Max Divergence:%e"  % ( self.I, self.max_D ) )
            
	    # If it is time to break, do so
	    if self.max_D < stopping_div:
                self.solve_time = time.time() - self.solve_start
		break

            # Alternative breaking criteria: exceeding iteration count
            if self.I >= max_iter:
                raise ValueError("Max iterations exceeded")

    def getMetaDict(self):
        '''This function returns a dictionary of metadata information about the solver.
        It is only 'valid' after one or more solution methods have been used'''
        ret_dict = {"Iteration_Count":self.I,
                    "P_and_V_DOF_Number":self.dof_number,
                    "Setup_Time":self.setup_time,
                    "Converge_Time":self.solve_time,
                    "Matrix_Solver":self.method,
                    "Total_Time":(self.setup_time + self.solve_time),
                    "Identity":gethostname()}
        return ret_dict
            
    def nuke_all_trilinos(self):
        '''Trilinos and the iPython task-client dont clean up swig-attached memory correctly.
        Why?  Who knows?  But call this to avoid unsightly memory leaks.'''
        self.sync("Pre-Nuke Sync")
        if not self.using_trilinos:
            return

        # Epetra Vectors
        del self.PMap
        del self.P_LHS
        del self.P_RHS
        del self.P_COR
        del self.PTEMP
        del self.Bi      
        del self.DIV_MULT
        del self.last_abs_div
        del self.D_LIN       
        del self.ABS_D_LIN   

        for x in range(3):
            del self.VMaps[0]
            del self.V_LHS[0]
            del self.V_RHS[0]
            del self.V_COR[0]
            del self.VTEMP[0]

        # Epetra Matrices
        del self.PM
        for x in range(3):
            del self.VM[0]
            del self.DM[0]
            del self.ST[0]
            del self.GM[0]
