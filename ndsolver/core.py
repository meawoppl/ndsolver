import logging
import warnings
from numpy import ( all, allclose, arange, array, average, concatenate,
                    c_, cumsum, dot, float64, inf,  int64, logical_and,
                    logical_not, logical_or, mean, memmap, ones, r_, roll, sqrt,
                    take, where, zeros, zeros_like )
import time
from socket import gethostname

# libs that do the symbolic S terms etc.
from .symbolic import ndim_eq, ndimed

logger = logging.getLogger(__name__)

# Scipy and Solvers are imported as needed to keep memory profile low!

class Solver():
    '''This is the setup method of the solver.  It instantiates a class
    with methods central to solving low Reynolds number flows.

    Required Arguments:
    'solid' is an n-dimensional array describing geometry of the problem
    'dP' is list/tuple of pressure difference across
    the n-th dimension of the domain"
    '''
    def __init__(self, solid_or_filename, dP, sol_method="default"):
        # Log the starting time
        self.start_time = time.time()

        # Single-threaded solver
        self.myID = 0
        self.cpuCount = 1

        logger.debug("Solver Instantiated")
        
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
            logger.info("Waiting for cached file to flush to disk.")
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
        logger.debug("Starting Setup Routine")
        # Everything is ND wrt.  the 0th axis
        self.h = 1./self.shape[0]

        ################# 
        # Stupid checks #
        #################        
        if self.ndim != len(self.dP):
            raise ValueError(f"Solid Array and Pressure Drop do not have matching dimensions:\n\tself.ndim:{self.ndim}\n\t{self.dP}")

        #################
        # FYI Printouts #
        #################
        # DOF Number total
        self.dof_number = sum(self.vel_dof_nums)  + self.P_dof_num
       
        # print some useful DOF debugging information:
        logger.info(f"Degree of freedom count: {self.dof_number}")
        logger.info(f"\tPressure: {self.P_dof_num}")
        for dim in range(self.ndim):
            logger.info(f"\tVelocity {dim}: {self.vel_dof_nums[dim]}")

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

        # Vector shaped Velocities
        self.myV_dof_min = [ 0 for dofs in self.vel_dof_nums ]
        self.myV_dof_max = [ dofs for dofs in self.vel_dof_nums ]

        self.V_LHS = [ zeros( dof_count ) for dof_count in self.vel_dof_nums ]
        self.V_RHS = [ zeros( dof_count ) for dof_count in self.vel_dof_nums ]
        self.V_COR = [ zeros( dof_count ) for dof_count in self.vel_dof_nums ]
        self.VTEMP = [ zeros( dof_count ) for dof_count in self.vel_dof_nums ]

        # Print pressure DOFs
        for cpu in range(self.cpuCount):
            if cpu == self.myID:
                logger.info(f"My Pressure Degrees of Freedom.  Min:{self.myP_dof_min} Max:{self.myP_dof_max}")
                self.sync()
        
        # Print the velocity degrees of freedom
        # In order!
        for x in range(self.ndim):
            for cpu in range(self.cpuCount):
                if cpu != self.myID:
                    continue
                logger.info(f"My Velocity ({x}) DOFs.  Min:{self.myV_dof_min[x]} Max:{self.myV_dof_max[x]}")
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
        from tables import open_file
        logger.info("BIG MODE! Setting up DOF Cache")
        # Should only be run by cpu 0!
        if self.myID != 0:
            raise RuntimeError("Only thread 0 should setup dof cache!")

        # Open the file to copy S from
        logger.info("Opening h5 file to read solid")
        source_h5 = open_file(s_filename)
        S = source_h5.root.geometry.S[:]
        logger.debug("Success!")
        shape = S.shape
        ndim = len(shape)

        # Write Memory Maps
        logger.info("Writing memmap files.")
        shape_map = memmap("shape.mem",    dtype="int64", mode='w+', shape=tuple([ndim]))
        s_memmap  =  memmap("S.mem",       dtype="int64", mode='w+', shape=shape)
        p_memmap  =  memmap("P.mem",       dtype="int64", mode='w+', shape=shape)
        v_memmaps = [memmap(f"V{x}.mem", dtype="int64", mode='w+', shape=shape) for x in range(ndim)]

        logger.info("Assigning Values.")

        # Assign the shape
        shape_map[:] = array(shape).astype(int64)[:]
        logger.info("Shape Done.")
        
        # Assign S
        s_memmap[:] = S[:].astype(int64)
        logger.info("S Done.")

        # Assign P's
        p_memmap[:] = ndim_eq.p_dof(S).astype(int64)
        logger.info("P Done.")

        # Assign V's
        for axis, v_mmap in enumerate(v_memmaps):
            v_mmap[:] = ndim_eq.velocity_dof(S, axis).astype(int64)
        logger.info("VS Done.")
        logger.info("Loaded Maps . . .  Flushing.")
        # Flush All to disk.
        shape_map.flush()
        s_memmap.flush()
        p_memmap.flush()
        [v_mmap.flush() for v_mmap in v_memmaps]

        source_h5.close()
        logger.info("\tDone Constructing memory mapped files.")
        
    def import_dof_cache(self):
        logger.info("Opening DOF Cache")
        sm = memmap("shape.mem", dtype="int64", mode='r')
        logger.debug("\tShape Done.")
        self.shape = tuple(sm[:])
        self.ndim = len(self.shape)

        self.S  = memmap("S.mem", dtype="int64", mode='r', shape=self.shape)
        logger.debug("\tSolid Done.")


        self.P_dof_grid  = memmap("P.mem", dtype="int64", mode='r', shape=self.shape)
        logger.debug("\tPressure Done.")
        self.vel_dof_grids = [ memmap(f"V{x}.mem", dtype="int64", mode='r', shape=self.shape) for x in range(self.ndim) ]
        logger.debug("\tVelocities Done.")

    # Get the matrix product where Mx = b
    def mat_mult(self, M, x, b):
        logger.debug("Matrix Multiply Called")
        b[:] = M * x
 

    def setup_matrices(self):
        ###################
        # Create Matrices #
        ###################
        # PM - Pressure Poisson Matrix
        # VM - Pressure Poisson Matrix
        # DM - Velocity Divergence Matrix
        # SM - S-Terms Matrix (RHS to Pressure Poisson)
        # GM - Gradient of P Matrix

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
        logger.info("Filling the Pressure Poisson Matrix")
        self._fill_PM()

        # Setup the n Velocity, divergence, and gradient matrices
        for dim in range(self.ndim):
            logger.info(f"Setting up Velocity Poisson Matrix ({dim})")
            self._fill_VM(dim)

            logger.info(f"Calculating Divergence/Biot Matrix ({dim})")
            self._fill_DM(dim)

            logger.info(f"Calculating Gradient Matrix ({dim})")
            self._fill_GM(dim)

        # Due to the symbolic nature of these, they are done
        # All at once
        logger.info("Calculating S-Terms")
        self._fill_S()

        ##############################
        # Convert/Finialize Matrices #
        ##############################
        logger.info("Converting All To CSR")
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

    # scipy bicgstab
    def _bicgstab_P(self, *args, **kwargs):
        result, info = self.bicgstab(self.PM, self.P_RHS)
        if info != 0:
            logger.warning(f"bicgstab P solve did not converge (info={info})")
        self.P_LHS = result
    def _bicgstab_V(self, dim, *args, **kwargs):
        result, info = self.bicgstab(self.VM[dim], self.V_RHS[dim])
        if info != 0:
            logger.warning(f"bicgstab V[{dim}] solve did not converge (info={info})")
        self.V_LHS[dim] = result

    def test_matrices(self):
        def vec_nnz(vec):
            return abs(vec).sum()

        def mat_nnz(mat):
            return mat.getnnz()

        logger.info("Matrix Non-Zero Check")
        logger.info(f"PM: {mat_nnz(self.PM)}")
        for dim in range(self.ndim):
            logger.info(f"VM ({dim}): {mat_nnz(self.VM[dim])}")
            logger.info(f"GM ({dim}): {mat_nnz(self.GM[dim])}")
            logger.info(f"DM ({dim}): {mat_nnz(self.DM[dim])}")
            logger.info(f"ST ({dim}): {mat_nnz(self.ST[dim])}")

        logger.info("Vector Norm Check")
        logger.info(f"P_COR: {vec_nnz(self.P_COR)}")
        for dim in range(self.ndim):
            logger.info(f"V_COR ({dim}): {vec_nnz(self.V_COR[dim])}")


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
            logger.debug("Bi Number disabled.")
            # Set the ignore Bi, flag and use spsolve
            self.useBi = False
            self.method = 'spsolve'

        # spsolve is the slowest but has no additional memory overhead 
        # (kept around for my shitty laptop) Also good for large domains where
        # splu blows up memory wise (200x200)+
        if self.method == 'spsolve':
            from scipy.sparse.linalg import spsolve
            self.spsolve = spsolve
            logger.debug("spsolve selected . . . doing nothing")            
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
            logger.debug("SPLU'ing Matrices")
            logger.info("\t Pressure.tocsc()", level=2)
            self.PM    = self.PM.tocsc()
            logger.debug("\t Pressure")
            self.PM_LU = splu(self.PM)

            self.VM_LU = [None] * self.ndim
            for dim in range(self.ndim):
                logger.debug(f"\t Velocity {dim} tocsc()")
                self.VM[dim] = self.VM[dim].tocsc()
                logger.debug(f"\t Velocity {dim} -splu")
                self.VM_LU[dim] = splu(self.VM[dim])

            self.SOLVE_P = self._splu_P
            self.SOLVE_V = self._splu_V

        # Iterative BiCGSTAB solver - good for large systems
        elif self.method == 'bicgstab':
            from scipy.sparse.linalg import bicgstab
            self.bicgstab = bicgstab
            logger.debug("bicgstab selected")
            self.PM = self.PM.tocsr()

            for dim in range(self.ndim):
                self.VM[dim] = self.VM[dim].tocsr()

            self.SOLVE_P = self._bicgstab_P
            self.SOLVE_V = self._bicgstab_V

        else:
            logger.warning(f"Solver type '{self.method}' not recognized!!!!")
            raise ValueError(f"Solver type '{self.method}' not recognized!!!!")

    def setup_bc(self):
        logger.debug("Calculating Periodic Correction Vectors")

        logger.debug("\tV - RHS Correction")
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
                    self.V_COR[dim][vdof] -= self.dP[dim]

        ###########################
        # Pressure RHS Correction #
        ###########################
        for point in self.Get_P_Iterator():
            pdof = self.pdp(point)

            # This Pins the pressure solution for the 0th DOF
            if pdof == 0:
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

            # The 'dof' is the row, and we gather the rows and cols
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
                logger.warning(f"Something bad probably just happened! {point}")

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
                this_DM[pd, dof] = -1

            # Positive is flowing out
            dof = rolled_v_dof_grid[point]
            if dof >= 0:
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
                for dofn, coeff in eq.items():
                    self.ST[dim][pd, dofn] = coeff

    def _fill_GM(self, dim):
        rolled_p = roll(self.P_dof_grid, 1, axis=dim)

        vals_added = 0
        for point in ndimed.full_iter_grid(self.P_dof_grid):
            vel_dof = self.vel_dof_grids[dim][point]

            # Still Necessary!
            if vel_dof < 0:
                continue

            p1 = self.P_dof_grid[point]
            if p1 >= 0:
                self.GM[dim][vel_dof, p1] = 1
                vals_added += 1

            p2 = rolled_p[point]
            if p2 >= 0:
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
            self.D_LIN[:] = self.D_LIN[:] + self.PTEMP
        
        # This is ok for all dimensions as 
        #     edge -> sa 
        #     sa   -> vol  
        # so h factor is constant
        self.D_LIN /= self.h
        # Calculate the max value (convergence test)

        # Abs divergence vector (for bi optimization)
        self.ABS_D_LIN = abs(self.D_LIN)
        self.max_D = self.ABS_D_LIN.max()        

    def monolithic_solve(self, method = "default"):
        self.force_la_setup()
        self.force_bc_setup()

        logger.info( "Starting Monolithic Solve")
        from scipy.sparse import lil_matrix
        self.MM = lil_matrix((self.dof_number, self.dof_number))

        # TODO: using coo you could just add offsets to all the matrices 
        # involved and cat them together making the setup 
        # faster etc . . .

        # DOF numbers
        pdof = self.P_dof_num
        vns = self.vel_dof_nums #[ndim]

        logger.debug("\tAdding Pressure Laplace")
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
            logger.debug(f"\tAdding Velocity Laplace ({dim})")
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
            logger.debug(f"\tAdding Gradient ({dim})")
            grad_mat = self.GM[dim]
            xi, yi = grad_mat.nonzero()
            for x, y in zip(xi, yi):
                self.MM[x + xo, y + yo] = - grad_mat[x, y] * self.h
            xo += self.VM[dim].shape[0]

        # # S-term Matrices
        xo = 0
        yo = self.PM.shape[0]
        for dim in range(self.ndim):
            logger.info(f"\tAdding S-terms ({dim})", 2)
            s_mat = self.ST[dim]
            xi, yi = s_mat.nonzero()
            for x, y in zip(xi, yi):
                self.MM[x + xo, y + yo] = -s_mat[x, y] / self.h
            yo += self.VM[dim].shape[0]

        logger.debug("\tAssembling RHS")
        self.MM_rhs = zeros(self.P_dof_num)
        self.MM_rhs[0:len(self.P_COR)] = self.P_COR

        for dim in range(self.ndim):
            self.MM_rhs = concatenate( (self.MM_rhs, self.V_COR[dim] * self.h) )

        # logger.info("DEBUG:TODO)
        # from sparse_to_h5 import storeSparseProblem
        # storeSparseProblem(self.MM, self.MM_rhs, "BigMatrixStorage.h5")

        logger.debug("Converting to CSR")
        self.MM = self.MM.tocsr()
        logger.info("Solving . . .")

        self.solve_start = time.time()
        if self.method == "spsolve" or self.method == "nobi":
            from scipy.sparse.linalg import spsolve
            ans = spsolve(self.MM, self.MM_rhs)
        elif self.method == "bicgstab":
            ans = self.bicgstab(self.MM, self.MM_rhs)
        else:
            logger.warning("Solver method not supported!")
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
        logger.info("Starting Bi Optimization")
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
            logger.debug(f"Bi DOF Count Increased {inc_count}")
    
            # Some Bi numbers go up
            dm_copy[lower] *= up_mult
            dm_copy[logical_not(lower)] *= dn_mult

            # Bi Capped at some value
            dm_copy[dm_copy > mx] = mx

            self.DIV_MULT[:] = dm_copy


        me = mean(self.DIV_MULT)
        mi = min(self.DIV_MULT)
        mx = max(self.DIV_MULT)
        
        dbinfo = f"Bi Optimization Finished - Local Info Mean:{me:e} Min:{mi:e} Max:{mx:e}"
        logger.debug( dbinfo)

    def sync(self, place=""):
        # No-op for single-threaded scipy solver
        pass
        
    def iterate(self):
        '''This is the main iteration loop that converges the system.'''
        # Make sure we are setup properly
        self.force_la_setup()
        self.force_bc_setup()
        
        self.sync("Beginning of Iteration")
        ###############################
        # Solve the Pressure Equation #
        ###############################
        logger.debug("\tCalculating P RHS")
        # This is the Bi contribution
        self.P_RHS[:] = self.DIV_MULT * self.D_LIN
        for d in range(self.ndim):
            self.mat_mult(self.ST[d], self.V_LHS[d], self.PTEMP)
            self.P_RHS[:] = self.P_RHS[:] + self.PTEMP

        # Divide by h and add PBC pressure corrections
        self.P_RHS[:] = (self.P_RHS / self.h) + self.P_COR

        # Solve the pressure Eq.
        logger.debug("\tSolving P")
        self.SOLVE_P()

        ################################
        # Solve the Velocity Equations #
        ################################
        for dim in range(self.ndim):
            # Setup the RHS to the Velocity equation
            logger.debug(f"\tCalculating Grad P (V RHS - {dim})")
            self.mat_mult(self.GM[dim], self.P_LHS, self.VTEMP[dim])
            self.V_RHS[dim][:] = (self.VTEMP[dim] + self.V_COR[dim]) * self.h

            # Solve the velocity Equation in the n'th dimension
            logger.debug(f"\tSolving V - {dim}")
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

        # There is a classier way to do this with "take" from numpy
        for point in self.P_points_list:
            pdof = self.pdp(point)
            self.P_LHS[pdof] = self.P[point]

        for axis in range(self.ndim):
            for point in self.V_points_list[axis]:
                dof = self.nvd( axis, point )
                self.V_LHS[axis][dof] = self.V_GRIDS[axis][point]

    def assign_P_to_obj(self, obj):
        '''Used for saving pressure results to HDF5.'''
        for point in self.Get_P_Iterator():
            dof = self.pdp(point)
            obj[point] = self.P_LHS[dof]

    def assign_V_to_obj(self, axis, obj):
        '''Used for saving velocity results to HDF5.'''
        for point in self.Get_V_Iterator(axis):
            dof = self.nvd(axis, point)
            obj[point] = self.V_LHS[axis][dof]

    def converge(self, stopping_div = 1.0e-8, max_iter = inf, use_biot = True):
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

            logger.debug("\tUpdating Divergence" )
            # Update the divergence and max divergence
            self.update_D()

            # Update Bi
            if use_biot: self.update_Bi()

            # Print some status Crap
            logger.info(f"Iteration:{self.I}, Max Divergence:{self.max_D:e}")

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
