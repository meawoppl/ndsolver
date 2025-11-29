"""Unit tests for ndsolver core functionality."""

import numpy as np
import pytest


class TestImports:
    """Test that the package imports correctly."""

    def test_import_package(self):
        import ndsolver
        assert hasattr(ndsolver, '__version__')

    def test_import_solver(self):
        from ndsolver import Solver
        assert Solver is not None

    def test_import_core_module(self):
        from ndsolver import core
        assert hasattr(core, 'Solver')

    def test_import_symbolic(self):
        from ndsolver.symbolic import ndim_eq, ndimed
        assert ndim_eq is not None
        assert ndimed is not None


class TestSolverInstantiation:
    """Test Solver class instantiation."""

    @pytest.fixture
    def simple_2d_domain(self):
        """Create a simple 8x8 2D domain with a central obstacle."""
        domain = np.zeros((8, 8), dtype=np.int8)
        domain[3:5, 3:5] = 1  # solid block
        return domain

    @pytest.fixture
    def empty_2d_domain(self):
        """Create an empty 8x8 2D domain (no obstacles)."""
        return np.zeros((8, 8), dtype=np.int8)

    @pytest.fixture
    def simple_3d_domain(self):
        """Create a simple 8x8x8 3D domain with a central obstacle."""
        domain = np.zeros((8, 8, 8), dtype=np.int8)
        domain[3:5, 3:5, 3:5] = 1  # solid cube
        return domain

    def test_solver_creation_2d(self, simple_2d_domain):
        """Test that solver can be instantiated with 2D domain."""
        from ndsolver import Solver
        s = Solver(simple_2d_domain, (1.0, 0.0), sol_method='spsolve')
        assert s is not None

    def test_solver_creation_empty_domain(self, empty_2d_domain):
        """Test solver with empty domain (no obstacles)."""
        from ndsolver import Solver
        s = Solver(empty_2d_domain, (1.0, 0.0), sol_method='spsolve')
        assert s is not None

    def test_solver_creation_3d(self, simple_3d_domain):
        """Test that solver can be instantiated with 3D domain."""
        from ndsolver import Solver
        s = Solver(simple_3d_domain, (1.0, 0.0, 0.0), sol_method='spsolve')
        assert s is not None

    def test_solver_methods(self, simple_2d_domain):
        """Test that different solver methods can be selected."""
        from ndsolver import Solver

        for method in ['spsolve', 'splu', 'bicgstab']:
            s = Solver(simple_2d_domain, (1.0, 0.0), sol_method=method)
            assert s is not None

    def test_jax_solver_methods(self, simple_2d_domain):
        """Test that JAX solver methods can be selected (if JAX installed)."""
        from ndsolver import Solver
        pytest.importorskip("jax")

        for method in ['jax_bicgstab', 'jax_gmres']:
            s = Solver(simple_2d_domain, (1.0, 0.0), sol_method=method)
            assert s is not None


class TestSolverConvergence:
    """Test solver convergence on simple problems."""

    @pytest.fixture
    def small_2d_domain(self):
        """Create a small 5x5 2D domain for fast testing."""
        domain = np.zeros((5, 5), dtype=np.int8)
        domain[2, 2] = 1  # single solid cell
        return domain

    def test_converge_2d_spsolve(self, small_2d_domain):
        """Test convergence with spsolve method."""
        from ndsolver import Solver
        s = Solver(small_2d_domain, (1.0, 0.0), sol_method='spsolve')
        s.converge()
        assert s.max_D < 1e-6  # should converge

    def test_converge_2d_splu(self, small_2d_domain):
        """Test convergence with splu method."""
        from ndsolver import Solver
        s = Solver(small_2d_domain, (1.0, 0.0), sol_method='splu')
        s.converge()
        assert s.max_D < 1e-6  # should converge

    @pytest.mark.slow
    def test_converge_2d_bicgstab(self, small_2d_domain):
        """Test convergence with bicgstab iterative method.

        Note: bicgstab is significantly slower than direct solvers for small problems.
        """
        from ndsolver import Solver
        s = Solver(small_2d_domain, (1.0, 0.0), sol_method='bicgstab')
        s.converge()
        assert s.max_D < 1e-6  # should converge

    def test_converge_different_pressure_directions(self, small_2d_domain):
        """Test convergence with pressure drop in different directions."""
        from ndsolver import Solver

        # Pressure drop in x direction
        s1 = Solver(small_2d_domain, (1.0, 0.0), sol_method='spsolve')
        s1.converge()
        assert s1.max_D < 1e-6

        # Pressure drop in y direction
        s2 = Solver(small_2d_domain, (0.0, 1.0), sol_method='spsolve')
        s2.converge()
        assert s2.max_D < 1e-6


class TestSymbolicModule:
    """Test the symbolic equation generation module."""

    def test_equation_class(self):
        """Test the Equation dictionary class."""
        from ndsolver.symbolic.equation import Equation

        eq1 = Equation()
        eq1[1] = 2.0
        eq1[2] = 3.0

        eq2 = Equation()
        eq2[1] = 1.0
        eq2[3] = 4.0

        # Test addition
        result = eq1 + eq2
        assert result[1] == 3.0
        assert result[2] == 3.0
        assert result[3] == 4.0

        # Test subtraction
        result = eq1 - eq2
        assert result[1] == 1.0
        assert result[2] == 3.0
        assert result[3] == -4.0

    def test_ndimed_perturb(self):
        """Test the perturb function for neighbor generation."""
        from ndsolver.symbolic import ndimed

        point = (1, 1)
        neighbors = list(ndimed.perturb(point))
        assert len(neighbors) == 4  # 2D has 4 neighbors
        assert (0, 1) in neighbors
        assert (2, 1) in neighbors
        assert (1, 0) in neighbors
        assert (1, 2) in neighbors

    def test_ndimed_roller(self):
        """Test periodic boundary wrapping."""
        from ndsolver.symbolic import ndimed

        shape = (5, 5)
        # Point within bounds
        assert ndimed.roller((2, 3), shape) == (2, 3)
        # Point outside bounds (should wrap)
        assert ndimed.roller((6, 7), shape) == (1, 2)
        assert ndimed.roller((-1, -1), shape) == (4, 4)

    def test_p_dof_generation(self):
        """Test pressure degree of freedom grid generation."""
        from ndsolver.symbolic import ndim_eq

        domain = np.zeros((5, 5), dtype=np.int8)
        domain[2, 2] = 1  # solid cell

        p_dof = ndim_eq.p_dof(domain)

        # Solid cells should have -1
        assert p_dof[2, 2] == -1
        # Fluid cells should have non-negative DOF numbers
        assert p_dof[0, 0] >= 0
        assert p_dof[4, 4] >= 0
