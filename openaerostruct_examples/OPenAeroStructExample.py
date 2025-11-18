import numpy as np

import openmdao.api as om
import warnings

from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint


class ARControlComp(om.ExplicitComponent):
    """Scale mesh spanwise so the resulting AR equals the provided AR_target.

    This component computes the current AR from the incoming mesh (assumed
    shape (nx, ny, 3) where nx is chordwise and ny is spanwise for rect wings),
    computes a scale factor k = sqrt(AR_target / AR_current),
    and scales the y-coordinates by k.
    """
    def __init__(self, mesh_shape, symmetry=True, **kwargs):
        super().__init__(**kwargs)
        self.mesh_shape = mesh_shape
        self.symmetry = symmetry

    def setup(self):
        # mesh has physical units of meters
        self.add_input('mesh_in', shape=self.mesh_shape, units='m')
        self.add_input('AR_target', val=1.0)
        # output mesh also in meters
        self.add_output('mesh_out', shape=self.mesh_shape, units='m')
        # expose geometric metrics so we can constrain feasibility
        self.add_output('area_full', val=0.0, units='m**2')
        self.add_output('min_chord', val=0.0, units='m')
        # we'll use finite-diff for simplicity
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        mesh = np.array(inputs['mesh_in'], copy=True)
        AR_target = float(inputs['AR_target'])

        # Validate input mesh
        if not np.isfinite(mesh).all():
            warnings.warn("Input mesh contains NaN or Inf values; using unscaled mesh")
            outputs['mesh_out'] = mesh
            outputs['area_full'] = 1.0
            outputs['min_chord'] = 1.0
            return

        # compute current AR
        # For rectangular wing mesh: shape is (nx, ny, 3) where axis 1 is spanwise
        y_coords = mesh[..., 1]
        span_half = y_coords.max() - y_coords.min()
        
        # Chords are computed along axis 0 (chordwise direction)
        nx = mesh.shape[0]
        ny = mesh.shape[1]
        chord_vals = np.zeros(ny)
        for j in range(ny):  # iterate over spanwise stations
            x_coords = mesh[:, j, 0]  # all chordwise points at this span station
            chord_vals[j] = x_coords.max() - x_coords.min()
        
        # Check for degenerate chords
        if np.any(chord_vals <= 1e-9):
            warnings.warn(f"Degenerate chord detected (min={chord_vals.min()}); using unscaled mesh")
            outputs['mesh_out'] = mesh
            outputs['area_full'] = 1.0
            outputs['min_chord'] = chord_vals.min() if chord_vals.size > 0 else 0.0
            return
        
        # Spanwise station y locations (mean across chordwise direction)
        y_mean = y_coords.mean(axis=0)  # average over chord (axis 0)
        idx = np.argsort(y_mean)
        y_sorted = y_mean[idx]
        chord_sorted = chord_vals[idx]
        area_half = 0.0
        for k in range(len(y_sorted) - 1):
            dy = abs(y_sorted[k+1] - y_sorted[k])
            area_half += 0.5 * (chord_sorted[k] + chord_sorted[k+1]) * dy
        
        if self.symmetry:
            span_full = 2.0 * span_half
            area_full = 2.0 * area_half
        else:
            span_full = span_half
            area_full = area_half
        AR_current = (span_full ** 2) / area_full if area_full > 0 else 1.0

        # DEBUG: Print current values
        print(f"  ARControlComp: AR_current={AR_current:.4f}, AR_target={AR_target:.4f}, "
              f"span={span_full:.4f}, area={area_full:.4f}")

        # compute scale factor: AR = span^2 / area, so if we scale span by k,
        # AR_new = k^2 * AR_current, thus k = sqrt(AR_target / AR_current)
        k = np.sqrt(AR_target / AR_current) if AR_current > 0 else 1.0

        # Safety: avoid extreme scaling
        min_k = 0.8
        max_k = 1.2
        if not np.isfinite(k):
            warnings.warn(f"Non-finite scale factor k={k}; setting k=1.0")
            k = 1.0
        k_clipped = float(np.clip(k, min_k, max_k))
        if k_clipped != k:
            warnings.warn(f"Clipping AR scale factor from {k:.6f} to {k_clipped:.6f} to avoid extreme geometry")
        k = k_clipped

        # scale y-coordinates about the mesh mean to avoid translating the wing
        y_center = mesh[..., 1].mean()
        mesh_scaled = np.array(mesh, copy=True)
        mesh_scaled[..., 1] = y_center + (mesh[..., 1] - y_center) * k

        # Validation checks
        if not np.isfinite(mesh_scaled).all():
            warnings.warn("Scaled mesh contains NaN or Inf values; reverting to input mesh")
            outputs['mesh_out'] = inputs['mesh_in']
            outputs['area_full'] = float(area_full) if np.isfinite(area_full) else 1.0
            outputs['min_chord'] = chord_vals.min() if chord_vals.size > 0 else 0.1
            return

        # recompute chords and area on scaled mesh
        y_coords_s = mesh_scaled[..., 1]
        span_half_s = y_coords_s.max() - y_coords_s.min()
        ny_s = mesh_scaled.shape[1]
        chord_vals_s = np.zeros(ny_s)
        for j in range(ny_s):
            x_coords = mesh_scaled[:, j, 0]
            chord_vals_s[j] = x_coords.max() - x_coords.min()

        y_mean_s = y_coords_s.mean(axis=0)
        idx_s = np.argsort(y_mean_s)
        y_sorted_s = y_mean_s[idx_s]
        chord_sorted_s = chord_vals_s[idx_s]
        area_half_s = 0.0
        for kk in range(len(y_sorted_s) - 1):
            dy = abs(y_sorted_s[kk+1] - y_sorted_s[kk])
            area_half_s += 0.5 * (chord_sorted_s[kk] + chord_sorted_s[kk+1]) * dy
        
        if self.symmetry:
            span_full_s = 2.0 * span_half_s
            area_full_s = 2.0 * area_half_s
        else:
            span_full_s = span_half_s
            area_full_s = area_half_s

        min_chord = chord_vals_s.min() if chord_vals_s.size > 0 else 0.0
        
        # More lenient validation
        if (not np.isfinite(area_full_s)) or area_full_s <= 1e-12:
            warnings.warn(f"Scaled mesh invalid (area={area_full_s}); reverting to input mesh")
            outputs['mesh_out'] = inputs['mesh_in']
            outputs['area_full'] = float(area_full) if np.isfinite(area_full) else 1.0
            outputs['min_chord'] = chord_vals.min() if chord_vals.size > 0 else 0.1
            return
        
        if min_chord <= 1e-9:
            warnings.warn(f"Scaled mesh invalid (min_chord={min_chord}); reverting to input mesh")
            outputs['mesh_out'] = inputs['mesh_in']
            outputs['area_full'] = float(area_full) if np.isfinite(area_full) else 1.0
            outputs['min_chord'] = chord_vals.min() if chord_vals.size > 0 else 0.1
            return

        # pass scaled mesh and metrics
        outputs['mesh_out'] = mesh_scaled
        outputs['area_full'] = float(area_full_s)
        outputs['min_chord'] = float(min_chord)
        
        # DEBUG: Print output values
        print(f"  ARControlComp output: area={area_full_s:.4f}, min_chord={min_chord:.6f}")


# Create a dictionary to store options about the mesh
# Using CRM (Common Research Model) wing which has realistic geometry
# Increase mesh resolution to avoid numerical issues
mesh_dict = {
    "num_y": 7,      # number of spanwise points
    "num_x": 3,      # number of chordwise points (increased from 2)
    "wing_type": "CRM",  # Use CRM wing
    "symmetry": True,
    "num_twist_cp": 5
}

# Generate the aerodynamic mesh based on the previous dictionary
mesh_result = generate_mesh(mesh_dict)
# Handle both single mesh return and (mesh, twist_cp) tuple return
if isinstance(mesh_result, tuple):
    mesh, twist_cp = mesh_result
else:
    mesh = mesh_result
    twist_cp = np.zeros(mesh_dict["num_twist_cp"])

# Validate initial mesh
print("Initial mesh shape:", mesh.shape)
print("Initial mesh finite:", np.isfinite(mesh).all())
print("Initial mesh y-range:", mesh[..., 1].min(), "to", mesh[..., 1].max())
print("Initial mesh x-range:", mesh[..., 0].min(), "to", mesh[..., 0].max())
print("Initial mesh z-range:", mesh[..., 2].min(), "to", mesh[..., 2].max())

# Create a dictionary with info and options about the aerodynamic
# lifting surface
surface = {
    # Wing definition
    "name": "wing",  # name of the surface
    "symmetry": True,  # if true, model one half of wing
    # reflected across the plane y = 0
    "S_ref_type": "wetted",  # how we compute the wing area,
    # can be 'wetted' or 'projected'
    "fem_model_type": "tube",
    "twist_cp": twist_cp,
    "mesh": mesh,
    # Aerodynamic performance of the lifting surface at
    # an angle of attack of 0 (alpha=0).
    # These CL0 and CD0 values are added to the CL and CD
    # obtained from aerodynamic analysis of the surface to get
    # the total CL and CD.
    # These CL0 and CD0 values do not vary wrt alpha.
    "CL0": 0.0,  # CL of the surface at alpha=0
    "CD0": 0.015,  # CD of the surface at alpha=0
    # Airfoil properties for viscous drag calculation
    "k_lam": 0.05,  # percentage of chord with laminar
    # flow, used for viscous drag
    "t_over_c_cp": np.array([0.15]),  # thickness over chord ratio (NACA0015)
    "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
    # thickness
    "with_viscous": True,  # if true, compute viscous drag
    "with_wave": False,  # if true, compute wave drag
}

# Create the OpenMDAO problem
prob = om.Problem()

# Create an independent variable component that will supply the flow
# conditions to the problem.
indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=248.136, units="m/s")
indep_var_comp.add_output("alpha", val=5.0, units="deg")
indep_var_comp.add_output("Mach_number", val=0.84)
indep_var_comp.add_output("re", val=1.0e6, units="1/m")
indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")
indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")

# Compute initial aspect ratio (AR) from the generated mesh so we can
# use it as a sensible default for the AR design variable.
def _compute_mesh_AR(mesh, symmetry=True):
    """Compute aspect ratio from mesh.
    
    For rectangular wing meshes from generate_mesh:
    mesh shape is (nx, ny, 3) where axis 0 is chordwise, axis 1 is spanwise
    """
    y_coords = mesh[..., 1]
    # span per half (if symmetry True) or full wing depending on mesh
    span_half = y_coords.max() - y_coords.min()
    # estimate chord at each spanwise station (axis 1)
    ny = mesh.shape[1]  # spanwise dimension
    chord_vals = np.zeros(ny)
    for j in range(ny):
        x_coords = mesh[:, j, 0]  # chordwise points at this span station
        chord_vals[j] = x_coords.max() - x_coords.min()
    
    print(f"  AR calc: ny={ny}, span_half={span_half:.4f}, chords={chord_vals}")
    
    # spanwise station y locations (mean of each column across chordwise)
    y_mean = y_coords.mean(axis=0)  # average over chordwise direction (axis 0)
    idx = np.argsort(y_mean)
    y_sorted = y_mean[idx]
    chord_sorted = chord_vals[idx]
    area_half = 0.0
    for k in range(len(y_sorted) - 1):
        dy = abs(y_sorted[k+1] - y_sorted[k])
        area_half += 0.5 * (chord_sorted[k] + chord_sorted[k+1]) * dy
        print(f"    k={k}: dy={dy:.4f}, chord_avg={0.5*(chord_sorted[k]+chord_sorted[k+1]):.4f}, area_contrib={0.5*(chord_sorted[k]+chord_sorted[k+1])*dy:.4f}")
    
    if symmetry:
        span_full = 2.0 * span_half
        area_full = 2.0 * area_half
    else:
        span_full = span_half
        area_full = area_half
    
    print(f"  AR calc result: span_full={span_full:.4f}, area_full={area_full:.4f}")
    
    AR = (span_full ** 2) / area_full if area_full > 0 else 0.0
    return AR

# sensible initial AR from the generated mesh
initial_AR = _compute_mesh_AR(mesh, symmetry=surface.get('symmetry', True))
indep_var_comp.add_output('AR_target', val=float(initial_AR))

# Add this IndepVarComp to the problem model
prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

# Create and add a group that handles the geometry for the
# aerodynamic lifting surface
geom_group = Geometry(surface=surface)
prob.model.add_subsystem(surface["name"], geom_group)

# Create the aero point group, which contains the actual aerodynamic
# analyses
aero_group = AeroPoint(surfaces=[surface])
point_name = "aero_point_0"
prob.model.add_subsystem(
    point_name, aero_group, promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"]
)

name = surface["name"]

# TEMPORARILY DISABLE AR control component to test basic setup
USE_AR_CONTROL = False

if USE_AR_CONTROL:
    # Insert AR control component that scales the mesh so AR == AR_target
    ar_ctrl = ARControlComp(mesh.shape, symmetry=surface.get('symmetry', True))
    prob.model.add_subsystem('ar_ctrl', ar_ctrl)

    # connect geometry mesh -> AR controller -> aero point
    prob.model.connect(name + ".mesh", 'ar_ctrl.mesh_in')
    prob.model.connect('AR_target', 'ar_ctrl.AR_target')
    prob.model.connect('ar_ctrl.mesh_out', point_name + "." + name + ".def_mesh")

    # Perform the connections with the modified names within the
    # 'aero_states' group (use scaled mesh)
    prob.model.connect('ar_ctrl.mesh_out', point_name + ".aero_states." + name + "_def_mesh")
else:
    # Direct connection without AR control
    prob.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")
    prob.model.connect(name + ".mesh", point_name + ".aero_states." + name + "_def_mesh")

prob.model.connect(name + ".t_over_c", point_name + "." + name + "_perf." + "t_over_c")

# Import the Scipy Optimizer and set the driver of the problem to use
# it, which defaults to an SLSQP optimization method
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["tol"] = 1e-9

# Setup problem and add design variables, constraint, and objective
prob.model.add_design_var("wing.twist_cp", lower=-10.0, upper=15.0)
# Make AR_target a direct design variable (controls mesh scaling inside ar_ctrl)
# Start with no AR optimization - just use initial value
# This avoids the scaling issues initially
print(f"\nInitial AR from mesh: {initial_AR:.6f}")
# Don't make AR a design variable for now - just use fixed initial AR
# prob.model.add_design_var('AR_target', lower=0.95 * float(initial_AR), upper=1.05 * float(initial_AR))

# Feasibility constraints: require positive area and a minimum chord to avoid
# degenerate geometry that causes NaNs in the aerodynamic solve.
# Use much more lenient values based on actual mesh
if USE_AR_CONTROL:
    area_tol = 1e-6  # m^2, very small tolerance
    min_chord_tol = 1e-6  # meters; very small tolerance
    prob.model.add_constraint('ar_ctrl.area_full', lower=area_tol)
    prob.model.add_constraint('ar_ctrl.min_chord', lower=min_chord_tol)

prob.model.add_constraint(point_name + ".wing_perf.CL", equals=0.5)
prob.model.add_objective(point_name + ".wing_perf.CD", scaler=1e4)

# Set up the problem
prob.setup()

# DEBUG RUN: run a single model evaluation (no optimization) to inspect mesh
print('\n--- DEBUG: running single model evaluation (no driver) ---')
try:
    prob.run_model()
    print("✓ Model ran successfully!")
    
    # Print results
    CL = prob.get_val(point_name + '.wing_perf.CL')
    CD = prob.get_val(point_name + '.wing_perf.CD')
    print(f'\nAerodynamic Results:')
    print(f'  CL = {float(CL):.6f}')
    print(f'  CD = {float(CD):.6f}')
    print(f'  L/D = {float(CL/CD):.6f}')
    
except Exception as e:
    print('Error during prob.run_model():', repr(e))
    raise

if USE_AR_CONTROL:
    try:
        ar_target_val = prob.get_val('AR_target')
        area_val = prob.get_val('ar_ctrl.area_full')
        min_chord_val = prob.get_val('ar_ctrl.min_chord')
        mesh_out = prob.get_val('ar_ctrl.mesh_out')
        print(f'\nAR Control Outputs:')
        print(f'  AR_target = {float(ar_target_val):.6g}')
        print(f'  ar_ctrl.area_full = {float(area_val):.6g} m^2')
        print(f'  ar_ctrl.min_chord = {float(min_chord_val):.6g} m')
        print(f'  mesh_out finite: {bool(np.isfinite(mesh_out).all())}')
    except Exception as e:
        print('Error reading AR/mesh outputs:', repr(e))

print('\n--- end debug run ---')
import sys
sys.exit(0)