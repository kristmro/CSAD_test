# allocation.py
import numpy as np

import skadipy
from skadipy.actuator import Azimuth
from skadipy.allocator._base import ForceTorqueComponent as FTC
from skadipy.allocator.reference_filters import MinimumMagnitudeAndAzimuth
from skadipy.safety import ControlBarrierFunctionType
from skadipy.toolbox import Point

class CSADThrusterAllocator:
    """
    A class that sets up 6 azimuth thrusters with outward angles
    (front thrusters near 0° ± 45°, rear thrusters near 180° ± 45°),
    and uses skadipy's MinimumMagnitudeAndAzimuth allocator to
    allocate [Fx, Fy, Mz] in the body frame.

    Example usage:
        allocator = CSADThrusterAllocator(time_step=0.05)
        tau_alloc = allocator.allocate(fx=2.0, fy=1.0, mz=0.5)
        # tau_alloc is shape (3,) => [Fx, Fy, Mz] actually allocated
        # The per-thruster forces are in self.actuators[i].force
    """

    def __init__(self, time_step=0.01):
        # Define thrusters => positions in [x,y], reference angles
        #   Forward thrusters => near 0° ± 45°
        #   Rear thrusters    => near 180° ± 45°
        #
        # Based on your lx, ly arrays:
        #   front_center => (x=+1.0678, y=0)
        #   front_left   => (x=+0.9344, y=+0.11)
        #   front_right  => (x=+0.9344, y=-0.11)
        #   rear_center  => (x=-1.1644, y=0)
        #   rear_left    => (x=-0.9911, y=+0.1644)
        #   rear_right   => (x=-0.9911, y=-0.1644)

        def azimuth_thruster(x, y, ref_angle, name):
            return Azimuth(
                position=Point([x, y, 0.0]),
                extra_attributes={
                    "rate_limit": 1.0,        # example
                    "saturation_limit": 1.0,  # example
                    "reference_angle": ref_angle,
                    "name": name
                }
            )

        # ±45° in radians
        ang_45 = np.deg2rad(45.0)

        # Build the 6 thrusters
        self.thrusters = [
            # forward center => near 0°
            azimuth_thruster(1.0678, 0.0, 0.0,                 "fwd_center"),
            # forward left => +45°
            azimuth_thruster(0.9344,  0.11,  +ang_45,          "fwd_left"),
            # forward right => -45°
            azimuth_thruster(0.9344, -0.11,  -ang_45,          "fwd_right"),

            # rear center => near 180°
            azimuth_thruster(-1.1644,  0.0,  np.pi,            "bwd_center"),
            # rear left => 180° + 45°
            azimuth_thruster(-0.9911,  0.1644, np.pi + ang_45, "bwd_left"),
            # rear right => 180° - 45°
            azimuth_thruster(-0.9911, -0.1644, np.pi - ang_45, "bwd_right"),
        ]

        # We want to allocate [X, Y, N] => surge, sway, yaw
        self.dofs = [FTC.X, FTC.Y, FTC.N]

        # Create the skadipy allocator
        self.allocator = MinimumMagnitudeAndAzimuth(
            actuators = self.thrusters,
            force_torque_components = self.dofs,
            gamma  = 0.1,   
            mu     = 0.1,
            rho    = 100,
            time_step = time_step,
            control_barrier_function = ControlBarrierFunctionType.ABSOLUTE
        )

        # Must compute configuration matrix once
        self.allocator.compute_configuration_matrix()

    def allocate(self, fx, fy, mz):
        """
        Allocate thruster forces for a desired 3-DOF command [Fx, Fy, Mz].
        Returns the allocated [Fx, Fy, Mz].
        Also updates self.thrusters[i].force for each thruster.
        """
        # skadipy expects a 6x1 vector [Fx, Fy, Fz, Mx, My, Mz].
        # We'll set Fz=Mx=My=0.0
        tau_cmd = np.array([
            [fx],
            [fy],
            [0.0],
            [0.0],
            [0.0],
            [mz]
        ], dtype=np.float32)

        # Allocate
        self.allocator.allocate(tau=tau_cmd)

        # The final (Fx,Fy) we can sum from each thruster; Mz from .allocated
        fx_sum, fy_sum = 0.0, 0.0
        for thr in self.thrusters:
            F = thr.force  # typically shape (2,1)
            if F.shape[0] == 1:
                # single dimension => purely X
                fx_sum += F[0,0]
            else:
                fx_sum += F[0,0]
                fy_sum += F[1,0]
        # Mz from the aggregator's "allocated" 6×1 vector
        tau_alloc = self.allocator.allocated
        mz_sum = tau_alloc[5, 0]

        return np.array([fx_sum, fy_sum, mz_sum], dtype=float)
