
from CSADtesting.allocation.allocation import CSADThrusterAllocator
from MCSimPython.utils import six2threeDOF
from MCSimPython.utils import Rz, pipi

def feedback_linearizing_pd_controller(vessel, eta, nu, eta_d, nu_d, ddr, kp=10.0, kd=0.1):
    # Compute the desired acceleration
    z1 = Rz(eta[-1]).T@(eta-eta_d)
    z1[-1] = pipi(eta[-1]-eta_d[-1])
    z2 = nu - nu_d

    ddr = Rz(eta_d[-1]).T@ddr

    dv = ddr - kp * z1 - kd * z2
    #print(dv)
    # # Compute the desired forces and moments
    M, D= six2threeDOF(vessel._M), six2threeDOF(vessel._D)  

    tau = M @ dv + D @ nu #assuming C = 0
    u = CSADThrusterAllocator().allocate(tau[0], tau[1], tau[2])

    return tau, u
