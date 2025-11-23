from ocelot import *
from ocelot.gui.accelerator import *
from ocelot.cpbd.orbit_correction import *

QF1 = Quadrupole(l=0.3, k1=1)
QF2 = Quadrupole(l=0.3, k1=1)
QF3 = Quadrupole(l=0.3, k1=1)
QF4 = Quadrupole(l=0.3, k1=1)
QF5 = Quadrupole(l=0.3, k1=1)
QF6 = Quadrupole(l=0.3, k1=1)

QD1 = Quadrupole(l=0.3, k1=-1)
QD2 = Quadrupole(l=0.3, k1=-1)
QD3 = Quadrupole(l=0.3, k1=-1)
QD4 = Quadrupole(l=0.3, k1=-1)
QD5 = Quadrupole(l=0.3, k1=-1)
QD6 = Quadrupole(l=0.3, k1=-1)

D1 = Drift(l=2)
D2 = Drift(l=0.06)

M0 = Monitor(l=0)
M1 = Monitor(l=0)
M2 = Monitor(l=0)
M3 = Monitor(l=0)
M4 = Monitor(l=0)
M5 = Monitor(l=0)
M6 = Monitor(l=0)
M7 = Monitor(l=0)
M8 = Monitor(l=0)
M9 = Monitor(l=0)
M10 = Monitor(l=0)
M11 = Monitor(l=0)
M12 = Monitor(l=0)

CH1 = Hcor(l=0, angle=0)
CH2 = Hcor(l=0, angle=0)
CH3 = Hcor(l=0, angle=0)
CH4 = Hcor(l=0, angle=0)
CH5 = Hcor(l=0, angle=0)
CH6 = Hcor(l=0, angle=0)
CH7 = Hcor(l=0, angle=0)
CH8 = Hcor(l=0, angle=0)
CH9 = Hcor(l=0, angle=0)
CH10 = Hcor(l=0, angle=0)
CH11 = Hcor(l=0, angle=0)
CH12 = Hcor(l=0, angle=0)

CV1 = Vcor(l=0, angle=0)
CV2 = Vcor(l=0, angle=0)
CV3 = Vcor(l=0, angle=0)
CV4 = Vcor(l=0, angle=0)
CV5 = Vcor(l=0, angle=0)
CV6 = Vcor(l=0, angle=0)
CV7 = Vcor(l=0, angle=0)
CV8 = Vcor(l=0, angle=0)
CV9 = Vcor(l=0, angle=0)
CV10 = Vcor(l=0, angle=0)
CV11 = Vcor(l=0, angle=0)
CV12 = Vcor(l=0, angle=0)

cell = (M0, D1,
        CH1, CV1, D2, D1, QF1, D2, M1,  
        CH2, CV2, D2, D1, QD1, D2, M2, 
        CH3, CV3, D2, D1, QF2, D2, M3,  
        CH4, CV4, D2, D1, QD2, D2, M4, 
        CH5, CV5, D2, D1, QF3, D2, M5,  
        CH6, CV6, D2, D1, QD3, D2, M6, 
        CH7, CV7, D2, D1, QF4, D2, M7,  
        CH8, CV8, D2, D1, QD4, D2, M8, 
        CH9, CV9, D2, D1, QF5, D2, M9,  
        CH10, CV10, D2, D1, QD5, D2, M10,
        CH11, CV11, D2, D1, QF6, D2, M11, 
        CH12, CV12, D2, D1, QD6, D2, M12)   

def get_k1_array(lattice):
    '''
    get k1 attribute values of all Quadrupole objects in lattice.sequence
    parameters:
        lattice: MagneticLattice object
    return:
        np.ndarray: k1 attribute values of all Quadrupole objects in lattice.sequence
    '''
    k1 = []
    for i in lattice.sequence:
        if i.__class__ == Quadrupole:
            k1.append(i.k1)
    return np.array(k1)

def get_hcors_angle(lattice):
    '''
    get angle attribute values of all Hcor objects in lattice.sequence
    parameters:
        lattice: MagneticLattice object
    return:
        np.ndarray: angle attribute values of all Hcor objects in lattice.sequence
    '''
    hcors = []
    for i in lattice.sequence:
        if i.__class__ == Hcor:
            hcors.append(i.angle)
    return np.array(hcors, dtype=np.float64)

def get_vcors_angle(lattice):
    '''
    get angle attribute values of all Vcor objects in lattice.sequence
    parameters:
        lattice: MagneticLattice object
    return:
        np.ndarray: angle attribute values of all Vcor objects in lattice.sequence
    '''
    vcors = []
    for i in lattice.sequence:
        if i.__class__ == Vcor:
            vcors.append(i.angle)
    return np.array(vcors, dtype=np.float64)

def get_RM(lattice, k1, hcors, vcors, tws, method):
    '''
    get response matrix in lattice.sequence
    parameters:
        lattice: MagneticLattice object
        k1: np.ndarray, k1 attribute values of all Quadrupole objects in lattice.sequence
        hcors: np.ndarray, angle attribute values of all Hcor objects in lattice.sequence
        vcors: np.ndarray, angle attribute values of all Vcor objects in lattice.sequence
        tws: Twiss object
        method: LinacOpticalRM, LinacSimRM, LinacRmatrixRM, LinacDisperseSimRM, LinacDisperseTmatrixRM (last two types can not use now)
    return:
        RM: np.ndarray, response matrix
    '''
    l = 0
    h = 0
    v = 0
    for i in lattice.sequence:
        if i.__class__ == Quadrupole:
            i.k1 = k1[l]
            l += 1
        elif i.__class__ == Hcor:
            i.angle = hcors[h]
            h += 1
        elif i.__class__ == Vcor:
            i.angle = vcors[v]
            v += 1
    orb = Orbit(lattice)
    method = method(orb.lat, orb.hcors, orb.vcors, orb.bpms)
    orb.response_matrix = method.calculate(tws)
    return orb.response_matrix

def get_ideal_orbit(lattice, k1, hcors, vcors, p_init):
    '''
    get ideal beam orbit in lattice.sequence
    parameters:
        lattice: MagneticLattice...
        p_init: Particle
    return:
        x, y: np.array, beam orbit in lattice.sequence
    '''
    l = 0
    h = 0
    v = 0
    for i in lattice.sequence:
        if i.__class__ == Quadrupole:
            i.k1 = k1[l]
            l += 1
        elif i.__class__ == Hcor:
            i.angle = hcors[h]
            h += 1
        elif i.__class__ == Vcor:
            i.angle = vcors[v]
            v += 1
    orb = Orbit(lattice)
    v_orb = MeasureResponseMatrix(orb.lat, orb.hcors, orb.vcors, orb.bpms)
    return v_orb.read_virtual_orbit(p_init)

def set_hcor_angle(lattice, hcor_angle):
    '''
    input angle attribute values of all Hcor objects in lattice.sequence
    parameters:
        lattice: MagneticLattice
        hcor_angle: list of float
    return: None
    '''
    h = 0
    for i in lattice.sequence:
        if i.__class__ == Hcor:
            i.angle = hcor_angle[h]
            h += 1
    # print('angle of hcors setted', end='\r')
    return lattice

def set_vcor_angle(lattice, vcor_angle):
    '''
    input angle attribute values of all Vcor objects in lattice.sequence
    parameters:
        lattice: MagneticLattice
        vcor_angle: list of float
    return: None
    '''
    v = 0
    for i in lattice.sequence:
        if i.__class__ == Vcor:
            i.angle = vcor_angle[v]
            v += 1
    # print('angle of hcors setted', end='\r')
    return lattice

def clip_cor_angles(cor_angles, max_angle=0.3):
    '''
    clip corrector angles to be within [-max_angle, max_angle]
    parameters:
        cor_angles: np.ndarray, corrector angles
        max_angle: float, maximum absolute angle value
    return:
        np.ndarray: clipped corrector angles
    '''
    return np.clip(cor_angles, -max_angle, max_angle).astype(np.float64)

def validate_action(action):
    '''
    validate action values to be within allowed range [0.001, 0.01] or [-0.01, -0.001]
    parameters:
        action: np.ndarray, action values
    return:
        np.ndarray: validated action values
    '''
    # Create a copy to avoid modifying the original
    validated_action = np.copy(action).astype(np.float64)
    
    # For positive values, clip to [0.001, 0.01]
    positive_mask = validated_action > 0
    validated_action[positive_mask] = np.clip(validated_action[positive_mask], 0.001, 0.01)
    
    # For negative values, clip to [-0.01, -0.001]
    negative_mask = validated_action < 0
    validated_action[negative_mask] = np.clip(validated_action[negative_mask], -0.01, -0.001)
    
    # For zero values, set to small positive value
    zero_mask = validated_action == 0
    validated_action[zero_mask] = 0.001
    
    return validated_action

def calculate_orbit_diff(current_orbit, previous_orbit, prev_prev_orbit):
    '''
    calculate orbit differences for state representation
    parameters:
        current_orbit: np.ndarray, current orbit readings
        previous_orbit: np.ndarray, previous orbit readings
        prev_prev_orbit: np.ndarray, previous to previous orbit readings
    return:
        tuple: (current_orbit, diff_to_prev, diff_to_prev_prev)
    '''
    # If previous orbits are None, set differences to zero
    if previous_orbit is None:
        diff_to_prev = np.zeros_like(current_orbit)
    else:
        diff_to_prev = current_orbit - previous_orbit
    
    if prev_prev_orbit is None:
        diff_to_prev_prev = np.zeros_like(current_orbit)
    else:
        diff_to_prev_prev = previous_orbit - prev_prev_orbit if previous_orbit is not None else np.zeros_like(current_orbit)
    
    return current_orbit, diff_to_prev, diff_to_prev_prev