#--------- READ ME----------------------------------------------------------------
# A collection of functions for the generation of a number of 
# "prepackaged" high lift arrays. This is also an example script on how to use
# the highliftdev module. These functions generate high lift devices with
# randomized parameters, putting great attention to the robustness and consistency
# of the results.
#---------------------------------------------------------------------------------

import numpy as np
import geometrics as gmt
import highliftdev as hld
import foilgeneral as flg
import functbook as fnb


# PRIVATE STUFF
def _nonerand(x, functstr, drange=[0,1]):
    """
    "Private" function. Return a random number or the number given.

    Args:
        x: if x != None then a random value is generated, else return x
        funcstr (str): string describing the function that will be used
            -random: default rng
            -betaab: beta distribution with a , b values
            -bool: bool rng 1 or 0
            -int: integer rng
        drange (array-like): the range of the randomgenerate valie
    
    Returns:
        randomly generated value

    """
    if x == None:
        if functstr == 'random':
            x = np.random.default_rng().random()
            x1 = drange[0]
            x2 = drange[1] - drange[0]
            x = x1 + x*x2
        elif functstr[0:4] == 'beta':
            a = float(functstr[4])
            b = float(functstr[5])
            x = np.random.default_rng().beta(a,b)
            x1 = drange[0]
            x2 = drange[1] - drange[0]
            x = x1 + x*x2
        elif functstr == 'bool':
            x = np.random.default_rng().choice([False, True])
        elif functstr == 'int':
            x = np.random.default_rng().integers(drange[0], high=drange[1], endpoint=True)
    elif (x != None) and (functstr == 'bool'):
        x = 1

    return x


def _fixed_fore_flap(afl: flg.Airfoil, divx, cs, gap):
    """
    Return a fixed leading edge fore flap device. Close to fixed slot but different.
    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '<')[:,1])
    cl = max(gmt.crv_cut(flg.camberline(afl), 50, '<')[:,1])
    r2 = _nonerand(None, 'beta33', [4, 5]) * t/10
    css1 = (cs**2 - (t/2 + cl)**2)**0.5
    csp1 = _nonerand(None, 'beta33', [3, 6]) * (t/10) + 1
    width = 0.7 * (cs - csp1)

    csp2 = csp1 + width
    css2 = css1 + gap*2

    cgenarg1 = 3*gap
    cgenarg2 = 'dakka'
    cgenfunc1, cgenfunc2 = 'arc_p', 'arc_tang'
    r1 = t/4
    
    arg1 = [css1, csp1, cgenfunc1, cgenarg1, r1]
    arg2 = [divx, css2, csp2, cgenfunc2, cgenarg2, r2]

    bood = flg.hld_gen(afl, hld.le_slot, arg2)
    sloot = flg.hld_gen(afl, hld.slat, arg1)

    return bood + sloot


def _act_fore_flap(afl: flg.Airfoil, divx, cs, dtheta, gap):
    """
    Return an actuating leading edge fore flap device. Close to act_slat but different.
    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '<')[:,1])
    cl = max(gmt.crv_cut(flg.camberline(afl), 50, '<')[:,1])

    crv_type = _nonerand(None, 'int', [1,3])
    cs = _nonerand(cs, 'beta33', [12*(t/10)**0.5, 16*(t/10)**0.5]) + (cl/t) * 10
    css = (cs**2 - (t/6 + cl)**2)**0.5
    csp = _nonerand(None, 'beta33', [4, 8]) * (t/10)**0.5 + 1
    r2 = _nonerand(None, 'beta33', [2, 2]) * (t/10)**0.5
    r1 = 7 / (t/10)**0.5

    if crv_type == 1:
        nf = _nonerand(None, 'beta33', [0.6, 1.1]) * (t/10)**0.25
        cgenarg = fnb.gen_pchip([0.52*nf, 0.54*nf, 0.56*nf])
        cgenfunc = 'median'
    elif crv_type == 2:
        nf = _nonerand(None, 'int', [7, 11])
        cgenarg = [nf, int(csp * nf/12)]
        cgenfunc = 'bezier'
    elif crv_type == 3:
        cgenarg = _nonerand(None, 'beta33', [1, 5])
        cgenfunc = 'arc_p'

    arg1 = [css, csp, cgenfunc, cgenarg, r1]
    arg2 = [divx, css, csp, cgenfunc, cgenarg, r2]
    bood = flg.hld_gen(afl, hld.le_slot, arg2)
    sloot = flg.hld_gen(afl, hld.slat, arg1)

    slootafl = flg.gs2afl(flg.crv2gs(sloot)[0])
    stheta = slootafl.default_state(transform=False)[0]/3
    gf = 1/afl.default_state(transform=False)[1]
    actheta = (-dtheta + stheta)/(0.5 + 40 * cl/t)
    tempslt = flg.hld_gen(afl, hld.slat, [css, csp, cgenfunc, cgenarg, 0])
    sp0 = tempslt[0][1][0]
    bp0 = gmt.crcl_fit(tempslt[0][1])[0]

    gap = gap * (1 + 2 * (cl/t)) + gf * 2.5 * (cs * np.sin(-stheta) * ((t/12.5)**0.5 + 0.2 * cl/t))
    tv = [-gap, 0]

    sloot = sloot[0][0]
    sloot = gmt.rotate(sloot, sp0, stheta)
    sloot = gmt.translate(sloot, tv)
    sloot = gmt.rotate(sloot, bp0, actheta)
    sloot = [[sloot]]

    return bood + sloot


def _slot_after_flap(afl: flg.Airfoil, divx, cf, dtheta, gap):
    """
    Return an actuating trailing edge after flap device. Close to flap_1slot but different.
    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '>')[:,1])
    cl = max(gmt.crv_cut(flg.camberline(afl), 50, '>')[:,1])
    crv_type = _nonerand(None, 'int', [1,3])

    if crv_type == 1:
        cgenfunc = 'marriage'
        cgenarg = [0.2, [4, 4, 12, 4]]
    elif crv_type == 2:
        cgenfunc = 'arc_tang'
        cgenarg = 'dakka'
    elif crv_type == 3:
        cgenfunc = 'arc_p'
        cgenarg = 5

    cfp = cf
    cfdif = _nonerand(None, 'beta33', [0.1*cfp, 0.2*cfp]) * (t/10)**0.5
    cfs = cfp - cfdif
    rf = t/6 * cfs/50

    flap_args = [cfs, cfp, cgenfunc, cgenarg, rf]
    body_args = [divx, cfs, cfp, cgenfunc, cgenarg, 0]

    bood = flg.hld_gen(afl, hld.te_slot, body_args)
    floop = flg.hld_gen(afl, hld.flap, flap_args)
    
    slot_curve = flg.hld_gen(afl, hld.te_slot, [divx, cfs, cfp, cgenfunc, cgenarg, 0])[0][1]
    p0 = gmt.crcl_fit(slot_curve)[0]
    tv = [gap, 0]

    floop = floop[0][0]
    floop = gmt.rotate(floop, p0, dtheta)
    floop = gmt.translate(floop, tv)
    floop = [[floop]]

    return bood + floop


__leflapdict = {1: hld.le_flap1, 2: hld.le_flap2, 3: hld.le_flap3}


# LEADING EDGE ARRAYS
def bare_le(afl: flg.Airfoil, divx = [60, 45]):
    """
    Returns the bare leading edge, just like in hld. Nothing more, just repeated here for the sake of consistency.
    """
    return flg.hld_gen(afl, hld.bare_le, [divx])


def le_flap(afl: flg.Airfoil, divx = [60, 45], flap_type = None, cf = None, dtheta = None):
    """
    Return a randomly generated leading edge flap.

    Args:
        afl (Airfoil): airfoil
        flap_type (int): 1 - 3. bot hinge, top hinge, varcamber
        cf (float): the chord of the flap
        dtheta (float): angle of rotation

    """
    flap_type = _nonerand(flap_type, 'int', [1,3])
    css = _nonerand(cf, 'beta33', [12, 18]) * (max(flg.thickness(afl)[:,1])/10)**0.25 
    csp = _nonerand(None, 'beta33', [css, css + 2])
    dtheta = np.radians(_nonerand(dtheta, 'beta33', [15, 30]))
    arg = [divx, css, csp, dtheta]
    return flg.hld_gen(afl, __leflapdict[flap_type], arg)


def fixed_slot(afl: flg.Airfoil, divx = [60, 45], cs = None, crv_type = None, gap = None, width = None, r = None):
    """
    Return a randomly generated leading edge slot.

    Args:
        afl (Airfoil): airfoil
        cs (float): the chord of the slat
        crv_type (int): 1 - 3 median, bezier, arc
        gap (float): the suction side gap of the slot
        width (float): the pressure side width of the slot
        r (float): the radius of the curvature of the slat 

    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '<')[:,1])
    cl = max(gmt.crv_cut(flg.camberline(afl), 50, '<')[:,1])

    width = _nonerand(width, 'beta33', [2, 6])
    gap = _nonerand(gap, 'beta33', 4 * np.array([width*0.5, width]) / (t/10)**0.25)
    r2 = _nonerand(None, 'beta33', [1, 2]) * (t/10)
    cs = _nonerand(cs, 'beta33', np.array([16, 20]) * (t/10)**0.5 + (cl/t) * 10) 
    css1 = (cs**2 - (t/2 + cl)**2)**0.5
    csp1 = _nonerand(None, 'beta33', [cs/5, cs/4]) * (t/10)**0.5 + 1

    r1 = _nonerand(None, 'bool')
    if r1 or r != None:
        r1 = _nonerand(r, 'beta33', [csp1 * 0.1 * (t/10)**1.5, csp1 * 0.2 * (t/10)**1.5 ]) 
    else:
        r1 = 0

    csp2 = csp1 + width
    css2 = css1 + gap

    crv_type = _nonerand(crv_type, 'int', [1,3])

    if crv_type == 1:
        nf = _nonerand(None, 'beta33', [0.6, 0.8])
        cgenarg1 = fnb.gen_pchip([0.34*nf, 0.56*nf, 0.58*nf])
        cgenarg2 = fnb.gen_pchip([0.32*nf, 0.54*nf, 0.56*nf])
        cgenfunc1, cgenfunc2 = 'median', 'median'
    elif crv_type == 2:
        nf = _nonerand(None, 'int', [5, 11])
        cgenarg1 = [nf, int(nf/1.4)]
        cgenarg2 = cgenarg1
        cgenfunc1, cgenfunc2 = 'bezier', 'bezier'
    elif crv_type == 3:
        cgenarg1 = 4*gap
        cgenarg2 = 'dakka'
        cgenfunc1, cgenfunc2 = 'arc_p', 'arc_tang'
    
    arg1 = [css1, csp1, cgenfunc1, cgenarg1, r1]
    arg2 = [divx, css2, csp2, cgenfunc2, cgenarg2, r2]

    return flg.hld_gen(afl, hld.le_slot, arg2) + flg.hld_gen(afl, hld.slat, arg1)


def fixed_slat(afl: flg.Airfoil, divx = [60, 45], cs = None, crv_type = None, d = None, h = None, dtheta = None):
    """
    Return a randomly generated leading edge fixed slat.

    Args:
        afl (Airfoil): airfoil
        cs (float): the chord of the slat
        crv_type (int): 1 - 3 median, bezier, arc
        d (float): the x distance of the slat from the leading edge
        h (float): the y distance of the slat from the leading edge
        dtheta (float): the angle of rotation of the slat

    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '<')[:,1])
    cl = max(gmt.crv_cut(flg.camberline(afl), 50, '<')[:,1])
    crv_type = _nonerand(crv_type, 'int', [1, 3])

    if crv_type == 1:
        nf = _nonerand(None, 'beta11', [0.3, 1.3])
        rs = nf * ((t/10)**0.5 + (cl/10)**0.5)
        cgenarg = fnb.gen_pchip([0.52*nf, 0.54*nf, 0.56*nf])
        cgenfunc = 'median'
    elif crv_type == 2:
        nf = _nonerand(None, 'int', [5, 11])
        rs = (t/10)**0.5 + (cl/10)**0.5
        cgenarg = [nf, int(nf/2)]
        cgenfunc = 'bezier'
    elif crv_type == 3:
        rs = (t/10)**0.5 + (cl/10)**0.5
        cgenarg = _nonerand(None, 'beta33', [3, 10])
        cgenfunc = 'arc_p'
    
    cs = _nonerand(cs, 'beta33', [9, 17])
    csp = 0.5
    css = _nonerand(None, 'beta33', [csp + 5, 50])
    d = _nonerand(d, 'beta33', [5, 10])
    h = _nonerand(h, 'beta33', np.array([3, 6]) * (t/10 + cl/10)) 
    dtheta = _nonerand(dtheta, 'beta33', [-2, 10])
    dtheta = np.radians(dtheta)

    args = [css, csp, cgenfunc, cgenarg, rs, True]
    sloot = flg.hld_gen(afl, hld.slat, args)
    sloot = flg.crv2gs(sloot)[0]
    sloot = flg.gs2afl(sloot)
    sloot.default_state()
    sloot.transform([0,0], dtheta, [cs/100, cs/100], [-d-cs, h])
    sloot = [[sloot.points[sloot.squencs[0]]]]

    return flg.hld_gen(afl, hld.bare_le, [divx]) + sloot


def act_slat(afl: flg.Airfoil, divx = [60, 45], cs = None, crv_type = None, gap = None, actheta = None, dtheta = None):
    """
    Return a randomly generated leading edge actuating slat.

    Args:
        afl (Airfoil): airfoil
        cs (float): the chord of the slat
        crv_type (int): 1 - 3, median, bezier, arc
        gap (float): the distance of the slat from the leading edge
        actheta (float): the angle of actuation
        dtheta (flaot): the angle of rotation of the slat

    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '<')[:,1])
    cl = max(gmt.crv_cut(flg.camberline(afl), 50, '<')[:,1])

    crv_type = _nonerand(crv_type, 'int', [1,3])
    cs = _nonerand(cs, 'beta33', np.array([12,16])*(t/10)**0.5 + (cl/t) * 10) 
    css = (cs**2 - (t/2 + cl)**2)**0.5
    csp = _nonerand(None, 'beta33', [1, 6])

    if crv_type == 1:
        nf = _nonerand(None, 'beta33', [0.8, 1.2])
        cgenarg = fnb.gen_pchip([0.52*nf, 0.54*nf, 0.56*nf])
        cgenfunc = 'median'
        thetaf = 0.96
    elif crv_type == 2:
        nf = _nonerand(None, 'int', [5, 11])
        cgenarg = [nf, int(csp * nf/6)]
        cgenfunc = 'bezier'
        thetaf = 1
    elif crv_type == 3:
        cgenarg = _nonerand(None, 'beta33', [3, 10])
        cgenfunc = 'arc_p'
        thetaf = 1.2

    args1 = [css, csp, cgenfunc, cgenarg, 0]
    args2 = [divx, css, csp, cgenfunc, cgenarg, 0]
    sloot = flg.hld_gen(afl, hld.slat, args1)
    bood = flg.hld_gen(afl, hld.le_slot, args2)
    p0 = gmt.crcl_fit(sloot[0][1])[0]
    arctheta = - gmt.vectorangle(sloot[0][1][-1]-p0, sloot[0][1][0]-p0)
    actheta = _nonerand(actheta, 'beta33', [0.6*arctheta, 0.7*arctheta]) * thetaf / ((t/12.5)**0.35 + 0.2 * cl/t)
    dtheta = - _nonerand(None, 'beta22', [3.5, 4]) * (arctheta - actheta) * (actheta/arctheta)**2 * ((t/12.5)**0.5 + 0.5 * cl/t)
    gap = _nonerand(gap, 'beta33', [1, 3]) + cs * np.sin(-dtheta) * ((t/12.5)**0.5 + 0.2 * cl/t)
    tv = gap * gmt.bisector_vct(sloot[0][1][0]-p0, sloot[0][1][-1]-p0)

    sloot = sloot[0]
    sloot = [gmt.rotate(sloot[0], sloot[1][0], dtheta), gmt.rotate(sloot[1], sloot[1][0], dtheta)]  
    sloot = [gmt.translate(sloot[0], tv), gmt.translate(sloot[1], tv)]
    sloot = [gmt.rotate(sloot[0], p0, actheta), gmt.rotate(sloot[1], p0, actheta)]
    sloot = [sloot]

    return bood + sloot


def max_slot(afl: flg.Airfoil, divx = [60, 45], cs = None, crv_type = None, gap = None, width = None, r = None):
    """
    Return a randomly generated leading edge maxwell slot.

    Args:
        afl (Airfoil): airfoil
        cs (float): the chord of the slat
        crv_type (int): 1 - 3, median, bezier, arc
        gap (float): the distance of the slat from the leading edge suction side
        width (float): the distance of the slat from the leading edge pressure side

    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '<')[:,1])
    cl = max(gmt.crv_cut(flg.camberline(afl), 50, '<')[:,1])

    crv_type = _nonerand(crv_type, 'int', [1,3])
    cs = _nonerand(cs, 'beta33', [18*(t/10)**0.5, 22*(t/10)**0.5]) + (cl/t) * 10
    gap = _nonerand(gap, 'beta33', [0.7, 1.3])
    css = (cs**2 - (t/2 + cl)**2)**0.5
    csp1 = _nonerand(None, 'beta33', [3, 4]) * (t/10) ** 0.5
    width = _nonerand(width, 'beta33', [7, 9])
    csp2 = csp1 + width
    dtheta = 1.2 * gap/cs

    if crv_type == 1:
        nf = _nonerand(None, 'beta33', [0.6, 0.8])
        cgenarg1 = fnb.gen_pchip([0.34*nf, 0.64*nf, 0.58*nf])
        cgenarg2 = fnb.gen_pchip([0.32*nf, 0.54*nf, 0.56*nf])
        cgenfunc1, cgenfunc2 = 'median', 'median'
        r = 0.55 * csp1
    elif crv_type == 2:
        nf = _nonerand(None, 'int', [5, 11])
        cgenarg1 = [nf, int(nf/1.4)]
        cgenarg2 = [nf, int(nf)]
        cgenfunc1, cgenfunc2 = 'bezier', 'bezier'
        r = 0.5 * csp1
    elif crv_type == 3:
        cgenarg1 = 2*gap
        cgenarg2 = 'dakka'
        cgenfunc1, cgenfunc2 = 'arc_p', 'arc_tang'
        r = 0.85 * csp1

    args1 = [css, csp1, cgenfunc1, cgenarg1, r]
    args2 = [divx, css, csp2, cgenfunc2, cgenarg2, 0]
    sloot = flg.hld_gen(afl, hld.slat, args1)
    bood = flg.hld_gen(afl, hld.le_slot, args2)
    aflsloot = flg.crv2gs(sloot)
    aflsloot = flg.gs2afl(aflsloot[0])
    p0 = flg.le_crcl(aflsloot)[0]
    sloot[0] = [gmt.rotate(sloot[0][0], p0, dtheta)]

    return bood + sloot


# TRAILING EDGE ARRAYS
def bare_te(afl: flg.Airfoil, divx = [60, 45]):
    """
    Returns the bare trailing edge, just like in hld. Nothing more, just repeated here for the sake of consistency.
    """
    return flg.hld_gen(afl, hld.bare_te, [divx])


def te_flap(afl: flg.Airfoil, divx = [60, 45], cf = None, dtheta = None):
    """
    Return a randomly generated trailing edge flap.

    Args:
        afl (Airfoil): airfoil
        cf (float): the chord of the flap
        dtheta (float): angle of rotation

    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '<')[:,1])
    cf = _nonerand(cf, 'beta33', [10 * (t/10)**0.5, 30 * (t/10)**0.5])
    dtheta = np.radians(_nonerand(dtheta, 'beta33', [15, 35]))
    arg = [divx, cf, dtheta]
    return flg.hld_gen(afl, hld.te_flap, arg)


def split_flap(afl: flg.Airfoil, divx = [60, 45], cf = None, dtheta = None):
    """
    Return a randomly generated split flap.

    Args:
        afl (Airfoil): airfoil
        cf (float): the chord of the flap
        dtheta (float): angle of rotation

    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '>')[:,1])
    cf = _nonerand(cf, 'beta33', [10, 40])
    dtheta = np.radians(_nonerand(dtheta, 'beta33', [45, 60]))
    ft = 1.5/t
    arg = [divx, cf, dtheta, ft]
    return flg.hld_gen(afl, hld.split_flap, arg)


def zap_flap(afl: flg.Airfoil, divx = [60, 45], cf = None, dtheta = None, dx = None, gap = None):
    """
    Return a randomly generated zap flap.

    Args:
        afl (Airfoil): airfoil
        cf (float): the chord of the flap
        dtheta (float): angle of rotation
        gap (float): the gap between the flap and the airfoil body

    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '>')[:,1])
    cf = _nonerand(cf, 'beta33', [15, 40])
    dx = _nonerand(dx, 'beta33', [0.4*cf , 0.8 * cf])
    dtheta = np.radians(_nonerand(dtheta, 'beta33', [15, 70]))
    gapb = _nonerand(None, 'int', [0,1])
    gap = _nonerand(gap, 'beta33', [gapb*0.25 * dx**0.5, gapb*0.5 * dx**0.5])
    ft = 1.5/t
    r = 0.15
    arg = [divx, cf, dtheta, ft, dx, gap, r]
    return flg.hld_gen(afl, hld.zap_flap, arg)


def junk_flap(afl: flg.Airfoil, divx = [60, 45], cf = None, crv_type = None, d = None, h = None, dtheta = None):
    """
    Return a randomly generated junkers flap.

    Args:
        afl (Airfoil): airfoil
        cf (float): the chord of the flap
        crv_type (int): 1 - 4 median, bezier, arc, mirror
        d (float): the x distance of the flap from the trailing edge
        h (float): the y distance of the flap from the trailing edge
        dtheta (float): the angle of rotation of the flap

    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '>')[:,1])
    cl = max(gmt.crv_cut(flg.camberline(afl), 50, '>')[:,1])
    crv_type = _nonerand(crv_type, 'int', [1, 3])

    if crv_type == 1:
        nf = _nonerand(None, 'beta11', [0.3, 1.3])
        rs = nf * ((t/10)**0.5 + (cl/10)**0.5)
        cgenarg = fnb.gen_pchip([0.52*nf, 0.54*nf, 0.56*nf])
        cgenfunc = 'median'
    elif crv_type == 2:
        nf = _nonerand(None, 'int', [5, 11])
        df = _nonerand(None, 'beta33', [0.1, 1.4])
        rs = (t/10)**0.5 + (cl/10)**0.5
        cgenarg = [nf, int(nf/df)]
        cgenfunc = 'bezier'
    elif crv_type == 3:
        cgenarg = _nonerand(None, 'beta33', [3, 10])
        cgenfunc = 'arc_p'
        rs = (t/10)**0.5 + (cl/10)**0.5
    
    cf = _nonerand(cf, 'beta33', [15, 30])
    csp = 0.5
    css = _nonerand(None, 'beta33', [csp + 5, 50])
    d = _nonerand(d, 'beta33', [-1, 3.5])
    h = _nonerand(h, 'beta33', [2, 3]) * (t/10 + cl/10) 
    dtheta = _nonerand(dtheta, 'beta33', [15, 60])
    dtheta = -np.radians(dtheta)

    args = [css, csp, cgenfunc, cgenarg, rs, True]
    sloot = flg.hld_gen(afl, hld.slat, args)

    sloot = flg.crv2gs(sloot)[0]
    sloot = flg.gs2afl(sloot)
    sloot.default_state()
    sloot.transform([0,0], dtheta, [cf/100, cf/100], [100+d, -h])
    floop = [[sloot.points[sloot.squencs[0]]]]

    return flg.hld_gen(afl, hld.bare_te, [divx]) + floop


def flap_1slot(afl: flg.Airfoil, divx = [60, 45], cf = None, crv_type = None, gap = None, dtheta = None):
    """
    Return a randomly generated single slotted flap.

    Args:
        afl (Airfoil): airfoil
        cf (float): the chord of the flap
        crv_type (int): 1 - 3 marriage, arc_tang, arc_p
        d (float): the x distance of the flap from the trailing edge
        h (float): the y distance of the flap from the trailing edge
        dtheta (float): the angle of rotation of the flap

    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '>')[:,1])
    crv_type = _nonerand(crv_type, 'int', [1,3])
    gap = _nonerand(gap, 'beta33', np.array([0.5, 2.5])*(t/10)**0.5 + 0.5)

    if crv_type == 1:
        cgenfunc = 'marriage'
        cgenarg = [0.2, np.array([6, 3, 12, 4])*(t/10)**0.5]
    elif crv_type == 2:
        cgenfunc = 'arc_tang'
        cgenarg = 'dakka'
    elif crv_type == 3:
        cgenfunc = 'arc_p'
        cgenarg = 4*gap*(t/10)**0.5

    cf = _nonerand(cf, 'beta23', [20*(t/10)**0.25, 35*(t/10)**0.25])
    cfp = cf
    cfdif = _nonerand(None, 'beta33', [0.14*cfp, 0.28*cfp]) * (t/10)**0.75
    cfs = cfp - cfdif
    dtheta = _nonerand(dtheta, 'beta33', [15, 50])
    dtheta = -np.radians(dtheta)
    rf = t/6 * cfs/50
    rb = _nonerand(None, 'int', [0, 1]) * _nonerand(None, 'beta33', [0, 1])

    flap_args = [cfs, cfp, cgenfunc, cgenarg, rf]
    body_args = [divx, cfs, cfp, cgenfunc, cgenarg, rb]

    bood = flg.hld_gen(afl, hld.te_slot, body_args)
    floop = flg.hld_gen(afl, hld.flap, flap_args)
    
    slot_curve = flg.hld_gen(afl, hld.te_slot, [divx, cfs, cfp, cgenfunc, cgenarg, 0])[0][1]
    p0 = gmt.crcl_fit(slot_curve)[0]
    tv = gmt.rotate(- gap * gmt.bisector_vct(slot_curve[0]-p0, slot_curve[-1]-p0), [0,0], np.pi/10)

    floop = floop[0][0]
    floop = gmt.rotate(floop, p0, dtheta)
    floop = gmt.translate(floop, tv)
    floop = [[floop]]

    return bood + floop


def flap_2slot_ff(afl: flg.Airfoil, divx = [60, 45], cf1 = None, dtheta1 = None, cf2 = None):
    """
    Return a randomly generated double slotted flap, with a fixed foreflap.

    Args:
        afl (Airfoil): airfoil
        cf1 (float): the chord of the whole flap
        cf2 (float): the chord of the fore flap
        dtheta1 (float): the angle of rotation of the flap

    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '>')[:,1])
    cl = max(gmt.crv_cut(flg.camberline(afl), 50, '>')[:,1])
    cf1 = _nonerand(cf1, 'beta23', [20, 35])
    cavfac = 1/3
    cgenfunc = 'marriage'
    cgenarg = [cf1 * cavfac * 1.1, [1, 1, 3, 1]]

    cfp = cf1 * (1- cavfac)
    cfs = _nonerand(None, 'beta33', [0.6*cfp, 0.9 * cfp])
    cf2 = _nonerand(cf2, 'beta33', [20, 30]) * (t/10)**0.5
    gap1 = _nonerand(None, 'beta33', [0.7, 2.2])
    gap2 = _nonerand(None, 'beta33', [gap1, 1.4*gap1]) * 100 / cf1
    dtheta1 = _nonerand(dtheta1, 'beta33', [55, 55])
    dtheta1 = -np.radians(dtheta1)
    rf = t/6 * cfs/50
    rb = 0
    divx2 = [75, 45]

    flap_args = [cfs, cfp, cgenfunc, cgenarg, rf]
    body_args = [divx, cfs, cfp, cgenfunc, cgenarg, rb]

    bood = flg.hld_gen(afl, hld.te_slot, body_args)
    floop = flg.hld_gen(afl, hld.flap, flap_args)

    floopafl = flg.gs2afl(flg.crv2gs(floop)[0])
    floop = flg.foilpatch(_fixed_fore_flap(floopafl, divx2, cf2, gap2), bare_te(floopafl, divx2))

    p2 = bood[0][-1][0]
    p1 = floop[1][0][np.argmax(np.hypot(floop[1][0][:,0] - floopafl.points[0,0], floop[1][0][:,1] - floopafl.points[0,1]))]  # this gets the leading edge point
    p0, r0 = flg.le_crcl(flg.gs2afl(flg.crv2gs(floop)[1]))
    tv = p2 - p1 - np.array([0, gap1 + r0])
    floop[0][0] = gmt.translate(gmt.rotate(floop[0][0], p0, dtheta1), tv)
    floop[1][0] = gmt.translate(gmt.rotate(floop[1][0], p0, dtheta1), tv)
    
    return bood + floop


def flap_2slot_af(afl: flg.Airfoil, divx = [60, 45], cf1 = None, cf2 = None, dtheta1 = None, dtheta2 = None):
    """
    Return a randomly generated double slotted flap, with an actuating foreflap.

    Args:
        afl (Airfoil): airfoil
        cf1 (float): the chord of the whole flap
        cf2 (float): the chord of the fore flap
        dtheta1 (float): the angle of rotation of the main flap
        dtheta2 (float): the angle of rotation of the fore flap

    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '>')[:,1])
    cl = max(gmt.crv_cut(flg.camberline(afl), 50, '>')[:,1])
    cf1 = _nonerand(cf1, 'beta23', [20, 35])
    cavfac = 1/3
    cgenfunc = 'marriage'
    cgenarg = [cf1 * cavfac * 1.1, [1, 1, 2, 1]]
    cfp = cf1 * (1 - cavfac)
    cfs = _nonerand(None, 'beta33', [0.6*cfp, 0.9 * cfp])
    cf2 = _nonerand(cf2, 'beta33', [30, 30])
    gf1 = 1 / afl.default_state(transform=False)[1]
    gap1 = _nonerand(None, 'beta33', [1, 2.5]) * gf1
    gap2 = _nonerand(None, 'beta33', [0.8*gap1, gap1]) * 2
    dtheta1 = _nonerand(dtheta1, 'beta33', [50, 50])
    dtheta1 = -np.radians(dtheta1)
    dtheta2 = _nonerand(dtheta2, 'beta33', [dtheta1/2, dtheta1/2])
    rf = t/6 * cfs/50
    t = (cfs/cfp)**0.25
    rb = 0
    divx2 = [65, 45]

    flap_args = [cfs, cfp, cgenfunc, cgenarg, rf]
    body_args = [divx, cfs, cfp, cgenfunc, cgenarg, rb]

    bood = flg.hld_gen(afl, hld.te_slot, body_args)
    floop = flg.hld_gen(afl, hld.flap, flap_args)

    floopafl = flg.gs2afl(flg.crv2gs(floop)[0])
    floop = flg.foilpatch(_act_fore_flap(floopafl, divx2, cf2, dtheta1-dtheta2, gap2), bare_te(floopafl, divx2))

    p2 = bood[0][-1][0]
    p1 = floop[1][0][np.argmax(np.hypot(floop[1][0][:,0] - floopafl.points[0,0], floop[1][0][:,1] - floopafl.points[0,1]))]  # this gets the leading edge point
    p0, r0 = flg.le_crcl(flg.gs2afl(flg.crv2gs(floop)[1]))
    tv = p2 - p1 - np.array([0, gap1 + r0])
    floop[0][0] = gmt.translate(gmt.rotate(floop[0][0], p0, dtheta1), tv)
    floop[1][0] = gmt.translate(gmt.rotate(floop[1][0], p0, dtheta1), tv)
    
    return bood + floop


def flap_2slot_sa(afl: flg.Airfoil, divx = [60, 45], cf1 = None, dtheta1 = None, cf2 = None, dtheta2 = None):
    """
    Return a randomly generated double slotted flap, with an actuating afterflap.

    Args:
        afl (Airfoil): airfoil
        cf1 (float): the chord of the whole flap
        cf2 (float): the chord of the after flap
        dtheta1 (float): the angle of rotation of the main flap
        dtheta2 (float): the angle of rotation of the after flap

    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '>')[:,1])
    cl = max(gmt.crv_cut(flg.camberline(afl), 50, '>')[:,1])
    cf1 = _nonerand(cf1, 'beta23', [20, 35])
    cavfac = 1/3
    cgenfunc = 'marriage'
    cgenarg = [cf1 * cavfac * 1.1, [1, 1, 1, 1]]
    cfp = cf1 * (1 - cavfac)
    cfs = _nonerand(None, 'beta33', [0.6*cfp, 0.74*cfp])
    cf2 = _nonerand(cf2, 'beta33', [40, 40])
    gap1 = _nonerand(None, 'beta33', [1, 2.5])
    gap2 = _nonerand(None, 'beta33', [0.8*gap1, 1.2*gap1])
    dtheta1 = _nonerand(dtheta1, 'beta33', [40, 40])
    dtheta1 = -np.radians(dtheta1)
    dtheta2 = _nonerand(dtheta2, 'beta33', [dtheta1/2, dtheta1/2])
    rf = t/6 * cfs/50
    divx2 = [55, 40]

    flap_args = [cfs, cfp, cgenfunc, cgenarg, rf]
    body_args = [divx, cfs, cfp, cgenfunc, cgenarg, 0]

    bood = flg.hld_gen(afl, hld.te_slot, body_args)
    floop = flg.hld_gen(afl, hld.flap, flap_args)
    floopafl = flg.gs2afl(flg.crv2gs(floop)[0])

    p2 = bood[0][-1][0]
    p1 = floop[0][0][np.argmax(np.hypot(floop[0][0][:,0] - floopafl.points[0,0], floop[0][0][:,1] - floopafl.points[0,1]))]  # this gets the leading edge point
    p0, r0 = flg.le_crcl(flg.gs2afl(flg.crv2gs(floop)[0]))
    tv = p2 - p1 - np.array([0, (gap1 + t * cf1 / 200) + r0])

    floop = flg.foilpatch(bare_le(floopafl, divx2), _slot_after_flap(floopafl, divx2, cf2, dtheta2, gap2))

    floop[0][0] = gmt.translate(gmt.rotate(floop[0][0], p0, dtheta1), tv)
    floop[0][1] = gmt.translate(gmt.rotate(floop[0][1], p0, dtheta1), tv)
    floop[1][0] = gmt.translate(gmt.rotate(floop[1][0], p0, dtheta1), tv)

    return bood + floop


def fowler_1slot(afl: flg.Airfoil, divx = [60, 45], cf = None, crv_type = None, dx = None, dy = None, dtheta = None):
    """
    Return a randomly generated single slotted fowler flap.

    Args:
        afl (Airfoil): airfoil
        cf (float): the chord of the flap
        crv_type (int): 1 - 3 median, bezier, arc
        d (float): the x distance of the flap from the trailing edge
        h (float): the y distance of the flap from the trailing edge
        dtheta (float): the angle of rotation of the flap

    """
    t = max(gmt.crv_cut(flg.thickness(afl), 50, '>')[:,1])

    cgenfunc = 'marriage'
    cgenarg = [0.2, [1, 4, 16, 1]]

    cf = _nonerand(cf, 'beta23', [20, 40])
    cfp = cf
    cfs = _nonerand(None, 'beta32', [0, 5])
    dtheta = _nonerand(dtheta, 'beta33', [15, 50])
    dtheta = -np.radians(dtheta)
    rf = t/6 * cfp/50
    rb = _nonerand(None, 'int', [0, 1]) * _nonerand(None, 'beta33', [0, 1])
    dx = _nonerand(dx, 'beta33', [-1.5, 1.5])
    dy = _nonerand(dy, 'beta33', [0.5, 3.5])

    flap_args = [cfs, cfp, cgenfunc, cgenarg, rf]
    body_args = [divx, cfs, cfp, cgenfunc, cgenarg, rb]

    bood = flg.hld_gen(afl, hld.te_slot, body_args)
    floop = flg.hld_gen(afl, hld.flap, flap_args)
    
    floopc = floop[0][0]
    ftei = np.argmax(np.hypot(floopc[:,0] - floopc[0,0], floopc[:,1] - floopc[0,1]))
    p1 = floopc[ftei]
    p2 = bood[0][-1][0]
    p0, r0 = gmt.crcl_fit(floopc[ftei-3:ftei+3])
    tv = p2 - p1 + np.array([dx, -dy -t * cf/125])
    floopc = gmt.rotate(floopc, p0, dtheta)
    floopc = gmt.translate(floopc, tv)
    floop = [[floopc]]

    return bood + floop


def fowler_2slot():
    pass


def fowler_3slot():
    pass
