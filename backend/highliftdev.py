#--------- READ ME----------------------------------------------------------------
# A package containing the functions to help generate all the high lift device
# types. These can be used directly, or, can be called by other scripts on an
# automated process like implemented in the randsectgen package.
#---------------------------------------------------------------------------------

import numpy as np
import geometrics as gmt
import curvegen as crvgen

# Some documentation for you
# cgenarg depending on cgenfunc
# - median: the wieght function weight_fun
# - bezier: [m, n] with m the number of control points (can also be the string 'all') and n the loops of the curve gen algorithm
# - arc_p: the dx for the equation xp = css + dx, the point needed will be found by interpolating the suction side at xp
# - arc_tang: the cgenarg will not be used
# - marriage: [x0, weights] x0 is the additional depth and its always positive or 0. weights is a list with the 4 weights going into the marriage factor frame, it dictates the shape of the curve
# - input: the input points of the curve


# PRIVATE STUFF
def _varcam_intersect(pflap_curve, pre_curve):
    """
    Find the intersection between the pressure sides of the flap and the main body.
    """
    # First check if intersection is beyond both curves
    lf1 = np.polyfit(pflap_curve[-2:,0], pflap_curve[-2:,1], 1)
    lf2 = np.polyfit(pre_curve[0:2,0], pre_curve[0:2,1], 1)
    pint = gmt.lnr_inters(lf1, lf2)
    if (pint[0] < pre_curve[0,0]) and  (pint[0] > pflap_curve[-1,0]):
        return [pint, np.shape(pflap_curve)[0], 0]
    
    # Else, check if intersection outside LE curve and inside BS curve
    for i in range(0, np.shape(pre_curve)[0]-1):
        lf2 = np.polyfit(pre_curve[i:i+2,0], pre_curve[i:i+2,1], 1)
        pint = gmt.lnr_inters(lf1, lf2)
        if (pint[0] > pflap_curve[-1,0]) and ((pint[0] > pre_curve[i,0]) == (pint[0] < pre_curve[i+1,0])):
            return [pint, np.shape(pflap_curve)[0], i+1]
        
    # Else, check if intersection inside both curves
    int_data = gmt.crv_inters(pflap_curve, pre_curve)
    if int_data[0]:
        return [int_data[1], int_data[2]+1, int_data[3]+1]
    
    # Lastly try to find last intersection inside LE and outside BS curve
    lf2 = np.polyfit(pre_curve[0:2,0], pre_curve[0:2,1], 1)
    for i in range(np.shape(pflap_curve)[0]-2, -1, -1):
        lf1 = np.polyfit(pflap_curve[i:i+2 ,0], pflap_curve[i:i+2 ,1], 1)
        pint = gmt.lnr_inters(lf1, lf2)
        if (pint[0] < pre_curve[0,0]) and ((pint[0] > pflap_curve[i,0]) == (pint[0] < pflap_curve[i+1,0])):
            return [pint, i, 0]
    
    # If all else fails
    raise Exception('Couldnt find intersection between bot_le_curve and pre_curve.')


def _division_line(sides, divx):
    return np.polyfit([divx[0], divx[1]], [gmt.cbs_interp(sides[0], divx[0], 0)[0][1], gmt.cbs_interp(sides[1], divx[1], 0)[0][1]], 1)


# HIGH LIFT DEVICES

# LEADING EDGE GOEMETRIES
def bare_le(sides, divx):
    """
    Leave the leading edge as it is.

    Args:
        sides: the suction and pressure side as returned by Airfoil.sides()
    
    Returns:
        c ([ndarray]): the curve of the leading edge in a list

    """
    curve = np.vstack(sides)
    curve = gmt.crv_cut(curve, _division_line(sides, divx), '>=')
    return [[curve]]


def le_flap1(sides, divx, css, csp, dtheta):
    """
    Generate a leading edge flap, hinging on the pressure side.

    Args:
        sides: the suction and pressure side as returned by Airfoil.sides()
        css (float): 0 < css < 50, the suction side flap chord
        csp (float): 0 < csp < 50, the pressure side flap chord
        dtheta (float): the angle at which the flap bends
        
    Returns:
        c ([ndarray]): the curves of the leading edge in a list

    """
    # Surface split point
    sp = gmt.cbs_interp(sides[0], css, 0)[0]
    pp = gmt.cbs_interp(sides[1], csp, 0)[0]
    # Surfaces
    suc_curve = np.vstack((gmt.crv_cut(gmt.crv_cut(sides[0], css, '>'), _division_line(sides, divx), '>='), sp))
    pre_curve = np.vstack((pp, gmt.crv_cut(gmt.crv_cut(sides[1], csp, '>'), _division_line(sides, divx), '>=')))
    flap_curve = np.vstack((sp, gmt.crv_cut(sides[0], css, '<'), gmt.crv_cut(sides[1], csp, '<'), pp))
    # Move flap
    flap_curve = gmt.rotate(flap_curve, pp, dtheta)
    # Connect flap to suction side
    lf1 = np.polyfit([flap_curve[0,0], flap_curve[1,0]], [flap_curve[0,1], flap_curve[1,1]], 1)
    lf2 = np.polyfit([suc_curve[-1,0], suc_curve[-2,0]], [suc_curve[-1,1], suc_curve[-2,1]], 1)
    pint = gmt.lnr_inters(lf1, lf2)
    sf_curve = gmt.bezier([suc_curve[-1], pint, flap_curve[0]])(np.linspace(0,1,5))
    # Patch and return
    curve1 = gmt.patch(gmt.patch(suc_curve, sf_curve), flap_curve)
    return [[curve1, pre_curve]]
    

def le_flap2(sides, divx, css, csp, dtheta):
    """
    Generate a leading edge flap, hinging on the suction side.

    Args:
        sides: the suction and pressure side as returned by Airfoil.sides()
        css (float): 0 < css < 50, the suction side flap chord
        csp (float): 0 < csp < 50, the pressure side flap chord
        dtheta (float): the angle at which the flap bends
        
    Returns:
        c ([ndarray]): the curves of the leading edge in a list

    """
    # Surface split point
    sp = gmt.cbs_interp(sides[0], css, 0)[0]
    pp = gmt.cbs_interp(sides[1], csp, 0)[0]
    # Surfaces
    suc_curve = np.vstack((gmt.crv_cut(gmt.crv_cut(sides[0], css, '>'), _division_line(sides, divx), '>='), sp))
    pre_curve = np.vstack((pp, gmt.crv_cut(gmt.crv_cut(sides[1], csp, '>'), _division_line(sides, divx), '>=')))
    sflap_curve = np.vstack((sp, gmt.crv_cut(sides[0], css, '<')))
    pflap_curve = np.vstack((gmt.crv_cut(sides[1], csp, '<'), pp))
    # Move flap suction side
    for i in range(0 ,np.shape(sflap_curve)[0]):
        theta = dtheta * ((css - sflap_curve[i,0])/css)
        sflap_curve[i] = gmt.rotate(sflap_curve[i], sp, theta)
    # Move flap pressure side
    pflap_curve = gmt.rotate(pflap_curve, sp, dtheta)
    # Find intersection
    pint, j1, j2 = _varcam_intersect(pflap_curve, pre_curve)
    # Remove curve points past intersection
    pflap_curve = pflap_curve[0:j1]
    pre_curve = pre_curve[j2:]
    # Add intersection point to curves
    pre_curve = np.vstack((pint, pre_curve))
    pflap_curve = np.vstack((pflap_curve, pint))
    # Patch and return
    curve1 = np.vstack((gmt.patch(suc_curve, sflap_curve), pflap_curve))
    return [[curve1, pre_curve]]


def le_flap3(sides, divx, css, csp, dtheta):
    """
    Generate a leading edge flap, with smooth curved, variable camber geometry.

    Args:
        sides: the suction and pressure side as returned by Airfoil.sides()
        css (float): 0 < css < 50, the suction side flap chord
        csp (float): 0 < csp < 50, the pressure side flap chord
        dtheta (float): the angle at which the flap bends
        
    Returns:
        c ([ndarray]): the curves of the leading edge in a list

    """
    # Surface split point
    sp = gmt.cbs_interp(sides[0], css, 0)[0]
    pp = gmt.cbs_interp(sides[1], csp, 0)[0]
    # Surfaces
    suc_curve = np.vstack((gmt.crv_cut(gmt.crv_cut(sides[0], css, '>'), _division_line(sides, divx), '>='), sp))
    pre_curve = np.vstack((pp, gmt.crv_cut(gmt.crv_cut(sides[1], csp, '>'), _division_line(sides, divx), '>=')))
    sflap_curve = np.vstack((sp, gmt.crv_cut(sides[0], css, '<')))
    pflap_curve = np.vstack((gmt.crv_cut(sides[1], csp, '<'), pp))
    # Move flap suction side
    rot_cntr = sp
    for i in range(0, np.shape(sflap_curve)[0]):
        theta = dtheta * (1 - sflap_curve[i,0] / css)
        sflap_curve[i,:] = gmt.rotate(sflap_curve[i,:], rot_cntr, theta)
    # Move flap pressure side
    for i in range(0, np.shape(pflap_curve)[0]):
        theta = dtheta * (1 - pflap_curve[i,0] / csp)
        pflap_curve[i,:] = gmt.rotate(pflap_curve[i,:], rot_cntr, theta)
    # Patch and return
    return [[np.vstack((gmt.patch(suc_curve, sflap_curve), gmt.patch(pflap_curve, pre_curve)))]]


def kruger():
    """
    NOT YET IMPLEMENTED.
    Generate a leading edge kruger flap.

    Args:
        
    Returns:

    """


def le_slot(sides, divx, css, csp, cgenfunc, cgenarg, r):
    """
    Cut the leading edge short, so the resulting geometry can be used in combination with a slat.

    Args:
        sides: the suction and pressure side as returned by Airfoil.sides()
        css (float): 0 < css < 50, the suction side flap chord
        csp (float): 0 < csp < 50, the pressure side flap chord
        cgenfunc (str): string corelating to the curve generation fucntion that will be used. can be one of: {'median', 'bezier', 'arc_p', 'arc_tang', 'input'}
        cgenarg: The arguments needed for the function (described at the top of the module)
        r (float): the radius of the fillet of the leading edge curve and the pressure side curve, must be above 0, if 0 the curves will be returned as separate
        
    Returns:
        c ([ndarray]): the curves of the leading edge in a list

    """
    # Curve point number constant
    n = 20
    # Surface split point
    sp = gmt.cbs_interp(sides[0], css, 0)[0]
    pp = gmt.cbs_interp(sides[1], csp, 0)[0]
    # Surfaces
    suc_curve = np.vstack((gmt.crv_cut(gmt.crv_cut(sides[0], css, '>'), _division_line(sides, divx), '>='), sp))
    pre_curve = np.vstack((pp, gmt.crv_cut(gmt.crv_cut(sides[1], csp, '>'), _division_line(sides, divx), '>=')))
    le_curve = np.vstack((sp, gmt.crv_cut(sides[0], css, '<'), gmt.crv_cut(sides[1], csp, '<'), pp))
    # Generate curves
    if cgenfunc == 'median':
        theta = - gmt.vectorangle(le_curve[0]-le_curve[1]) - np.pi/2 + 0.015
        le_curve = crvgen.median(le_curve, theta, cgenarg, n)
    
    elif cgenfunc == 'bezier':
        le_curve = crvgen.bezier(le_curve, *cgenarg, n)

    elif cgenfunc == 'arc_p':
        crvp = gmt.cbs_interp(sides[0], css + cgenarg, 0)[0]
        le_curve = crvgen.arc_p(le_curve, crvp, n)

    elif cgenfunc == 'arc_tang':
        le_curve = crvgen.arc_tang(le_curve, n)

    elif cgenfunc == 'input':
        le_curve = gmt.crv_fit2p(cgenarg, suc_curve[-1], pre_curve[0], proxi_snap=True)

    if r>0:
        return [[gmt.fillet_aprox(suc_curve, gmt.fillet_aprox(le_curve, pre_curve, r, [1, 1])[1], 5, 4)[1]]]
    elif r==0:
        return [[gmt.fillet_aprox(suc_curve, le_curve, 5, 2)[1], pre_curve]]


# TRAILING EDGE GOEMETRIES
def bare_te(sides, divx):
    """
    Leave the trailing edge as it is.

    Args:
        sides: the suction and pressure side as returned by Airfoil.sides()
    
    Returns:
        c ([ndarray]): the curve of the trailing edge in a list

    """
    suc_curve = gmt.crv_cut(sides[0], _division_line(sides, divx), '<')
    pre_curve = gmt.crv_cut(sides[1], _division_line(sides, divx), '<')
    return [[pre_curve, suc_curve]]


def te_flap(sides, divx, cf, dtheta):
    """
    Generate a trailing edge flap

    Args:
        sides: the suction and pressure side as returned by Airfoil.sides()
        cf (float): the flap chord
        dtheta (float): the angle at which the flap bends
        
    Returns:
        c ([ndarray]): the curves of the trailing edge in a list

    """
    cf = 100 - cf
    # Surfaces
    suc_curve = gmt.crv_cut(sides[0], _division_line(sides, divx), '<')
    pre_curve = np.flipud(gmt.crv_cut(sides[1], _division_line(sides, divx), '<'))
    # Joint rotation center, and surface split points
    p0, ptan1, ptan2, i, j = gmt.crcl_tang_2crv(suc_curve, pre_curve, np.polyfit([cf - 10**-3, cf + 10**-3], [-10, 10], 1))[1:]
    # Surfaces
    sflap_curve = np.vstack((suc_curve[0:i+1], ptan1))
    pflap_curve = np.flipud(np.vstack((pre_curve[0:j+1], ptan2)))
    suc_curve = np.vstack((ptan1, suc_curve[i+1:]))
    pre_curve = np.vstack((ptan2, pre_curve[j+1:]))
    # Move flap surfaces
    sflap_curve = gmt.rotate(sflap_curve, p0, -dtheta)
    pflap_curve = gmt.rotate(pflap_curve, p0, -dtheta)
    # Intersection of pre_curve and pflap_curve
    pint, i, j = gmt.crv_inters(pflap_curve, pre_curve)[1:]
    pflap_curve = np.vstack((pint, pflap_curve[i+1:]))
    pre_curve = np.flipud(np.vstack((pint, pre_curve[j+1:])))
    # Connect suction side flap to suction side
    sf_curve = gmt.arc_gen(sflap_curve[-1], ptan1, p0, int(np.ceil(np.degrees(dtheta)/20)))
    # Patch and return
    suc_curve = gmt.patch(gmt.patch(sflap_curve, sf_curve), suc_curve)
    return [[pre_curve, pflap_curve, suc_curve]]


def split_flap(sides, divx, cf, dtheta, ft):
    """
    Generate a trailing edge split flap.

    Args:
        sides: the suction and pressure side as returned by Airfoil.sides()
        cf (float): the flap chord
        dx (float): the displacement of the flap along the chord line
        dtheta (float): the angle at which the flap bends
        ft(float): the thickness factor of the flap 0<ft<1 
        
    Returns:
        c ([ndarray]): the curves of the trailing edge in a list

    """
    cf = 100 - cf
    # Generate median line
    interpoints = gmt.cbs_interp(np.vstack(sides), np.linspace(30,100,71), np.zeros(71, dtype=int), [10**-3, 10**-3])
    med_curve = ft * np.flipud(interpoints[0:70]) + (1-ft) * interpoints[70:]
    # Surface split points
    pp = gmt.cbs_interp(sides[1], cf, 0)[0]
    # Surfaces
    suc_curve = gmt.crv_cut(sides[0], _division_line(sides, divx), '<')
    pre_curve = np.vstack((gmt.crv_cut(gmt.crv_cut(sides[1], _division_line(sides, divx), '<'), cf, '<'), pp))
    pflap_curve = np.vstack((pp, gmt.crv_cut(sides[1], cf, '>')))
    # Moving flap
    pflap_curve = gmt.rotate(pflap_curve, pp, -dtheta)
    sflap_curve = gmt.rotate(med_curve, pp, -dtheta)   
    # Intersection and slice
    pint, i, j = gmt.crv_inters(sflap_curve, med_curve)[1:]
    sflap_curve = np.flipud(np.vstack((pint, sflap_curve[i+1:])))
    med_curve = np.vstack((pint, med_curve[j+1:]))
    return [[pre_curve, pflap_curve, sflap_curve, med_curve, suc_curve]]


def zap_flap(sides, divx, cf, dtheta, ft, dx, gap, r):
    """
    Generate a trailing edge slotted zap flap.

    Args:
        sides: the suction and pressure side as returned by Airfoil.sides()
        cf (float): the flap chord
        dx (float): the displacement of the flap along the chord line
        dtheta (float): the angle at which the flap bends
        ft (float): the thickness factor of the flap 0<ft<1 
        dx (float): the x displacement of th flap
        gap (float): gap >= 0 the y displacement of the flap, if 0, there wont be a slot
        r (float): if gap > 0, the radius of the fillet of the flaps suction and pressure side
        
    Returns:
        c ([ndarray]): the curves of the trailing edge in a list

    """
    cf = 100 - cf
    # Generate median line
    interpoints = gmt.cbs_interp(np.vstack(sides), np.linspace(30,100,71), np.zeros(71, dtype=int), [10**-3, 10**-3])
    med_curve = ft * np.flipud(interpoints[0:70]) + (1-ft) * interpoints[70:]
    # Surface split point
    pp = gmt.cbs_interp(sides[1], cf, 0)[0]
    # Surfaces
    suc_curve = gmt.crv_cut(sides[0], _division_line(sides, divx), '<')
    pre_curve = np.vstack((gmt.crv_cut(gmt.crv_cut(sides[1], _division_line(sides, divx), '<'), cf, '<'), pp))
    # arc center and tangents
    dy = gmt.cbs_interp(med_curve, cf, 0)[0][1] - pp[1]
    p0 = - dy * (pre_curve[-2] - pp) / np.linalg.norm(pre_curve[-2] - pre_curve[-1]) + pre_curve[-1]
    mp = gmt.rotate(pp, p0, -np.pi/2.5)
    # Main body surfaces
    farc = gmt.arc_gen(mp, pp, p0, 4)
    pte_curve = np.vstack((mp, gmt.crv_cut(med_curve, mp[0], '>')))
    pte_curve = gmt.patch(farc, pte_curve)
    # Translation
    tv = gmt.cbs_interp(sides[1], p0[0] + dx, 0)[0] - p0
    # Flap surfaces
    if gap == 0:
        sflap_curve = gmt.translate(gmt.rotate(med_curve, p0, -dtheta), tv)
        pflap_curve = gmt.translate(gmt.rotate(sides[1], p0, -dtheta), tv)
        # Intersection and slicing
        pint, i, j = gmt.crv_inters(sflap_curve, pte_curve)[1:]
        sflap_curve = np.flipud(np.vstack((pint, sflap_curve[i+1:])))
        pte2_curve = np.vstack((pint, pte_curve[j+1:]))
        pint, i, j = gmt.crv_inters(pflap_curve, pte_curve)[1:]
        pflap_curve = np.vstack((pint, pflap_curve[i+1:]))
        pte1_curve = np.vstack((pte_curve[0:j+1], pint))
        return [[pre_curve, pte1_curve, pflap_curve, sflap_curve, pte2_curve, suc_curve]]
    elif gap > 0:
        sflap_curve = pte_curve
        pflap_curve = np.vstack((pp, gmt.crv_cut(sides[1], pp[0], '>')))
        sflap_curve = gmt.translate(gmt.rotate(sflap_curve, p0, -dtheta), tv - np.array([0, gap]))
        pflap_curve = gmt.translate(gmt.rotate(pflap_curve, p0, -dtheta), tv - np.array([0, gap]))
        if r == 0:
            return [[pre_curve, pte_curve, suc_curve], [np.flipud(sflap_curve), pflap_curve]]
        elif r > 0:
            return [[pre_curve, pte_curve, suc_curve], [gmt.fillet_aprox(sflap_curve, pflap_curve, r, 4, [0, 0])[1]]]


def te_slot(sides, divx, cfs, cfp, cgenfunc, cgenarg, r):
    """
    Cut the trailing edge short, so the resulting geometry can be used in combination with a flap.

    Args:
        sides: the suction and pressure side as returned by Airfoil.sides()
        cfs (float): 0 < cfs < 50, the suction side flap chord
        cfp (float): 0 < cfp < 50, the pressure side flap chord
        cgenfunc (str): string corelating to the curve generation fucntion that will be used. can be one of: {'median', 'bezier', 'arc_p', 'arc_tang', 'input'}
        cgenarg: The arguments needed for the function (described at the top of the module)
        r (float): the radius of the fillet of the leading edge curve and the pressure side curve, must be above 0, if 0 the curves will be returned as separate
        
    Returns:
        c ([ndarray]): the curves of the leading edge in a list

    """
    # Curve point number constant
    n = 15
    cfs = 100 - cfs
    cfp = 100 - cfp
    # Surface split point
    sp = gmt.cbs_interp(sides[0], cfs, 0)[0]
    pp = gmt.cbs_interp(sides[1], cfp, 0)[0]
    # Surfaces
    suc_curve = np.vstack((sp, gmt.crv_cut(gmt.crv_cut(sides[0], _division_line(sides, divx), '<'), cfs, '<')))
    pre_curve = np.vstack((gmt.crv_cut(gmt.crv_cut(sides[1], _division_line(sides, divx), '<'), cfp, '<'), pp))

    # Generate trailing edge curve
    if cgenfunc == 'marriage':
        te_curve = crvgen.marriage(sides[0], sides[1], cfs, cfp, cfp - cgenarg[0], n, cgenarg[1])

    elif cgenfunc == 'arc_p':
        crvp = gmt.cbs_interp(sides[0], cfp + cgenarg, 0)[0]
        te_curve = crvgen.arc_p([pre_curve[-1], suc_curve[0]], crvp, n)

    elif cgenfunc == 'arc_tang':
        tan_points = np.vstack((suc_curve[0:2], pre_curve[-1]))
        te_curve = np.flipud(crvgen.arc_tang(tan_points, n))
    
    elif cgenfunc == 'input':
        te_curve = gmt.crv_fit2p(cgenarg, pre_curve[-1], suc_curve[0], proxi_snap=True)

    # Fillet leading edge curve and presure side
    if r>0:
        return [[gmt.fillet_aprox(pre_curve, te_curve, r, [1, 1])[1], suc_curve]]
    elif r==0:
        if gmt.comcheck(pre_curve[-1], te_curve[-1], 0.01):
            te_curve = np.flipud(te_curve)
        return [[pre_curve, te_curve, suc_curve]]


# ELEMENT GENERATION
def slat(sides, css, csp, cgenfunc, cgenarg, r, mirror = False):
    """
    Generate a slat so the resulting geometry can be used in combination with a leading edge slot.

    Args:
        sides: the suction and pressure side as returned by Airfoil.sides()
        css (float): 0 < css < 50, the suction side flap chord
        csp (float): 0 < csp < 50, the pressure side flap chord
        cgenfunc (str): string corelating to the curve generation fucntion that will be used. can be one of: {'median', 'bezier', 'arc_p', 'arc_tang', 'input'}
        cgenarg: The arguments needed for the function (described at the top of the module)
        r (float): the radius of the fillet of the suction side curve and the pressure side curve, must be above 0, if 0 the curves will be returned as separate
        mirror (bool): if True, the generated curve will be mirrored
        
    Returns:
        c ([ndarray]): the curves of the leading edge in a list

    """
    # Curve point number constant
    n = 20
    # Surface split point
    sp = gmt.cbs_interp(sides[0], css, 0)[0]
    pp = gmt.cbs_interp(sides[1], csp, 0)[0]
    # Surfaces
    suc_curve = np.vstack((sp, gmt.crv_cut(sides[0], css, '<'), gmt.crv_cut(sides[1], csp, '<'), pp))
    # Generate curve
    if cgenfunc == 'median':
        theta = - gmt.vectorangle(suc_curve[0]-suc_curve[1]) - np.pi/2 + 0.015
        pre_curve = crvgen.median(suc_curve, theta, cgenarg, n)

    elif cgenfunc == 'bezier':
        pre_curve = crvgen.bezier(suc_curve, *cgenarg, n)

    elif cgenfunc == 'arc_p':
        crvp = gmt.cbs_interp(sides[0], css + cgenarg, 0)[0]
        pre_curve = crvgen.arc_p(suc_curve, crvp, n)

    elif cgenfunc == 'arc_tang':
        pre_curve = crvgen.arc_tang(suc_curve, n)
    
    elif cgenfunc == 'input':
        pre_curve = gmt.crv_fit2p(cgenarg, suc_curve[-1], suc_curve[0], proxi_snap=True)

    # Mirror
    if mirror:
        pre_curve = gmt.mirror(pre_curve, pre_curve[[0,-1]])

    # Fillet suction edge curve and pressure side
    if r>0:
        return [[gmt.fillet_aprox(suc_curve, pre_curve, r, [1, 1], [-1])[1]]]
    elif r==0:
        return [[suc_curve, np.flipud(pre_curve)]]


def flap(sides, cfs, cfp, cgenfunc, cgenarg, r):
    """
    Generate a flap, so the resulting geometry can be used in combination with a trailing edge slot.

    Args:
        sides: the suction and pressure side as returned by Airfoil.sides()
        cfs (float): 0 < cfs < 50, the suction side flap chord
        cfp (float): 0 < cfp < 50, the pressure side flap chord
        cgenfunc (str): string corelating to the curve generation fucntion that will be used. can be one of: {'median', 'bezier', 'arc_p', 'arc_tang', 'input'}
        cgenarg: The arguments needed for the function (described at the top of the module)
        r (float): the radius of the fillet of the leading edge curve and the pressure side curve, must be above 0, if 0 the curves will be returned as separate
        tmorphl (list): list containing the variables that are used in morphing the top side of the curve (described at the top of the module)
        
    Returns:
        c ([ndarray]): the curves of the leading edge in a list

    """
    # Curve point number constant
    n = 15
    cfs = 100 - cfs
    cfp = 100 - cfp
    # Surface split point
    sp = gmt.cbs_interp(sides[0], cfs, 0)[0]
    pp = gmt.cbs_interp(sides[1], cfp, 0)[0]
    # Surfaces
    suc_curve = np.vstack((gmt.crv_cut(sides[0], cfs, '>'), sp))
    pre_curve = np.vstack((pp, gmt.crv_cut(sides[1], cfp, '>')))

    if cgenfunc == 'marriage':
        le_curve = crvgen.marriage(sides[0], sides[1], cfs, cfp, cfp - cgenarg[0], n, cgenarg[1])

    elif cgenfunc == 'arc_p':
        crvp = gmt.cbs_interp(sides[0], cfp + cgenarg, 0)[0]
        le_curve = crvgen.arc_p([pre_curve[0], suc_curve[-1]], crvp, n)

    elif cgenfunc == 'arc_tang':
        tan_points = np.vstack((sp, gmt.crv_cut(sides[0], cfs, '<')[0], pre_curve[0]))
        le_curve = crvgen.arc_tang(tan_points, n)
    
    elif cgenfunc == 'input':
        le_curve = gmt.crv_fit2p(cgenarg, suc_curve[-1], pre_curve[0], proxi_snap=True)

    # Fillet leading edge curve and presure side
    if r>0:
        return [[gmt.patch(suc_curve, gmt.fillet_aprox(le_curve, pre_curve, r, [1, 1])[1], [-1])]]
    elif r==0:
        return [[gmt.patch(suc_curve, le_curve), pre_curve]]
