#--------- READ ME----------------------------------------------------------------
# A package of geometric functions using geomdl and numpy.
# This consists the base of all other scripts. While there is room for improvement,
# its best to avoid editing anything unless the editor has a deep understanding of
# all the functions here and on all other scripts. 
#---------------------------------------------------------------------------------

import numpy as np
import math
from geomdl import fitting

# A point is a vector of the x , y coordinates that define it
# A curve is a list of points 
# Classes are not built, to facilitate diversity and inclusion policies.

# PRIVATE STUFF
_quadict = {1: [1,1], 2: [-1,1], 3: [-1,-1], 4:[1,-1]}
_opdict = {'<': lambda x,y: x<y, '>': lambda x,y: x>y, '<=': lambda x,y: x<=y, '>=': lambda x,y: x>=y}


# FUNCTION SOLVING
def bisect_solver(funct, y, seg, tol):
    """
    Solve a function with the bisection method.

    Args:
        funct (function): function to be solved
        y (float): value at which the function is solved
        seg (list): [x1, x2] the starting segment
        tol (float): the maximum difference of x1 and x2 in the final segment,  tol > 0
    
    Returns:
        (float): estimated x root

    Raises:
        Error: Segment is fucked
        
    """
    y1, y2 = funct(seg[0]) - y, funct(seg[1]) - y 

    while abs(seg[0] - seg[1]) > tol:
        midp = np.mean(seg)
        y0 = funct(midp) - y
        if y1*y0 < 0:
            y2, seg[1] = y0, midp
        elif y2*y0 < 0:
            y1, seg[0] = y0, midp
        else:
            raise Exception('Error: Segment is fucked')

    return seg[0] - (seg[1]-seg[0]) * y1 / (y2-y1)


# BASIC MANIPULATION
def translate(p, tv):
    """
    Translate points by vector tv.

    Args:
        p (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        tv (array-like): [x, y] components of displacement vector

    Returns:
        (ndarray): Translated point coordinates

    """
    p = np.array(p)
    tv = np.array(tv)
    if p.ndim == 1:
        return p + tv
    elif p.ndim ==2:
        return p + np.repeat([tv], np.shape(p)[0], axis=0)


def rotate(p, center, theta):
    """
    Rotate points p around center by theta.

    Args:
        p (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        center (array-like): [x, y] coordinates of rotation center
        theta (float): the angle of rotation in radiants
    
    Returns:
        (ndarray): Rotated point coordinates

    """
    p = np.array(p)
    center = np.array(center)
    p = translate(p, -center)
    transform = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    p = np.transpose(transform @ np.transpose(p))
    return translate(p, center)


def scale(p, center, fv):
    """
    Scale points p around center accodring to vector fv.

    Args:
        p (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        center (array-like): [x, y] coordinates of rotation center
        fv (array-like): [xf, yf] factors by which each coordinate is scaled
    
    Returns:
        (ndarray): Scaled point coordinates

    """
    p = np.array(p)
    center = np.array(center)
    p = translate(p, -center)
    p[:,0] = p[:,0] * fv[0]
    p[:,1] = p[:,1] * fv[1]
    return translate(p, center)


def mirror(p, ax):
    """
    Mirror points p around axis ax.

    Args:
        p (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        ax (array-like): [[xa1, ya1], [xa2, ya2]] a matrix of two points that define the mirroring axis 
    
    Returns:
        (ndarray): Mirrored point coordinates
    
    """
    p = np.array(p)
    ax = np.array(ax)
    theta = vectorangle(ax[1]-ax[0])
    p = translate(p, -ax[0])
    p = rotate(p, [0,0], -theta)
    p[:,1] = - p[:,1]
    p = rotate(p, [0,0], theta)
    p = translate(p, ax[0])
    return p


def comcheck(p1, p2, tol):
    """
    Check if two points have common coordinates, with tolerance.

    Args:
        p1 (array-like): [x1, y1] coordinates of point 1
        p2 (array-like): [x2, y2] coordinates of point 2
        tol (float): tolerance
    
    Returns:
        (bool): True if points are common
    
    """
    return tol >= np.hypot(p1[0] - p2[0], p1[1] - p2[1])


# SPLINE
def spline_param_find(spline, val, axis, tol = 10**-3, delta = 10**-3):
    """
    Return the parametre of a spline, at certain coordinates, using bisection method.

    Args:
        nurbs (nurbs curve): nurbs curve object
        val (list of floats): contains the values of the coordinates
        axis (list of ints): contains the axes of the coordinates, 0 for x axis coordinates, 1 for y axis coordinates, must be as long as the val list
        tol (float): the maximum difference of the values of the final bisection segment. This refers to the parametres and is multiplied by the delat value.
        delta (float): the starting division step, to start evaluating, 0 < delta < 1, if the algorithm cant find a point you know exists, try decreasing this.
    
    Returns:
        (list): contains all the found parametres sorted.
        
    """
    if np.shape(val) == ():
        val = [val]
        axis = [axis]
    tol = tol * delta
    spline.delta =  1/(1/delta + 1)
    points = np.array(spline.evalpts)
    params = []
    for i in range(len(val)):
        ps = points[:, axis[i]] - val[i]
        # Find all segments
        sci = np.array(np.nonzero(ps[0:-1]*ps[1:] < 0))[0]

        # Find parametres
        u = []
        funct = lambda u: spline.evaluate_single(u)[axis[i]]
        for si in sci:
            seg = [si * delta, (si+1) * delta]    
            u.append(bisect_solver(funct, val[i], seg, tol))

        params = params + u
    
    params.sort()
    return params


def cbs_interp(c, val, axis, spfargs = [], centripetal = True):
    """
    Use centripetaly fitted cubic bspline to interpolate points at certain values.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        val (list of floats): contains the values of the coordinates
        axis (list of ints): contains the axes of the coordinates, 0 for x axis coordinates, 1 for y axis coordinates, must be as long as the val list
        centripetal (bool): if True, centripetal algorithm is used
        spfargs (list): contains the following optional arguments of spline_param_find function: [tol, delta]
    
    Returns:
        p (ndarray): points interpolated, sorted by their respective parametre on the curve

    """
    spline = fitting.interpolate_curve(list(c), 3, centripetal=centripetal)
    params = spline_param_find(spline, val, axis, *spfargs)
    return np.array(spline.evaluate_list(params))


# ANGLES
def vectorangle(v1, v2 = [1,0], reflex = False):
    """
    Calculate the angle of a vector and x axis, or between two vectors, from vector 2 to vector 1.

    Args:
        v1 (array-like): [x, y] coordinates of vector 1
        v2 (array-like): [x, y] coordinates of vector 2
        reflex (bool): If True, the opposite angle is given 
    
    Returns:
        (float): Angle in radiants

    """
    v1, v2 = np.array(v1), np.array(v2)
    if v1.ndim == 1:
        normprod = np.linalg.norm(v1) * np.linalg.norm(v2)
        costh = np.dot(v2, v1) / normprod
    else:
        normprod = np.hypot(v1[:,0], v1[:,1]) * np.hypot(v2[:,0], v2[:,1])
        costh = np.diagonal(v2 @ np.transpose(v1)) / normprod

    sinth = np.cross(v2, v1) / normprod
    theta = np.arctan2(sinth, costh)
    if reflex:
        theta = theta - np.sign(theta) * 2*np.pi
    return theta


def quadrant(v1, v2 = np.zeros(2), reflex = False):
    """
    Find the quadrant at which a vector, or the bisector of two vectors, point at.

    Args:
        v1 (array-like): [x, y] coordinates of vector 1
        v2 (array-like): [x, y] coordinates of vector 2
        reflex (bool): If true, the bisector of the reflex angle is taken instead

    Returns:
        (int): The number of the quadrant        

    """
    if reflex:
        s = -1
    else:
        s = 1
    v1 = np.array(v1)
    v2 = np.array(v2)
    v1 = v1 / np.linalg.norm(v1)
    if not np.all(v2 == np.zeros(2)):
        v2 = v2 / np.linalg.norm(v2)
    v = (v1 + v2) * s
    if v[0] >= 0 and v[1] > 0:
        return 1
    elif v[0] < 0 and v[1] >= 0:
        return 2
    elif v[0] <= 0 and v[1] < 0:
        return 3
    elif v[0] > 0 and v[1] <= 0:
        return 4


def bisector_lnr(lf1, lf2):
    """
    Find the line bisectors of two lines.

    Args:
        lf1 (array-like): Line 1 factors as given by np.polyfit()
        lf2 (array-like): Line 2 factors as given by np.polyfit()
    
    Returns:
        list containing:
        -lfb1 (ndarray): Bisector line 1 factors as given by np.polyfit()
        -lfb2 (ndarray): Bisector line 2 factors as given by np.polyfit()
    
    """
    sqr1 = (lf1[0]**2 + 1)**0.5
    sqr2 = (lf2[0]**2 + 1)**0.5
    lfb1 = np.zeros(2)
    lfb1[0] = (lf1[0] * sqr2 + lf2[0] * sqr1) / (sqr2 + sqr1)
    lfb1[1] = (lf1[1] * sqr2 + lf2[1] * sqr1) / (sqr2 + sqr1)
    lfb2 = np.zeros(2)
    lfb2[0] = (lf1[0] * sqr2 - lf2[0] * sqr1) / (sqr2 - sqr1)
    lfb2[1] = (lf1[1] * sqr2 - lf2[1] * sqr1) / (sqr2 - sqr1)
    return [lfb1, lfb2]


def vertical_lnr(lf, p):
    """
    Find the line vertical to the first, passing through point.

    Args:
        lf (array-like): Line factors as given by np.polyfit()
        p (array-like): [x, y] coordinates of point
    
    Returns:
        (ndarray): the vertical line factors as given by np.polyfit()
    
    """
    lfv = np.zeros(2)
    lfv[0] = -1/lf[0]
    lfv[1] = p[1] - lfv[0]*p[0]
    return lfv


def bisector_vct(v1, v2):
    """
    Find the unit vector that bisects two vectors.

    Args:
        v1 (array-like): [x, y] coordinates of vector 1
        v2 (array-like): [x, y] coordinates of vector 2
    
    Returns:
        (ndarray): [x, y] coordinates of bisectorvector

    """
    v1, v2 = np.array(v1), np.array(v2)
    if v1.ndim == 2:
        v1norm = np.hypot(v1[:,0], v1[:,1])
        v1norm = np.transpose(np.vstack((v1norm, v1norm)))
        v2norm = np.hypot(v2[:,0], v2[:,1])
        v2norm = np.transpose(np.vstack((v2norm, v2norm)))
        v = v1/v1norm + v2/v2norm
        vnorm = np.hypot(v[:,0], v[:,1])
        vnorm = np.transpose(np.vstack((vnorm, vnorm)))
    elif v1.ndim == 1:
        v1norm = np.hypot(v1[0], v1[1])
        v2norm = np.hypot(v2[0], v2[1])
        v = v1/v1norm + v2/v2norm
        vnorm = np.hypot(v[0], v[1])
    return v/vnorm


def vertical_vct(v, side = False):
    """
    Find a vector vertical to the given.

    Args:
        v (array-like): [x, y] coordinates of vector
        side (bool): If True the right side is picked, esle, the left
    
    Returns:
        (ndarray): [x, y] coordinates of vertical vector

    """
    if side:
        s = 1
    else:
        s = -1
    a = [[v[0], v[1]], [-v[1], v[0]]]
    b = [0, s]
    vv = np.linalg.solve(a,b)
    return vv/np.linalg.norm(vv)


def project(p, lf):
    """
    Project a point onto a line.

    Args:
        p (array-like): [x, y] coordinates of point
        lf (array-like): Line factors as given by np.polyfit()

    Returns:
        (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the projected point coordinates

    """
    lfv = np.zeros((len(p), 2))
    lfv[0] = -1/lf[0]
    lfv[1] = p[1] - lfv[0]*p[0]
    return lnr_inters(lf, lfv)


# INTERSECTION
def lnr_inters(lf1, lf2):
    """
    Find the intersection of two lines.

    Args:
        lf1 (array-like): Line 1 factors as given by np.polyfit()
        lf2 (array-like): Line 2 factors as given by np.polyfit()
    
    Returns:
        (ndarray): [x, y] coordinates of the intersection
    
    """
    x0 = (lf2[1]-lf1[1])/(lf1[0]-lf2[0])
    y0 = (lf1[0]*lf2[1] - lf2[0]*lf1[1])/(lf1[0]-lf2[0])
    return np.array([x0, y0])


def crv_inters(c1, c2):
    """
    Find the first intersection of two curves.

    Args:
        c1 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 1
        c2 (array-like): [[x0, y0], [x1, y1], ... , [xm, ym]] the matrix containing all the point coordinates of curve 2
    
    Returns:
        list containing:
        -b (bool): bool showing whether the function succeded in finding an intersection or not
        -p (ndarray): [x, y] coordinates of the intersection, None if failed to find intersection
        -i (int): index of coordinates of last point before the intersection, of the curve 1 matrix, None if failed to find intersection
        -j (int): index of coordinates of last point before the intersection, of the curve 2 matrix, None if failed to find intersection
    
    """
    c1, c2 = np.array(c1), np.array(c2)
    failed_intersect = True
    for i in range(0, np.shape(c1)[0]-1):
        for j in range(0, np.shape(c2)[0]-1):
            # Find intersection
            lf1 = np.polyfit(c1[i:i+2,0], c1[i:i+2,1], 1)
            lf2 = np.polyfit(c2[j:j+2,0], c2[j:j+2,1], 1)
            p0 = lnr_inters(lf1, lf2)

            # Check intesection
            pint1bool = min([c1[i,0], c1[i+1,0]]) <  p0[0] < max([c1[i,0], c1[i+1,0]]) 
            pint2bool = min([c2[j,0], c2[j+1,0]]) < p0[0] < max([c2[j,0], c2[j+1,0]])

            if pint1bool and pint2bool:
                failed_intersect = False
                break
        
        if not failed_intersect:
            break
    
    if not failed_intersect:
        return [True, p0, i, j]
    else:
        return [False, None, None, None]


def crvself_inters(c):
    """
    Find the first intersection of a 

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
    
    Returns:
        list containing:
        -b (bool): bool showing whether the function succeded in finding an intersection or not
        -p (ndarray): [x, y] coordinates of the intersection, None if failed to find intersection
        -i (int): index of coordinates of last point before the first segment of intersection, None if failed to find intersection
        -j (int): index of coordinates of last point before the second segment of intersection, None if failed to find intersection
    
    """
    c = np.array(c)
    p, i, j = None, None, None
    for ic in range(np.shape(c)[0]-1):
        b, p, i, j = crv_inters(c[ic:ic+2], c[ic+2:])
        if b:
            i = ic
            j = j + ic + 1
            break

    return [b,p,i,j]


def crv_inters_insert(c1, c2):
    """
    Get the first point of intersections of two curves and insert it to the curves.

    Args:
        c1 (array-like): [[x0, y0], [x1, y1], ... , [xn1, yn1]] the matrix containing all the point coordinates of curve 1
        c2 (array-like): [[x0, y0], [x1, y1], ... , [xn2, yn2]] the matrix containing all the point coordinates of curve 2
    
    Returns:
        list containing:
        -c1 (array-like): [[x0, y0], [x1, y1], ... , [xm1, ym1]] the matrix containing all the point coordinates of curve 1
        -c2 (array-like): [[x0, y0], [x1, y1], ... , [xm2, ym2]] the matrix containing all the point coordinates of curve 2

    """
    b, p, i, j = crv_inters(c1, c2)
    if b:
        c1, c2 = np.insert(c1, p, i+1), np.insert(c2, p, j+1) 
    return [[c1, i], [c2, j]]


# CIRCLES
def crcl_2pr(p1, p2, r, side = True):
    """
    Find a circle passing through two points, with a given radius.

    Args:
        p1 (array-like): [x, y] coordinates of point 1
        p2 (array-like): [x, y] coordinates of point 2
        r (float): radius of the circle
        side (bool): If true the center at the right side of the vector (p2 -p1) is returned. Else, the left.
    
    Returns:
        (ndarray): [x, y] coordinates of the center of the circle
    
    """
    p1, p2 = np.array(p1), np.array(p2)
    b = (p2[1]**2 - p1[1]**2 + p2[0]**2 - p1[0]**2) / (p2[0] - p1[0])
    a = (p1[1] - p2[1]) / (p2[0] - p1[0])
    a2 = 1 + a**2
    a1 = b*a - 2*p1[1] - 2*p1[0]*a
    a0 = (b/2)**2 - p1[0]*b + p1[0]**2 + p1[1]**2 - r**2
    yc = np.roots([a2, a1, a0])
    xc = b/2 + a*yc
    centers = np.transpose([xc, yc])
    v1 = centers[0] - p1
    v2 = p2 - p1

    if side:
        s = 1
    else:
        s = -1

    if s * np.cross(v1, v2) > 0:
        return centers[0]
    else:
        return centers[1]


def crcl_3p(p1, p2, p3):
    """
    Find the circle passing through three points.

    Args:
        p1 (array-like): [x, y] coordinates of point 1
        p2 (array-like): [x, y] coordinates of point 2
        p3 (array-like): [x, y] coordinates of point 3
    
    Returns:
        (ndarray): [x, y] coordinates of the center of the circle
    
    """
    x1, x2, x3 = p1[0], p2[0], p3[0]
    y1, y2, y3 = p1[1], p2[1], p3[1]
    mx = np.linalg.det([[x1**2 + y1**2, y1, 1], [x2**2 + y2**2, y2, 1], [x3**2 + y3**2, y3, 1]])
    my = np.linalg.det([[x1**2 + y1**2, x1, 1], [x2**2 + y2**2, x2, 1], [x3**2 + y3**2, x3, 1]])
    mxy = np.linalg.det([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
    x0 = mx / (mxy * 2)
    y0 = - my / (mxy * 2)
    return np.array([x0, y0])


def crcl_fit(p):
    """
    Fit a circle onto points. Points must be 4 or more.

    Args:
        p (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
    
    Returns:
        list containing:
        - p0 (ndarray): the coordinates of the circle center
        - r (float): the circle radius
    
    """
    p = np.array(p)
    n = np.shape(p)[0]
    x, y = np.sum(p[:,0]), np.sum(p[:,1])
    x2, y2 = np.sum(p[:,0]**2), np.sum(p[:,1]**2)
    xy = np.sum(p[:,0] * p[:,1])
    a = [[2*x2, 2*xy, x], [2*xy, 2*y2, y], [2*x, 2*y, n]]
    b = [np.sum(p[:,0]**3 + p[:,0] * p[:,1]**2), np.sum(p[:,1]**3 + p[:,1] * p[:,0]**2), x2 + y2]
    sv = np.linalg.solve(a, b)
    c = (1/n) * (x2 + y2 - 2 * np.dot(sv, [x, y, 0]))
    r = (sv[0]**2 + sv[1]**2 + c)**0.5
    p0 = np.array([sv[0], sv[1]])
    return [p0, r]


def crcl_tang_2lnr(lf1, lf2, r, quadrnt):
    """
    Find circle tangent to two lines with radius r.

    Args:
        lf1 (array-like): Line 1 factors as given by np.polyfit()
        lf2 (array-like): Line 2 factors as given by np.polyfit()
        r (float): Radius of the circle
        quadrnt (int): The quadrant that the center is located, in relation to the intersection of lines 1 and 2
    
    Returns:
        list containing:
        -p0 (ndarray): [x, y] coordinates of the center of the circle
        -ptan1 (ndarray): [x, y] coordinates of point of tangency on line 1
        -ptan2 (ndarray): [x, y] coordinates of point of tangency on line 2
    
    """
    # Find centers
    lf11 = [lf1[0], lf1[1] + (lf1[0]**2 + 1)**0.5 * r]
    lf12 = [lf1[0], lf1[1] - (lf1[0]**2 + 1)**0.5 * r]
    lf21 = [lf2[0], lf2[1] + (lf2[0]**2 + 1)**0.5 * r]
    lf22 = [lf2[0], lf2[1] - (lf2[0]**2 + 1)**0.5 * r]
    p0 = np.array([lnr_inters(lf11, lf21), lnr_inters(lf11, lf22), lnr_inters(lf12, lf21), lnr_inters(lf12, lf22)])

    # Select center
    xp, yp = lnr_inters(lf1, lf2)
    fx, fy = _quadict[quadrnt]
    i = np.argwhere(np.logical_and(fx * (p0[:,0] - xp) >= 0, fy * (p0[:,1] - yp) >= 0))[0,0]
    p0 = p0[i]

    # Find tangent points
    lfc = vertical_lnr(lf1, p0)
    ptan1 = lnr_inters(lfc, lf1)
    lfc = vertical_lnr(lf2, p0)
    ptan2 = lnr_inters(lfc, lf2)

    return [p0, ptan1, ptan2]


def crcl_tang_segrlim(seg1, seg2):
    """
    Find the minimum and maximum radius of a circle tangent to two line segments, in the quadrant pointed at bu their bisecting vector.

    Args:
        seg1 (array-like): [[x0, y0], [x1, y1]] the point coordinates of segment 1
        seg2 (array-like): [[x0, y0], [x1, y1]] the point coordinates of segment 2
    
    Returns:
        list containing:
        -minimum radius of the circle
        -maximum radius of the circle
        -empty list if no tangent circle exists
    
    """
    seg1, seg2 = np.array(seg1), np.array(seg2)
    lf1 = np.polyfit(seg1[:,0], seg1[:,1], 1)
    lf2 =  np.polyfit(seg2[:,0], seg2[:,1], 1)
    p0 = lnr_inters(lf1, lf2)
    insidebool1 = min(seg1[:,0]) <= p0[0] <= max(seg1[:,0])      # Is intersection point inside segment?
    insidebool2 = min(seg2[:,0]) <= p0[0] <= max(seg2[:,0])      # Is intersection point inside segment?
    pointawaybool1 = np.linalg.norm(p0-seg1[0]) < np.linalg.norm(p0-seg1[1])  # Is segment vector pointing away from intersection point?
    pointawaybool2 = np.linalg.norm(p0-seg2[0]) < np.linalg.norm(p0-seg2[1])  # Is segment vector pointing away from intersection point?

    if ((not insidebool1) and (not pointawaybool1)) or ((not insidebool2) and (not pointawaybool2)):
        return []

    bisectvct = bisector_vct(seg1[1]-seg1[0], seg2[1]-seg2[0])
    theta = vectorangle(bisectvct)
    blf = np.polyfit([p0[0], bisectvct[0] + p0[0]], [p0[1], bisectvct[1] + p0[1]], 1)
    p11 = lnr_inters(vertical_lnr(lf1, seg1[0]), blf)
    p12 = lnr_inters(vertical_lnr(lf1, seg1[1]), blf)
    p21 = lnr_inters(vertical_lnr(lf2, seg2[0]), blf)
    p22 = lnr_inters(vertical_lnr(lf2, seg2[1]), blf)
    segpoints = np.array([p11,p12,p21,p22])
    intersi = list(np.argsort(rotate(segpoints, [0,0], -theta)[:,0]))[1:-1]
    if intersi == [3,0] or intersi == [1,2]:       # No intersecting bisector segments
        return []
    interseg = segpoints[intersi]
    clp = np.vstack((seg1, seg2))[intersi]
    r = np.sort([np.linalg.norm(clp[0]-interseg[0]), np.linalg.norm(clp[1]-interseg[1])])
    if insidebool1 and insidebool2:                            # If segments intersect minimum radius is set to 0
        r[0] = 0
    
    return r


def crcl_tang_2lnln(lf1, lf2, lf0, quadrnt):
    """
    Find center tangent to two lines, with center located on a third line.

    Args:
        lf1 (array-like): Line 1 factors as given by np.polyfit()
        lf2 (array-like): Line 2 factors as given by np.polyfit()
        lf0 (array-like): Line 0 factors as given by np.polyfit()
        quadrnt (int): The quadrant that the center is located, in relation to the intersection of lines 1 and 2
    
    Returns:
        list containing:
        -p0 (ndarray): [x, y] coordinates of the center of the circle
        -ptan1 (ndarray): [x, y] coordinates of point of tangency on line 1
        -ptan2 (ndarray): [x, y] coordinates of point of tangency on line 2
    
    Raises:
        Error: No centers found in specified quadrant
    
    """
    # Find bisector lines
    lfb1, lfb2 = bisector_lnr(lf1, lf2)
    p0 = np.array([lnr_inters(lfb1, lf0), lnr_inters(lfb2, lf0)])

    # Select center
    xp, yp = lnr_inters(lf1, lf2)
    fx, fy = _quadict[quadrnt]

    logic_mat = np.logical_and(fx * (p0[:,0] - xp) >= 0, fy * (p0[:,1] - yp) >= 0)

    if not np.any(logic_mat):
        raise Exception('Error: No centers found in specified quadrant')
    
    i = np.argwhere(logic_mat)[0,0]
    p0 = p0[i]

    # Find tangent points
    lfc = vertical_lnr(lf1, p0)
    ptan1 = lnr_inters(lfc, lf1)
    lfc = vertical_lnr(lf2, p0)
    ptan2 = lnr_inters(lfc, lf2)
    
    return [p0, ptan1, ptan2]


def crcl_tang_lnpp(lf, p1, p2):
    """
    Find the circle tangent on a line on point 1 and passing through point 2.

    Args:
        lf (array-like): Line factors as given by np.polyfit()
        p1 (array-like): [x, y] coordinates of point 1, the coordinates must satisfy the line equation
        p2 (array-like): [x, y] coordinates of point 2
    
    Returns:
        (ndarray): [x, y] coordinates of the center of the circle

    """
    # Find intersecting lines
    lf1 = np.zeros(2)
    lf1[0] = -1/lf[0]
    lf1[1] = p1[1] - lf1[0] * p1[0]
    lf2 = np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], 1)
    lf2[0] = -1/lf2[0]
    lf2[1] = (p1[1] + p2[1]) / 2 - lf2[0] * (p1[0] + p2[0]) / 2
    # Return center
    return lnr_inters(lf1, lf2)


def crcl_tang_2crv(c1, c2, arg):
    """
    Find first circle tangent on two curves that satisfies the given argument.

    Args:
        c1 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 1
        c2 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2
        arg (float or array-like): Either the line factors as given by np.polyfit(), defining the line at which the center is located, or the radius
    
    Returns:
        list containing:
        -b (bool): bool showing whether the function succeded in finding an intersection or not
        -p0 (ndarray): [x, y] coordinates of the center of the circle, None if failed to find circle
        -ptan1 (ndarray): [x, y] coordinates of point of tangency on line 1, None if failed to find circle
        -ptan2 (ndarray): [x, y] coordinates of point of tangency on line 2, None if failed to find circle
        -i (int): index of coordinates of last point before the intersection, of the curve 1 matrix, None if failed to find circle
        -j (int): index of coordinates of last point before the intersection, of the curve 2 matrix, None if failed to find intersection
    
    Raises:
        FormError: Arguments dont fit needed format if arg is none of the following: list, ndarray, int, float

    """
    c1, c2 = np.array(c1), np.array(c2)
    # Select function depending on arg
    if type(arg) == np.ndarray or type(arg) == list:
        funct = crcl_tang_2lnln
    else:
        funct = crcl_tang_2lnr

    # Find tangent circle
    failed_tangent = True
    for i in range(0, np.shape(c1)[0]-1):
        for j in range(0, np.shape(c2)[0]-1):

            lf1 = np.polyfit(c1[i:i+2,0], c1[i:i+2,1], 1)
            lf2 = np.polyfit(c2[j:j+2,0], c2[j:j+2,1], 1)
            quadrnt = quadrant(c1[i+1] - c1[i], c2[j+1] - c2[j])

            p0, ptan1, ptan2 = funct(lf1, lf2, arg, quadrnt)

            # Check if points are valid
            ptan1bool = np.sort([c1[i,0], ptan1[0], c1[i+1,0]])[1] == ptan1[0]
            ptan2bool = np.sort([c2[j,0], ptan2[0], c2[j+1,0]])[1] == ptan2[0]

            if ptan1bool and ptan2bool:
                failed_tangent = False
                break
        
        if not failed_tangent:
            break

    if not failed_tangent:
        return [True, p0, ptan1, ptan2, i, j]
    else:
        return [False, None, None, None, None, None]


def arc_gen(p1, p2, p0, n, reflex = False):
    """
    Generate points of an arc, from point 1 to point 2 with point 0 as center. Generated points include point 1 and 2.

    Args:
        p1 (array-like): [x, y] coordinates of point 1
        p2 (array-like): [x, y] coordinates of point 2
        p0 (array-like): [x, y] coordinates of the center
        n (int): Number of points to be generated
        reflex (bool): If False, the smaller of the two possible arcs will be generated, else the greater
    
    Returns:
        (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the arc

    """
    p1, p2, p0 = np.array(p1), np.array(p2), np.array(p0)
    r = np.linalg.norm(p1-p0)
    theta = vectorangle(p2-p0, p1-p0, reflex)
    theta1 = vectorangle(p1-p0)
    thetan = np.linspace(theta1, theta1 + theta, n + 1)[1:-1]
    x = r * np.cos(thetan) + p0[0]
    y = r * np.sin(thetan) + p0[1]
    return np.vstack((p1, np.transpose([x, y]), p2))


# CURVE BUILDING
def bezier(p, w = 1):
    """
    Return the function of a rational bezier curve with control points p and weights w.

    Args:
        p (array-like): [[x0, y0], [x1, y1], ... , [xm, ym]] the matrix containing all the point coordinates
        w (array-like): [w0, w1, ..., wn] the vector containing all the weight values, should have same length as p
    
    Returns:
        (function): Function that generates points based on the u parametre (0 <= u <= 1)

    """
    p = np.array(p)
    n = np.shape(p)[0]-1
    i = np.arange(n+1)
    nfact = math.factorial(n)
    binoms = np.array(list(map(lambda i: nfact / (math.factorial(i) * math.factorial(n-i)), i)))

    def bez(u):
        evalpoints = []
        for uval in u:
            wbernstein = np.array(w) * binoms * (1-uval)**(n-i) * uval**i
            x = np.sum(wbernstein * p[:,0]) / np.sum(wbernstein)
            y = np.sum(wbernstein * p[:,1]) / np.sum(wbernstein)
            evalpoints.append([x,y])
        return np.array(evalpoints)
    
    return bez


def parallcrv(c):
    """
    Find a curve parallel to the first, at unit distance.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
    
    Returns:
        (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]], adding this matrix to the argument curve will give the displaced curve

    """
    c = np.array(c)
    v1 = c[1:-1]-c[0:-2]
    v2 = c[2:]-c[1:-1]
    s = - np.sign(np.cross(v1, v2))
    s = np.transpose(np.vstack((s,s)))
    cdisp = s * bisector_vct(v2, -v1)
    v1 = vertical_vct(c[1]-c[0])
    v2 = vertical_vct(c[-1]-c[-2])
    cdisp = np.vstack((v1, cdisp, v2))
    return cdisp


# SINGLE CURVE METRICS
def crv_len(c):
    """
    Calculate the length of the curve at every point of it.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
    
    Returns:
        (ndarray): [l0, l1, ..., ln] the vector containing the lengths for every point

    """
    c = np.array(c)
    norms = np.hypot(c[1:,0] - c[0:-1,0], c[1:,1] - c[0:-1,1])
    sh = np.shape(norms)[0]
    clen = np.tril(np.ones((sh,sh))) @ norms
    clen = np.insert(clen, 0, 0)
    return clen


def crv_ang(c):
    """
    Calculate the additive curvature angle of the curve at every point of it.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
    
    Returns:
        (ndarray): [l0, l1, ..., ln-1] the vector containing the angle for every segment

    """
    c = np.array(c)
    thetas = vectorangle(c[2:] - c[1:-1], c[1:-1] - c[0:-2])
    sh = np.shape(thetas)[0]
    cang = np.tril(np.ones((sh,sh))) @ thetas
    cang = np.hstack(([0], cang))
    return cang


def crv_curvature(c):
    """
    Calculate the curvature of the curve.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... ,[xn, yn]] the matrix containing the points of the curve

    Returns:
        (ndarray): vector containing the curvature value for every segment of the curve

    """
    c = np.array(c)
    cang = vectorangle(c[2:] - c[1:-1], c[1:-1] - c[0:-2])
    cang = np.append(np.insert(cang, 0, -cang[0]), -cang[-1])
    slen = np.hypot(c[1:,0] - c[0:-1,0], c[1:,1] - c[0:-1,1])
    return (cang[0:-1] + cang[1:]) / (2 * slen)


# SINGLE CURVE MORPHOLOGY
def smooth_zigzag(c, minang, varp = 0.5):
    """
    Smoothen a curve by removing zig-zagging segments.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
        minang (float): minimum angle in radians, of a zig-zag
        varp (float): must be possitive, the greater the variation power (varp) the less similar two consecutive angles need to be to be considered a zig-zag
    
    Returns:
        (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the smoothened curve

    """
    c = np.array(c, dtype=float)
    sc = c
    zigzag = True
    while zigzag:
        # Minimum angle limitation check
        thetas = vectorangle(sc[2:] - sc[1:-1], sc[1:-1] - sc[0:-2])
        cons = thetas[0:-1] * thetas[1:]
        lim = - minang**2
        mincheck = cons < lim
        # Variation limitation check
        maxthetas = np.max(np.abs([thetas[0:-1], thetas[1:]]), axis=0)
        minthetas = np.min(np.abs([thetas[0:-1], thetas[1:]]), axis=0)
        vf = minthetas**varp
        varlim = np.max([minthetas*vf, minthetas/vf], axis=0)
        varcheck = maxthetas < varlim
        # Smoothen
        i = np.array(np.nonzero(np.logical_and(varcheck, mincheck)))[0]
        zigzag = np.shape(i)[0] > 0
        sc[i+1] = (sc[i+1]+sc[i+2])/2
        sc = np.delete(sc, i+2, 0)
    return sc


def smooth_loops(c):
    """
    Clear looping sections of a curve.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... ,[xn, yn]] the matrix containing the points of the curve
    
    Returns:
        (ndarray): [[x0, y0], [x1, y1], ... ,[xm, ym]] the matrix containing the points of the clear curve

    """
    c = np.array(c)
    cs = c
    b = True
    while b:
        b, p, i, j = crvself_inters(cs)
        if b:
            cs = np.delete(cs, range(i+1,j+2), axis=0)
            cs = np.insert(cs, i+1, p, axis=0)
    return cs


def smooth_fillet_crcl(c, minang, c_factor = 0.8):
    """
    Smoothen a curve's sharp angles by filleting them away with circular arcs.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
        minang (float): any angle equal or greater than this will be filleted
        c_factor (float): a magic number between 0 and 1. The bigger the number the bigger the radius of the fillet, above 0.5 may lead to funky results for consecutive filleted angles

    Returns:
        (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing the points of the smooth curve

    """
    c = np.array(c)
    csmooth = c
    cang = np.abs(vectorangle(c[2:] - c[1:-1], c[1:-1] - c[0:-2]))
    shpts = np.array(np.nonzero(cang >= minang))[0] + 1
    extri = 0
    for i in shpts:
        # Get points of tangency
        l1 = np.linalg.norm(c[i] - c[i-1])
        l2 = np.linalg.norm(c[i] - c[i+1])
        minli = i + np.argmin([l1, l2])*2 - 1

        # Check if consecutive angles, and if True, reduce r factor (if needed) to avoid funky shit when filleting
        if (minli in shpts) and (c_factor > 0.499):
            r_factor = 0.499
        else:
            r_factor = c_factor

        fillen = r_factor * np.linalg.norm(c[i] - c[minli])
        ptan1 = c[i] + fillen * (c[i-1] - c[i]) / l1
        ptan2 = c[i] + fillen * (c[i+1] - c[i]) / l2
        # Get center of tangent circle
        lf1 = np.polyfit([c[i,0], ptan1[0]], [c[i,1], ptan1[1]],1)
        lf2 = np.polyfit([c[i,0], ptan2[0]], [c[i,1], ptan2[1]],1)
        lf1 = vertical_lnr(lf1, ptan1)
        lf2 = vertical_lnr(lf2, ptan2)
        p0 = lnr_inters(lf1, lf2)
        # Generate arc
        arcp = arc_gen(ptan1, ptan2, p0, int(np.ceil(2*cang[i-1]/minang)))
        # Remove and add points to curve
        csmooth = np.delete(csmooth, i+extri, axis=0)
        csmooth = np.insert(csmooth, i+extri, arcp, axis=0)
        extri = extri + len(arcp) - 1
    
    return csmooth


def smooth_fillet_bezier(c, minang, b_factor = 0.8):
    """
    Smoothen a curve's sharp angles by filleting them away with bezier.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
        minang (float): any angle equal or greater than this will be filleted
        b_factor (float): a magic number between 0 and 1. The bigger the number the bigger the "radius" of the fillet

    Returns:
        (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing the points of the smooth curve

    """
    c = np.array(c)
    csmooth = c
    cang = np.abs(vectorangle(c[2:] - c[1:-1], c[1:-1] - c[0:-2]))
    shpts = np.array(np.nonzero(cang >= minang))[0] + 1
    extri = 0
    for i in shpts:
        # Check if consecutive angles, and if True, reduce r factor (if needed) to avoid funky shit when filleting
        if (i-1 in shpts) and (b_factor >= 0.499):
            r_factor1 = 0.499
        else:
            r_factor1 = b_factor
        if (i+1 in shpts) and (b_factor >= 0.499):
            r_factor2 = 0.499
        else:
            r_factor2 = b_factor

        # Get points of tangency
        ptan1 = c[i] + r_factor1 * (c[i-1] - c[i])
        ptan2 = c[i] + r_factor2 * (c[i+1] - c[i])
        # Generate curve
        t = np.linspace(0, 1, int(np.ceil(2*cang[i-1]/minang))+2)
        bzrp = bezier([ptan1, c[i], ptan2])(t)
        
        # Remove and add points to curve
        csmooth = np.delete(csmooth, i+extri, axis=0)
        csmooth = np.insert(csmooth, i+extri, bzrp, axis=0)
        extri = extri + len(bzrp) - 1
    
    return csmooth


def smooth_spline(c, centripetal = True, smoothfact = 0.7):
    """
    Smoothen a curve by aproximating it with a spline.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... ,[xn, yn]] the matrix containing the points of the curve
        centripetal (bool): if True, centripetal algorithm is used
        smoothfact (float): smoothness factor

    Returns:
        c (ndarray): [[x0, y0], [x1, y1], ... ,[xn, yn]] the matrix containing the points of the smooth curve
    
    """
    ctrspts_size = int(np.ceil((1-smoothfact) * len(c)))
    spline = fitting.approximate_curve(list(c), 3, centripetal=centripetal, ctrlpts_size=5)
    spline.delta = 1/len(c)
    return np.array(spline.evalpts)


def clr_duplicates(c, tol = 10**-4):
    """
    Clear "duplicate" points of a curve, within a certain tolerance.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... ,[xn, yn]] the matrix containing the points
        tol (float): The tolerance when checking if two points are common.
    
    Returns:
        (ndarray): the array without the duplicate values    
    
    """
    c = np.array(c)
    loopbool = True
    while loopbool:
        loopbool = False
        cc = c[0]
        i = 1
        carrlen = np.shape(c)[0] - 2

        while i < carrlen:
            if comcheck(c[i], c[i+1], tol):
                cc = np.vstack((cc, (c[i] + c[i+1])/2))
                loopbool = True
                i += 2
            else:
                cc = np.vstack((cc, c[i]))
                i += 1

        if i == carrlen:
            cc = np.vstack((cc, c[-2]))

        cc = np.vstack((cc, c[-1]))
        c = cc
    
    return cc


def crv_cleanup(c, zigzagang, filletang, fmethod, varp = 0.3, ffactor = 0.8, tol = 10**-4):
    """
    Clean a curve from zigzagging patterns, loops, duplicate points and sharp angles.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
        zigzagang (float): the minang argument of the smooth_zigzag function
        filletang (float): the minang argument of the smooth_fillet function
        fmethod (str): determines the method of the fillet, can be either 'circle' for circular arc filleting or 'bezier' for bezier curve filleting
        varp (float): the varp argument of the smooth_zigzag function
        r_factor (float): the r_factor argument of the smooth_fillet function
        tol (float): the tolerance of clr_duplicates function

    Returns:
        (ndarray): [[x0, y0], [x1, y1], ... , [xm, ym]] the coords of the clean curve

    """
    if fmethod == 'circle':
        fillet_func = smooth_fillet_crcl
    elif fmethod == 'bezier':
        fillet_func = smooth_fillet_bezier
    else:
        raise Exception('Invalid filleting style, read the doc string with your eyes.')

    c = smooth_loops(c)
    c = clr_duplicates(c, tol)
    c = smooth_zigzag(c, zigzagang, varp)
    c = clr_duplicates(c, tol)
    c = fillet_func(c, filletang, ffactor)
    return clr_duplicates(c, tol)


def crv_div(c, div):
    """
    Divide a curve into segments.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... ,[xn, yn]] the matrix containing the points of the curve
        div (array-like): The list of the fractions of the curve length at which the curve is divided (0< div <1)
    
    Returns:
        list containing:
        -cd (ndarray): [[x0, y0], [x1, y1], ... ,[x(m), y(m)]] the matrix containing the points of the curve, including points of division
        -j (ndarray): vector containing the indexes of the division points in cd

    """
    c = np.array(c)
    clen = crv_len(c)
    divlen = np.array(div) * clen[-1]
    clenmat = np.repeat(clen[:,np.newaxis], np.shape(divlen)[0], axis=1)
    divlenmat = np.repeat(divlen[np.newaxis,:], np.shape(clen)[0], axis=0)
    i = np.sum(clenmat < divlenmat, axis=0)
    rv = divlen - clen[i-1]
    vl = np.hypot(c[i,0] - c[i-1,0], c[i,1] - c[i-1,1])
    unvec = (c[i] - c[i-1]) / np.transpose([vl,vl])
    p = unvec * np.transpose([rv, rv]) + c[i-1]
    cd = np.insert(c, i, p, axis=0)
    j = i + np.arange(len(i))
    return [cd, j]


def crv_cut(c, arg, op, return_index = False):
    """
    Return the curve part that satisfy the inequality: c (op) arg.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the curve
        arg (float or array-like): Either the vector with the factors of a line as given by np.polyfit(), or an x value
        op (str): Can be one of the four inequality operators '<', '>', '<=', '>='
        return_index (bool): if True, return the indexes instead of the curve

    Returns:
        (ndarray): Curve consisting of the parts satisfying the op according to arg

    Raises:
        FormError: Arguments dont fit needed format if arg is none of the following: list, ndarray, int, float

    """
    if (type(arg) == list) or (type(arg) == np.ndarray):
        y = np.polyval(arg, c[:,0])
        i = np.nonzero(_opdict[op](c[:,1], y))[0]
    else:
        i = np.nonzero(_opdict[op](c[:,0], arg))[0]

    if return_index:
        return i
    else:
        return c[i]


def crv_fit2p(c, p1, p2, i1 = 0, i2 = -1, proxi_snap = False):
    """
    Transforms points of a curve, so the indexed points fall onto given coordinates.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates, must be ascending
        p1 (array-like): [x, y] coordinates of point 1
        p2 (array-like): [x, y] coordinates of point 2
        i1 (int): index of point of curve that will be moved onto point 1
        i2 (int): index of point of curve that will be moved onto point 2
        proxi_snap (bool): if True, i1 and i2 are ignored and instead the points of the curve closest to the points 1 and 2 will be selected accordingly.
    
    Returns:
        (ndarray): Transformed point coordinates

    """
    c = np.array(c)
    p1 = np.array(p1)
    p2 = np.array(p2)

    if proxi_snap:
        dists = crv_dist([p1,p2], c)
        indxs = np.argmin(dists, axis=1)
        if indxs[0] == indxs[1]:
            print('Warning: crv_fit2p proxi_snap failed, defaulting to the use of index arguments.')
        else:
            i1, i2 = indxs[0], indxs[1]

    c = translate(c, p1-c[i1])
    theta = vectorangle(c[i2] - c[i1], p2 - p1)
    c = rotate(c, p1, -theta)
    sf = np.linalg.norm(p2 - p1) / np.linalg.norm(c[i2] - c[i1])
    c = scale(c, p1, [sf, sf])
    return c


# CURVE INTERACTIONS
def crv_dist(c1, c2):
    """
    Measure distance between points of two curves.

    Args:
        c1 (array-like): [[x0, y0], [x1, y1], ... , [xm, ym]] the matrix containing all the point coordinates of curve 1
        c2 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2

    Returns:
        (ndarray): M x N matrix containing distances of all points, M and N being the number of points of curve 1 and 2

    """
    c1, c2 = np.array(c1), np.array(c2)
    m, n = np.shape(c1)[0], np.shape(c2)[0]
    c1 = np.repeat(c1[:,np.newaxis,:], n, axis=1)
    c2 = np.repeat(c2[np.newaxis,:,:], m, axis=0)
    c1x, c1y = c1[:,:,0], c1[:,:,1]
    c2x, c2y = c2[:,:,0], c2[:,:,1]
    return np.hypot(c1x-c2x, c1y-c2y)


def crv_p_dist(c, p):
    """
    Accurately measure distance of a point from curve.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
        p (array-liek): [x, y] coordinates of the point
    
    Returns:
        list containing:
        -d (float): distance

    """

    num = np.abs((c[1:,1] - c[0:-1,1]) * p[0] - (c[1:,0] - c[0:-1,0]) * p[1] + c[1:,0] * c[0:-1,1] - c[1:,1] * c[0:-1,0])
    den = ((c[1:,1] - c[0:-1,1])**2 + (c[1:,0] - c[0:-1,0])**2)**0.5
    lnrdist = np.min(num/den)
    vrtxdist = np.min(np.hypot(c[:,0] - p[0], c[:,1] - p[1]))
    return min(lnrdist, vrtxdist)


def crv_common(c1, c2, tol = 10**-4):
    """
    Check if two curves have common points within tolerance.

    Args:
        c1 (array-like): [[x0, y0], [x1, y1], ... , [xm, ym]] the matrix containing all the point coordinates of curve 1
        c2 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2
        tol (float): tolerance

    Returns:
        (ndarray): M x N matrix containing True if points are within tolerance, M and N being the number of points of curve 1 and 2

    """
    m, n = np.shape(c1)[0], np.shape(c2)[0]
    tolmat = np.ones((m, n)) * tol
    return crv_dist(c1, c2) <= tolmat


# def clr_common(c1, c2, tol = 10**-4):
#     """
#     Check if two curves have common points within tolerance. If c1 == c2, clear duplicate points of the curve.

#     Args:
#         c1 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 1
#         c2 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2
#         tol (float): tolerance

#     Returns:
#         if c1 != c2, list containing:
#         -c1 (ndarray): curve 1 cleared of common points
#         -c2 (ndarray): curve 2 cleared of common points
#         elif c1 == c2:
#         c (ndarray): curve cleared of duplicate points

#     """
#     boolmat = crv_common(c1, c2, tol)

#     if c1 == c2:
#         diag = np.arange(0, np.shape(c1)[0])
#         boolmat[diag, diag] = False

#     comnalty = [np.sum(boolmat, axis = 0), np.sum(boolmat, axis = 1)]  # commonality being how many points the particular point is considered common with
    
#     # Convoluted point clearing algorithm
#     clr_indxs = [[],[]]
#     while np.any(comnalty[0] > 0):
#         i = np.argmax(max(comnalty[0]), max(comnalty[1]))              # Get the curve that has the point with highest commonality
#         j = np.argmax(comnalty[i])                                     # Get the index of the point most common within the curve 
#         clr_indxs[i].append(j)                                         # Add index to the list of indexes, indexing points to be removed
#         comnalty[i][j] = 0                                             # Zero the value of "commonality" on that curve
#         if i == 0:
#             bslc = np.s_[j,:]
#         else:
#             bslc = np.s_[:,j]
            
#         comindxs = np.nonzero(boolmat[bslc])
#         comnalty[i-1][comindxs] = comnalty[i-1][comindxs] - 1          # Remove 1 commonality from every point (of the other curve) common with the particular, to simulate it removed
#         boolmat[bslc] = False                                          # Mask point in the bool matrix too

#     c1u = np.delete(c1, clr_indxs[0])
#     c2u = np.delete(c2, clr_indxs[1])
#     return [c1u, c2u]


def clr_common(c1, c2, tol = 10**-4):
    """
    Clears the points of c1 that are common with c2.

    Args:
        c1 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 1
        c2 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2
        tol (float): tolerance

    Returns:
        (ndarray): curve 1 cleared of common points

    """
    boolmat = crv_common(c1, c2, tol)
    if c1 == c2:
        diag = np.arange(0, np.shape(c1)[0])
        boolmat[diag, diag] = False
    inds = np.nonzero(np.sum(boolmat), axis = 0)[0]
    return np.delete(c1, inds)


def common_ends(c1, c2, end1 = None, end2 = None, tol = 10**-4):
    """
    Find two end points of two curves that are common.

    Args:
        c1 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        c2 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        end1 (int): The end of the first list that will be patched, in case of multiple common points. 0 assigns the beggining of the list and -1 the end
        end2 (int): The end of the second list that will be patched, in case of multiple common points. 0 assigns the beggining of the list and -1 the end
        tol (float): The tolerance when checking if two points are common.
    
    Returns:
        (int): Int showing which ends meet
        - 0: both curves meet at their begginings
        - 1: both curves meet at their ends 
        - 2: first curve at the beggining, second at the end 
        - 3: first curve at the end, second at the beggining
    
    Raises:
        CommonalityError: no common points within tolerance , if no common points are found within tolerance, specified or otherwise
        SpecificityError: need to specify ends , if too many points are found common within tolerance, and not enough specifications
    
    """
    c1, c2 = np.array(c1), np.array(c2)
    indxlist = [0, -1]
    counter = 0

    # Find common ends
    boolistc = [False, False, False, False]
    for i in range(2):
        p1 = c1[indxlist[i]]
        for j in range(2):
            p2 = c2[indxlist[j]]
            if comcheck(p1, p2, tol):
                boolistc[counter] = True
            counter += 1

    # Restrict common ends
    boolist1 = [True, True, True, True]
    boolist2 = [True, True, True, True]
    if end1 != None:
        boolist1[3 + 2*end1] = False
        boolist1[2 + 2*end1] = False
    if end2 != None:
        boolist2[1 + end2] = False
        boolist2[3 + end2] = False

    boolist = np.logical_and(np.logical_and(boolist1, boolist2), boolistc)
    ti = np.count_nonzero(boolist)
    if ti == 0:
        raise Exception('CommonalityError: no common points within tolerance')
    elif ti > 1:
        raise Exception('SpecificityError: need to specify ends')
    
    return np.nonzero(boolist)[0][0]


def patch(c1, c2, comend_args = []):
    """
    Patch two curves into one.

    Args:
        c1 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        c2 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        comend_args (list): list containing the arguments for the common ends function [end1, end2, tol]
    
    Returns:
        (ndarray): Curve consisting of two curves
    
    Raises:
        CommonalityError: no common points within tolerance , if no common points are found within tolerance, specified or otherwise
        SpecificityError: need to specify ends , if too many points are found common within tolerance, and not enough specifications
    
    """
    ti = common_ends(c1, c2, *comend_args)

    if ti == 0:
        c1 = np.flipud(c1)
    elif ti == 1:
        c1 = np.flipud(c1)
        c2 = np.flipud(c2)
    elif ti == 3:
        c2 = np.flipud(c2)
    
    c2 = c2[1:]
    return np.vstack((c1,c2))


def fillet(c1, c2, r, n, comend_args = [], reflex = False, crosscheck = False):
    """
    ### DEPRECATED ###
    Fillet two patchable curves into one. If it fails to find a fillet circle, a simple patched curve is returned.

    Args:
        c1 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 1
        c2 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2
        r (float): The radius of the fillet arc
        n (int or array-like): Either the number of points, or a vector with the angular and longitudinal density [ad, ld], ad being the number of points in a full circle, ld being the number of points per unit of length
        comend_args (list): list containing the arguments for the common ends function [end1, end2, tol]
        reflex (bool): If True, the generated arc will be "indented" rather than "smooth"
        crosscheck (bool): If True, checks if the curves cross each other at any point. If they do, the last intersection is assigned as the new start point
    
    Returns:
        list containing:
        -b (bool): True if the function succeded in building an arc
        -c (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]] the patched/filleted curve
        -crcl_data (list): [p0, ptan1, ptan2, i, j] List containing the data of the fillet circle as shown bellow, None if failed to find circle
            >p0 (ndarray): [x, y] coordinates of the center of the fillet circle
            >ptan1 (ndarray): [x, y] coordinates of point of tangency on curve 1
            >ptan2 (ndarray): [x, y] coordinates of point of tangency on curve 2
            >i (int): index of coordinates of last point before the intersection, of the curve 1 matrix
            >j (int): index of coordinates of last point before the intersection, of the curve 2 matrix

    """
    c1, c2 = np.array(c1), np.array(c2)

    # Correcting curve indexing
    ti = common_ends(c1, c2, *comend_args)

    if ti == 1:
        c2 = np.flipud(c2)
    elif ti == 2:
        c1 = np.flipud(c1)
    elif ti == 3:
        c1 = np.flipud(c1)
        c2 = np.flipud(c2)

    # Check for intersections
    if crosscheck:
        c1 = np.flipud(c1)
        c2 = np.flipud(c2)
        b, p, i, j = crv_inters(c1[1:-1], c2[1:-1])

        if b:
            c1 = np.vstack((c1[0:i+2], p))
            c2 = np.vstack((c2[0:j+2], p))

        c1 = np.flipud(c1)
        c2 = np.flipud(c2)

    # Long explanation of some strange choises up ahead: 
    # Checking both quadrants fucks up computational efficiency. Thus the quadrant
    # for research will be picked according to the following product: sang1 * sign(angc).
    #
    # sang1 is the sign of the angle of the first vectors/segments. We use this sign as these segments have a common start point.
    # angc is the angle of the currently researched segments.
    # When the curves have an diverging trend, the product of sang1 and sign(angc) will be positive. Thus the quadrant of the small angle bisector is picked.
    # When the curves have a converging trend, the product of sang1 and sign(angc) will be negative. Thus the quadrant of the great angle bisector is picked.
    #
    # Obviously the curves must not cross each other.

    sang1 = np.sign(vectorangle(c1[1]-c1[0], c2[1]-c2[0]))

    # Find tangent circle
    failed_tangent = True
    for i in range(0, np.shape(c1)[0]-1):
        for j in range(0, np.shape(c2)[0]-1):

            lf1 = np.polyfit(c1[i:i+2,0], c1[i:i+2,1], 1)
            lf2 = np.polyfit(c2[j:j+2,0], c2[j:j+2,1], 1)
            rs = sang1 * np.sign(vectorangle(c1[i+1] - c1[i], c2[j+1] - c2[j]))
            quadrnt = quadrant(c1[i+1] - c1[i], c2[j+1] - c2[j], reflex=rs<0)

            p0, ptan1, ptan2 = crcl_tang_2lnr(lf1, lf2, r, quadrnt)

            # Check if points are valid
            ptan1bool = np.sort([c1[i,0], ptan1[0], c1[i+1,0]])[1] == ptan1[0]
            ptan2bool = np.sort([c2[j,0], ptan2[0], c2[j+1,0]])[1] == ptan2[0]

            if ptan1bool and ptan2bool:
                failed_tangent = False
                break
        
        if not failed_tangent:
            break
    
    if failed_tangent:      # Unsuccesful fillet
        b = False
        c = patch(c1, c2, [0, 0])
        crcl_data = None
        return [b, c, crcl_data]
    
    else:                   # Succesful fillet
        b = True

        # Delete points before tangency
        c1 = c1[i+1:]
        c2 = c2[j+1:]

        # Fillet arc
        # Indent or smooth
        if reflex:
            rs = -rs
        
        # Number of points
        if type(n) == np.ndarray or type(n) == list:
            ad, ld = n
            na = abs(ad * vectorangle(ptan1-p0, ptan2-p0, rs<0) / 2*np.pi)
            nl = abs(ld * vectorangle(ptan1-p0, ptan2-p0, rs<0) * np.linalg.norm(ptan1-p0))
            n = int(nl + na)
        
        # Generate arc
        ca = arc_gen(ptan1, ptan2, p0, n, rs<0)

        # Assemble curve
        c1 = np.flipud(c1)
        c = np.vstack((c1, ca, c2))

        crcl_data = [p0, ptan1, ptan2, i, j]

        return [b, c, crcl_data]


def fillet_aprox(c1, c2, r, n, comend_args = [], reflex = False, crosscheck = False):
    """
    Fillet two patchable curves into one. If it fails to find a fillet circle with requested radius an aproximate radius is used. 
    Note: This fillet finds only circles tangent to diverging portions of curves, this works well for our purposes. To generalise it, one must modify the crcl_tang_2ln_rlim
    function to take a quadrant argument. But that allows the function to sometimes find unholy circles that will end up making airfoils into tear drops. Thats bad.

    Args:
        c1 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 1
        c2 (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2
        r (float): the requested radius of the fillet arc
        n (int or array-like): either the number of points, or a vector with the angular and longitudinal density [ad, ld], ad being the number of points in a full circle, ld being the number of points per unit of length
        comend_args (list): list containing the arguments for the common ends function [end1, end2, tol]
        reflex (bool): if True, the generated arc will be "indented" rather than "smooth"
        crosscheck (bool): if True, checks if the curves cross each other at any point. If they do, the last intersection is assigned as the new start point
    
    Returns:
        list containing:
        -r (float): The radius that the fillet was done with
        -c (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]] the filleted curve
        -crcl_data (list): [p0, ptan1, ptan2, i, j] List containing the data of the fillet circle as shown bellow, None if failed to find circle
            >p0 (ndarray): [x, y] coordinates of the center of the fillet circle
            >ptan1 (ndarray): [x, y] coordinates of point of tangency on curve 1
            >ptan2 (ndarray): [x, y] coordinates of point of tangency on curve 2
            >i (int): index of coordinates of last point before the intersection, of the curve 1 matrix
            >j (int): index of coordinates of last point before the intersection, of the curve 2 matrix

    """
    c1, c2 = np.array(c1), np.array(c2)

    # Correcting curve indexing
    ti = common_ends(c1, c2, *comend_args)

    if ti == 1:
        c2 = np.flipud(c2)
    elif ti == 2:
        c1 = np.flipud(c1)
    elif ti == 3:
        c1 = np.flipud(c1)
        c2 = np.flipud(c2)

    # Check for intersections
    if crosscheck:
        c1 = np.flipud(c1)
        c2 = np.flipud(c2)
        b, p, i, j = crv_inters(c1[1:-1], c2[1:-1])

        if b:
            c1 = np.vstack((c1[0:i+2], p))
            c2 = np.vstack((c2[0:j+2], p))

        c1 = np.flipud(c1)
        c2 = np.flipud(c2)

    # Obviously the curves must not cross each other. A couple of checks:
    # To reduce hopeless looping which eats up resources, if finding a tangent circle of any radius fails too many consecutive times on the same segment (failim times) the function skips the
    # segment and continues on the next one. 
    ifailim = 5
    jfailim = int(np.shape(c2)[0] / 5)

    # Find tangent circle
    failed_tangent = True
    ifails = 0
    rpool, ipool, jpool = [], [], []
    for i in range(0, np.shape(c1)[0]-1):
        jfails = 0
        jsuccess = False
        for j in range(0, np.shape(c2)[0]-1):

            rlims = crcl_tang_segrlim(c1[i:i+2], c2[j:j+2])

            if rlims == []:
                if jsuccess:
                    jfails += 1
                    if jfails > jfailim:
                        break
                continue
            else:
                jsuccess = True
                jfails = 0
            
            # Check if a segment combination that satisfies r perfectly is found
            if rlims[0] < r < rlims[1]:
                failed_tangent = False
                break
            else:
                # Select the best suited r and add it to the pool
                rbest = rlims[np.argmin(np.abs(rlims - r))]
                rpool.append(rbest)
                ipool.append(i)
                jpool.append(j)
        
        if jsuccess == False:
            ifails += 1
        else:
            ifails = 0
        
        if (not failed_tangent) or (ifails > ifailim):
            break


    if failed_tangent:      # If perfect fillet fails, get aproximate fillet
        rstr = str(r)
        selecti = np.argmin(np.abs(np.array(rpool) - r))
        r = rpool[selecti]
        i = ipool[selecti]
        j = jpool[selecti]
        print('Warning: fillet_aprox couldnt do requested fillet of radius ' + rstr + ', doing a fillet of radius ' + str(r) + ' instead.')
    
    # Get tangent circle
    quadrnt = quadrant(c1[i+1] - c1[i], c2[j+1] - c2[j])
    lf1 = np.polyfit(c1[i:i+2,0], c1[i:i+2,1], 1)
    lf2 = np.polyfit(c2[j:j+2,0], c2[j:j+2,1], 1)
    p0, ptan1, ptan2 = crcl_tang_2lnr(lf1, lf2, r, quadrnt)

    # Delete points before tangency
    c1 = c1[i+1:]
    c2 = c2[j+1:]

    # Fillet arc
    # Number of points
    if type(n) == np.ndarray or type(n) == list:
        ad, ld = n
        na = abs(ad * vectorangle(ptan1-p0, ptan2-p0, reflex) / 2*np.pi)
        nl = abs(ld * vectorangle(ptan1-p0, ptan2-p0, reflex) * np.linalg.norm(ptan1-p0))
        n = int(nl + na)
    
    # Generate arc
    ca = arc_gen(ptan1, ptan2, p0, n, reflex)

    # Assemble curve
    c1 = np.flipud(c1)
    c = np.vstack((c1, ca, c2))

    crcl_data = [p0, ptan1, ptan2, i, j]

    return [r, c, crcl_data]
