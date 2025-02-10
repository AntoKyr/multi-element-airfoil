#--------- READ ME----------------------------------------------------------------
# A package of geometric functions using geomdl and numpy.
# This consists the base of all other scripts. While there is room for improvement,
# its best to avoid editing anything unless the editor has a deep understanding of
# all the functions here and on all other scripts. 
#---------------------------------------------------------------------------------

import numpy as np
import math
from geomdl import fitting
from geomdl.BSpline import Curve
from matplotlib import pyplot as plt
from matplotlib.path import Path
from typing import Callable, Union

# A point is a vector of the x , y coordinates that define it
# A curve is a list of points 
# Classes are not built, to facilitate diversity and inclusion policies.

# PRIVATE STUFF
_quadict = {1: [1,1], 2: [-1,1], 3: [-1,-1], 4:[1,-1]}
_opdict = {'<': lambda x,y: x<y, '>': lambda x,y: x>y, '<=': lambda x,y: x<=y, '>=': lambda x,y: x>=y}


# GENERAL
def bisect_solver(funct: Callable[[float], float], y: float, seg: Union[list,tuple,np.ndarray], tol: float) -> float:
    """
    Solve a function with the bisection method.

    Args:
        funct: function to be solved
        y: value at which the function is solved
        seg: [x1, x2] the starting segment
        tol: the maximum difference of x1 and x2 in the final segment,  tol > 0
    
    Returns:
        Estimated x root

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


def nxt_i(x: int) -> int:
    """ Get 1 for indx 0 and -2 for indx -1. """
    if x == 0:
        return 1
    else:
        return x - 1


def opst_i(x: int) -> int:
    """ Get -1 for indx 0 and 0 for indx -1. """
    if x == 0:
        return -1
    else:
        return 0


def lfp(p1, p2):
    """
    Get the polynomial factors of the line passign through two points.

    Args:
        p1: [x0, y0] the matrix containing the point 1 coordinates
        p2: [x0, y0] the matrix containing the point 2 coordinates

    Returns:
        [a1, a0] the line poly factors

    """
    x1, y1 = p1
    x2, y2 = p2
    return np.array([(y2 - y1) / (x2 - x1), (y1 * x2 - y2 * x1) / (x2 - x1)])


# BASIC MANIPULATION
def translate(p: Union[list,tuple,np.ndarray], tv: Union[list,tuple,np.ndarray]) -> np.ndarray:
    """
    Translate points by vector tv.

    Args:
        p: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        tv: [x, y] components of displacement vector

    Returns:
        Translated point coordinates

    """
    p = np.array(p)
    tv = np.array(tv)
    if p.ndim == 1:
        return p + tv
    elif p.ndim ==2:
        return p + np.repeat([tv], np.shape(p)[0], axis=0)


def rotate(p: Union[list,tuple,np.ndarray], center: Union[list,tuple,np.ndarray], theta: float) -> np.ndarray:
    """
    Rotate points p around center by theta.

    Args:
        p: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        center: [x, y] coordinates of rotation center
        theta: the angle of rotation in radiants
    
    Returns:
        Rotated point coordinates

    """
    p = np.array(p)
    center = np.array(center)
    p = translate(p, -center)
    transform = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    p = np.transpose(transform @ np.transpose(p))
    return translate(p, center)


def scale(p: Union[list,tuple,np.ndarray], center: Union[list,tuple,np.ndarray], fv: Union[list,tuple,np.ndarray]) -> np.ndarray:
    """
    Scale points p around center accodring to vector fv.

    Args:
        p: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        center: [x, y] coordinates of rotation center
        fv: [xf, yf] factors by which each coordinate is scaled
    
    Returns:
        Scaled point coordinates

    """
    p = np.array(p)
    center = np.array(center)
    p = translate(p, -center)
    p[:,0] = p[:,0] * fv[0]
    p[:,1] = p[:,1] * fv[1]
    return translate(p, center)


def mirror(p: Union[list,tuple,np.ndarray], ax: Union[list,tuple,np.ndarray]) -> np.ndarray:
    """
    Mirror points p around axis ax.

    Args:
        p: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        ax: [[xa1, ya1], [xa2, ya2]] a matrix of two points that define the mirroring axis 
    
    Returns:
        Mirrored point coordinates
    
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


def comcheck(p1: Union[list,tuple,np.ndarray], p2: Union[list,tuple,np.ndarray], tol: float) -> bool:
    """
    Check if two points have common coordinates, with tolerance.

    Args:
        p1: [x1, y1] coordinates of point 1
        p2: [x2, y2] coordinates of point 2
        tol: tolerance
    
    Returns:
        True if points are common
    
    """
    return tol >= np.hypot(p1[0] - p2[0], p1[1] - p2[1])


# SPLINE
def spline_cord_param(spline: Curve, val: Union[float,list,tuple,np.ndarray], axis: Union[int,list,tuple,np.ndarray], tol: float = 10**-3, delta: float = 10**-3) -> list:
    """
    Return the parametre of a spline, at certain coordinates, using bisection method.

    Args:
        nurbs: nurbs curve object
        val: contains the values of the coordinates
        axis: contains the axes of the coordinates, 0 for x axis coordinates, 1 for y axis coordinates, must be as long as the val list
        tol: the maximum difference of the values of the final bisection segment. This refers to the parametres and is multiplied by the delat value.
        delta: the starting division step, to start evaluating, 0 < delta < 1, if the algorithm cant find a point you know exists, try decreasing this.
    
    Returns:
        Contains all the found parametres sorted.
        
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


def spline_len_param(spline: Curve, val: Union[float,list,tuple,np.ndarray], tol: float = 10**-3, delta: float = 10**-3) -> list:
    """"""


def cbs_interp(c: Union[list,tuple,np.ndarray], val: Union[float,list,tuple,np.ndarray], axis: Union[int,list,tuple,np.ndarray], spfargs: Union[list,tuple,np.ndarray] = [], centripetal: bool = True) -> np.ndarray:
    """
    Use a fitted cubic bspline to interpolate points at certain values.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        val: contains the values of the coordinates
        axis: contains the axes of the coordinates, 0 for x axis coordinates, 1 for y axis coordinates, must be as long as the val list
        spfargs: contains the following optional arguments of spline_param_find function: [tol, delta]
        centripetal: if True, centripetal algorithm is used
    
    Returns:
        Points interpolated, sorted by their respective parametre on the curve

    """
    spline = fitting.interpolate_curve(list(c), 3, centripetal=centripetal)
    params = spline_cord_param(spline, val, axis, *spfargs)
    return np.array(spline.evaluate_list(params))


# ANGLES
def vectorangle(v1: Union[list,tuple,np.ndarray], v2: Union[list,tuple,np.ndarray] = [1,0], reflex: bool = False) -> float:
    """
    Calculate the angle of a vector and x axis, or between two vectors, from vector 2 to vector 1.

    Args:
        v1: [x, y] coordinates of vector 1
        v2: [x, y] coordinates of vector 2
        reflex: If True, the opposite angle is given 
    
    Returns:
        Angle in radiants

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


def quadrant(v1: Union[list,tuple,np.ndarray], v2: Union[list,tuple,np.ndarray] = np.zeros(2), reflex: bool = False) -> int:
    """
    Find the quadrant at which a vector, or the bisector of two vectors, point at.

    Args:
        v1: [x, y] coordinates of vector 1
        v2: [x, y] coordinates of vector 2
        reflex: If true, the bisector of the reflex angle is taken instead

    Returns:
        The number of the quadrant        

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


def bisector_lnr(lf1: Union[list,tuple,np.ndarray], lf2: Union[list,tuple,np.ndarray]) -> list:
    """
    Find the line bisectors of two lines.

    Args:
        lf1: Line 1 factors as given by np.polyfit()
        lf2: Line 2 factors as given by np.polyfit()
    
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


def vertical_lnr(lf: Union[list,tuple,np.ndarray], p: Union[list,tuple,np.ndarray]) -> np.ndarray:
    """
    Find the line vertical to the first, passing through point.

    Args:
        lf: Line factors as given by np.polyfit()
        p: [x, y] coordinates of point
    
    Returns:
        The vertical line factors as given by np.polyfit()
    
    """
    lfv = np.zeros(2)
    lfv[0] = -1/lf[0]
    lfv[1] = p[1] - lfv[0]*p[0]
    return list(lfv)


def bisector_vct(v1: Union[list,tuple,np.ndarray], v2: Union[list,tuple,np.ndarray]) -> np.ndarray:
    """
    Find the unit vector that bisects two vectors.

    Args:
        v1: [x, y] coordinates of vector 1
        v2: [x, y] coordinates of vector 2
    
    Returns:
        [x, y] coordinates of bisectorvector

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


def vertical_vct(v: Union[list,tuple,np.ndarray], side: bool = False) -> np.ndarray:
    """
    Find a vector vertical to the given.

    Args:
        v: [x, y] coordinates of vector
        side: If True the right side is picked, esle, the left
    
    Returns:
        [x, y] coordinates of vertical vector

    """
    if side:
        s = 1
    else:
        s = -1
    a = [[v[0], v[1]], [-v[1], v[0]]]
    b = [0, s]
    vv = np.linalg.solve(a,b)
    return vv/np.linalg.norm(vv)


def project(p: Union[list,tuple,np.ndarray], lf: Union[list,tuple,np.ndarray]) -> np.ndarray:
    """
    Project a point onto a line.

    Args:
        p: [x, y] coordinates of point
        lf: Line factors as given by np.polyfit()

    Returns:
        [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the projected point coordinates

    """
    lfv = np.zeros(2)
    lfv[0] = -1/lf[0]
    lfv[1] = p[1] - lfv[0]*p[0]
    return lnr_inters(lf, lfv)


# INTERSECTION
def lnr_inters(lf1: Union[list,tuple,np.ndarray], lf2: Union[list,tuple,np.ndarray]) -> np.ndarray:
    """
    Find the intersection of two lines.

    Args:
        lf1: Line 1 factors as given by np.polyfit()
        lf2: Line 2 factors as given by np.polyfit()
    
    Returns:
        [x, y] coordinates of the intersection
    
    """
    x0 = (lf2[1]-lf1[1])/(lf1[0]-lf2[0])
    y0 = (lf1[0]*lf2[1] - lf2[0]*lf1[1])/(lf1[0]-lf2[0])
    return np.array([x0, y0])


def crv_inters(c1: Union[list,tuple,np.ndarray], c2: Union[list,tuple,np.ndarray]) -> list:
    """
    Find the first intersection of two curves.

    Args:
        c1: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 1
        c2: [[x0, y0], [x1, y1], ... , [xm, ym]] the matrix containing all the point coordinates of curve 2
    
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
            lf1 = lfp(c1[i], c1[i+1])
            lf2 = lfp(c2[j], c2[j+1])
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


def crvself_inters(c: Union[list,tuple,np.ndarray]) -> list:
    """
    Find the first intersection of a 

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
    
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


def raytrace(c: Union[list,tuple,np.ndarray], ray: Union[list,tuple,np.ndarray]) -> list:
    """
    Trace a ray on a curve and get all the intersections.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
        ray: [[x0,y0], [x1,y1]] ray vector

    Returns:
        list containing array of indexes before the intersections on the curve, and an array of the coresponding points

    """
    c, ray = np.array(c), np.array(ray)
    # Get curve segments that the ray line passes through
    rf = lfp(ray[0], ray[1])
    yr = np.polyval(rf, c[:,0])
    i = c[:,1] > yr
    i = np.nonzero(i[0:-1] != i[1:])[0]
    if len(i) > 0:
        # Get solutions segments
        an = (c[i+1,1] - c[i,1]) / (c[i+1,0] - c[i,0])
        bn = (c[i,1] * c[i+1,0] - c[i+1,1] * c[i,0]) / (c[i+1,0] - c[i,0])
        x = (bn - rf[1]) / (rf[0] - an)
        # Get ray line traces (intersections with curve)
        y = np.polyval(rf, x)
        points = np.transpose([x,y])
        # Sort them by closest to ray cast point to furthest
        sortindx = np.argsort(points[:,0])
        xray = ray[:,0]
        if ray[0,0] > ray[1,0]:
            sortindx = np.flip(sortindx)
            xray = np.flipud(xray)
        points = points[sortindx]
        # Keep only traces that are within the ray segment 
        valindx = np.nonzero(np.logical_and(points[:,0] > xray[0], points[:,0] < xray[1]))[0]
        i = i[sortindx][valindx]
        points = points[valindx]
        return [i, points]
    else:
        return [[],[]]


# CIRCLES
def crcl_2pr(p1: Union[list,tuple,np.ndarray], p2: Union[list,tuple,np.ndarray], r: float, side:bool = True) -> np.ndarray:
    """
    Find a circle passing through two points, with a given radius.

    Args:
        p1: [x, y] coordinates of point 1
        p2: [x, y] coordinates of point 2
        r: radius of the circle
        side: If true the center at the right side of the vector (p2 -p1) is returned. Else, the left.
    
    Returns:
        [x, y] coordinates of the center of the circle
    
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


def crcl_3p(p1: Union[list,tuple,np.ndarray], p2: Union[list,tuple,np.ndarray], p3: Union[list,tuple,np.ndarray]) -> np.ndarray:
    """
    Find the circle passing through three points.

    Args:
        p1: [x, y] coordinates of point 1
        p2: [x, y] coordinates of point 2
        p3: [x, y] coordinates of point 3
    
    Returns:
        [x, y] coordinates of the center of the circle
    
    """
    x1, x2, x3 = p1[0], p2[0], p3[0]
    y1, y2, y3 = p1[1], p2[1], p3[1]
    mx = np.linalg.det([[x1**2 + y1**2, y1, 1], [x2**2 + y2**2, y2, 1], [x3**2 + y3**2, y3, 1]])
    my = np.linalg.det([[x1**2 + y1**2, x1, 1], [x2**2 + y2**2, x2, 1], [x3**2 + y3**2, x3, 1]])
    mxy = np.linalg.det([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
    x0 = mx / (mxy * 2)
    y0 = - my / (mxy * 2)
    return np.array([x0, y0])


def crcl_fit(p: Union[list,tuple,np.ndarray]) -> list:
    """
    Fit a circle onto points. Points must be 4 or more.

    Args:
        p: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
    
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


def crcl_tang_2lnr(lf1: Union[list,tuple,np.ndarray], lf2: Union[list,tuple,np.ndarray], r: float, quadrnt: int) -> list:
    """
    Find circle tangent to two lines with radius r.

    Args:
        lf1: Line 1 factors as given by np.polyfit()
        lf2: Line 2 factors as given by np.polyfit()
        r: Radius of the circle
        quadrnt: The quadrant that the center is located, in relation to the intersection of lines 1 and 2
    
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


def crcl_tang_segrlim(seg1: Union[list,tuple,np.ndarray], seg2: Union[list,tuple,np.ndarray]) -> float:
    """
    Find the minimum and maximum radius of a circle tangent to two line segments, in the quadrant pointed at bu their bisecting vector.

    Args:
        seg1: [[x0, y0], [x1, y1]] the point coordinates of segment 1
        seg2: [[x0, y0], [x1, y1]] the point coordinates of segment 2
    
    Returns:
        list containing:
        -minimum radius of the circle
        -maximum radius of the circle
        -empty list if no tangent circle exists
    
    """
    seg1, seg2 = np.array(seg1), np.array(seg2)
    lf1 = lfp(seg1[0], seg1[1])
    lf2 =  lfp(seg2[0], seg2[1])
    p0 = lnr_inters(lf1, lf2)
    insidebool1 = min(seg1[:,0]) <= p0[0] <= max(seg1[:,0])      # Is intersection point inside segment?
    insidebool2 = min(seg2[:,0]) <= p0[0] <= max(seg2[:,0])      # Is intersection point inside segment?
    pointawaybool1 = np.linalg.norm(p0-seg1[0]) < np.linalg.norm(p0-seg1[1])  # Is segment vector pointing away from intersection point?
    pointawaybool2 = np.linalg.norm(p0-seg2[0]) < np.linalg.norm(p0-seg2[1])  # Is segment vector pointing away from intersection point?

    if ((not insidebool1) and (not pointawaybool1)) or ((not insidebool2) and (not pointawaybool2)):
        return []

    bisectvct = bisector_vct(seg1[1]-seg1[0], seg2[1]-seg2[0])
    theta = vectorangle(bisectvct)
    blf = lfp(p0, bisectvct + p0)
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


def crcl_tang_2lnln(lf1: Union[list,tuple,np.ndarray], lf2: Union[list,tuple,np.ndarray], lf0: Union[list,tuple,np.ndarray], quadrnt: int) -> list:
    """
    Find center tangent to two lines, with center located on a third line.

    Args:
        lf1: Line 1 factors as given by np.polyfit()
        lf2: Line 2 factors as given by np.polyfit()
        lf0: Line 0 factors as given by np.polyfit()
        quadrnt: The quadrant that the center is located, in relation to the intersection of lines 1 and 2
    
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


def crcl_tang_lnpp(lf: Union[list,tuple,np.ndarray], p1: Union[list,tuple,np.ndarray], p2: Union[list,tuple,np.ndarray]) -> np.ndarray:
    """
    Find the circle tangent on a line on point 1 and passing through point 2.

    Args:
        lf: Line factors as given by np.polyfit()
        p1: [x, y] coordinates of point 1, the coordinates must satisfy the line equation
        p2: [x, y] coordinates of point 2
    
    Returns:
        [x, y] coordinates of the center of the circle

    """
    # Find intersecting lines
    lf1 = np.zeros(2)
    lf1[0] = -1/lf[0]
    lf1[1] = p1[1] - lf1[0] * p1[0]
    lf2 = lfp(p1, p2)
    lf2[0] = -1/lf2[0]
    lf2[1] = (p1[1] + p2[1]) / 2 - lf2[0] * (p1[0] + p2[0]) / 2
    # Return center
    return lnr_inters(lf1, lf2)


def crcl_tang_2crv(c1: Union[list,tuple,np.ndarray], c2: Union[list,tuple,np.ndarray], arg: Union[float,list,tuple,np.ndarray]) -> list:
    """
    Find first circle tangent on two curves that satisfies the given argument.

    Args:
        c1: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 1
        c2: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2
        arg: Either the line factors as given by np.polyfit(), defining the line at which the center is located, or the radius
    
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

            lf1 = lfp(c1[i], c1[i+1])
            lf2 = lfp(c2[j], c2[j+1])
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


def arc_gen(p1: Union[list,tuple,np.ndarray], p2: Union[list,tuple,np.ndarray], p0: Union[list,tuple,np.ndarray], n: int, reflex: bool = False) -> np.ndarray:
    """
    Generate points of an arc, from point 1 to point 2 with point 0 as center. Generated points include point 1 and 2.

    Args:
        p1: [x, y] coordinates of point 1
        p2: [x, y] coordinates of point 2
        p0: [x, y] coordinates of the center
        n: Number of points to be generated
        reflex: If False, the smaller of the two possible arcs will be generated, else the greater
    
    Returns:
        [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the arc

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
def bezier(p: Union[list,tuple,np.ndarray], w: Union[list,tuple,np.ndarray] = 1) -> Callable[[float], np.ndarray]:
    """
    Return the function of a rational bezier curve with control points p and weights w.

    Args:
        p: [[x0, y0], [x1, y1], ... , [xm, ym]] the matrix containing all the point coordinates
        w: [w0, w1, ..., wn] the vector containing all the weight values, should have same length as p
    
    Returns:
        function that generates points based on the u parametre (0 <= u <= 1)

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


def parallcrv(c: Union[list,tuple,np.ndarray]) -> np.ndarray:
    """
    Find a curve parallel to the first, at unit distance.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
    
    Returns:
        [[x0, y0], [x1, y1], ... , [xn, yn]], adding this matrix to the argument curve will give the displaced curve

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
def crv_len(c: Union[list,tuple,np.ndarray]) -> np.ndarray:
    """
    Calculate the length of the curve at every point of it.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
    
    Returns:
        [l0, l1, ..., ln] the vector containing the lengths for every point

    """
    c = np.array(c)
    norms = np.hypot(c[1:,0] - c[0:-1,0], c[1:,1] - c[0:-1,1])
    sh = np.shape(norms)[0]
    clen = np.tril(np.ones((sh,sh))) @ norms
    clen = np.insert(clen, 0, 0)
    return clen


def crv_ang(c: Union[list,tuple,np.ndarray]) -> np.ndarray:
    """
    Calculate the additive curvature angle of the curve at every point of it.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
    
    Returns:
        [l0, l1, ..., ln-1] the vector containing the angle for every segment

    """
    c = np.array(c)
    thetas = vectorangle(c[2:] - c[1:-1], c[1:-1] - c[0:-2])
    sh = np.shape(thetas)[0]
    cang = np.tril(np.ones((sh,sh))) @ thetas
    cang = np.hstack(([0], cang))
    return cang


def crv_curvature(c: Union[list,tuple,np.ndarray]) -> np.ndarray:
    """
    Calculate the curvature of the curve.

    Args:
        c: [[x0, y0], [x1, y1], ... ,[xn, yn]] the matrix containing the points of the curve

    Returns:
        vector containing the curvature value for every segment of the curve

    """
    c = np.array(c)
    cang = vectorangle(c[2:] - c[1:-1], c[1:-1] - c[0:-2])
    cang = np.append(np.insert(cang, 0, -cang[0]), -cang[-1])
    slen = np.hypot(c[1:,0] - c[0:-1,0], c[1:,1] - c[0:-1,1])
    return (cang[0:-1] + cang[1:]) / (2 * slen)


# SINGLE CURVE MORPHOLOGY
def smooth_zigzag(c: Union[list,tuple,np.ndarray], minang: float, varp: float = 0.5) -> np.ndarray:
    """
    Smoothen a curve by removing zig-zagging segments.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
        minang: minimum angle in radians, of a zig-zag
        varp: must be possitive, the greater the variation power (varp) the less similar two consecutive angles need to be to be considered a zig-zag
    
    Returns:
        [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the smoothened curve

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


def smooth_loops(c: Union[list,tuple,np.ndarray]) -> np.ndarray:
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


def smooth_fillet_crcl(c: Union[list,tuple,np.ndarray], minang: float, c_factor: float = 0.8) -> np.ndarray:
    """
    Smoothen a curve's sharp angles by filleting them away with circular arcs.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
        minang: any angle equal or greater than this will be filleted
        c_factor: a magic number between 0 and 1. The bigger the number the bigger the radius of the fillet, above 0.5 may lead to funky results for consecutive filleted angles

    Returns:
        [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing the points of the smooth curve

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
        lf1 = lfp(c[i], ptan1)
        lf2 = lfp(c[i], ptan2)
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


def smooth_fillet_bezier(c: Union[list,tuple,np.ndarray], minang: float, b_factor: float = 0.8) -> np.ndarray:
    """
    Smoothen a curve's sharp angles by filleting them away with bezier.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
        minang: any angle equal or greater than this will be filleted
        b_factor: a magic number between 0 and 1. The bigger the number the bigger the "radius" of the fillet

    Returns:
        [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing the points of the smooth curve

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


def smooth_spline(c: Union[list,tuple,np.ndarray], centripetal: bool = True, smoothfact: float = 0.7) -> np.ndarray:
    """
    Smoothen a curve by aproximating it with a spline.

    Args:
        c: [[x0, y0], [x1, y1], ... ,[xn, yn]] the matrix containing the points of the curve
        centripetal: if True, centripetal algorithm is used
        smoothfact: smoothness factor

    Returns:
        [[x0, y0], [x1, y1], ... ,[xn, yn]] the matrix containing the points of the smooth curve
    
    """
    ctrspts_size = int(np.ceil((1-smoothfact) * len(c)))
    spline = fitting.approximate_curve(list(c), 3, centripetal=centripetal, ctrlpts_size=5)
    spline.delta = 1/len(c)
    return np.array(spline.evalpts)


def clr_duplicates(c: Union[list,tuple,np.ndarray], tol: float = 10**-4) -> np.ndarray:
    """
    Clear "duplicate" points of a curve, within a certain tolerance.

    Args:
        c: [[x0, y0], [x1, y1], ... ,[xn, yn]] the matrix containing the points
        tol: The tolerance when checking if two points are common.
    
    Returns:
        the array without the duplicate values    
    
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


def crv_cleanup(c: Union[list,tuple,np.ndarray], zigzagang: float, filletang: float, fmethod: str, varp: float = 0.3, ffactor: float = 0.8, tol: float = 10**-4) -> np.ndarray:
    """
    Clean a curve from zigzagging patterns, loops, duplicate points and sharp angles.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
        zigzagang: the minang argument of the smooth_zigzag function
        filletang: the minang argument of the smooth_fillet function
        fmethod: determines the method of the fillet, can be either 'circle' for circular arc filleting or 'bezier' for bezier curve filleting
        varp: the varp argument of the smooth_zigzag function
        r_factor: the r_factor argument of the smooth_fillet function
        tol: the tolerance of clr_duplicates function

    Returns:
        [[x0, y0], [x1, y1], ... , [xm, ym]] the coords of the clean curve

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


def crv_div(c: Union[list,tuple,np.ndarray], div: Union[list,tuple,np.ndarray]) -> list:
    """
    Divide a curve into segments.

    Args:
        c: [[x0, y0], [x1, y1], ... ,[xn, yn]] the matrix containing the points of the curve
        div: The list of the fractions of the curve length at which the curve is divided (0< div <1)
    
    Returns:
        list containing:
        -cd (ndarray): [[x0, y0], [x1, y1], ... ,[x(m), y(m)]] the matrix containing the points of the curve, including points of division
        -j (ndarray): vector containing the indexes of the division points in cd

    """
    pass # NEED TO EDIT IT WITH SPLINE
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


def crv_ln_cut(c: Union[list,tuple,np.ndarray], arg: Union[float,list,tuple,np.ndarray], op: str, return_index: bool = False) -> np.ndarray:
    """
    Return the curve part that satisfy the inequality: c (op) arg.

    Args:
        c (array-like): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the curve
        arg (float or array-like): Either the vector with the factors of a line as given by np.polyfit(), or an x value
        op (str): Can be one of the four inequality operators '<', '>', '<=', '>='
        return_index (bool): if True, return the indexes instead of the curve

    Returns:
        (ndarray): Curve consisting of the parts satisfying the op according to arg

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


def crv_fit2p(c: Union[list,tuple,np.ndarray], p1: Union[list,tuple,np.ndarray], p2: Union[list,tuple,np.ndarray], i1: int = 0, i2: int = -1, proxi_snap: bool = False) -> np.ndarray:
    """
    Transforms points of a curve, so the indexed points fall onto given coordinates.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates, must be ascending
        p1: [x, y] coordinates of point 1
        p2: [x, y] coordinates of point 2
        i1: index of point of curve that will be moved onto point 1
        i2: index of point of curve that will be moved onto point 2
        proxi_snap: if True, i1 and i2 are ignored and instead the points of the curve closest to the points 1 and 2 will be selected accordingly.
    
    Returns:
        transformed point coordinates

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


def crv_snap(c: Union[list,tuple,np.ndarray], coords: Union[list,tuple,np.ndarray]) -> np.ndarray:
    """
    Snap a curve's end onto given coordinates so it retains its original general shape. (The curve's start remains at the original point)

    Args:
        c: [[x0, y0], [x1, y1], ... , [xm, ym]] the matrix containing all the point coordinates of curve
        coords: [x, y] coordinates to snap to
        end: The end of the cuvre that will be snapped onto the coords, 0 assigns the beggining of the list and -1 the end

    Returns:
        snapped curve

    """
    c = np.array(c)
    clen = crv_len(c)
    lenfract = clen / clen[-1]
    lenfract = np.transpose([lenfract,lenfract])
    cf = crv_fit2p(c, c[0], coords)
    return cf * lenfract + (1 - lenfract) * c


# CURVE INTERACTIONS
def mean_crv(c1: Union[list,tuple,np.ndarray], c2: Union[list,tuple,np.ndarray], ratio: Union[float,list,tuple,np.ndarray] = 0.5, centripetal: bool = True) -> np.ndarray:
    """
    Generate a mean curve based on a ratio number.

    Args:
        c1: [[x0, y0], [x1, y1], ... , [xm, ym]] the matrix containing all the point coordinates of curve 1
        c2: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2
        ratio: the ratio of the curves to be used
    
    Returns:
        [[x0, y0], [x1, y1], ... , [xm, ym]] the calculated curve

    """
    c1, c2 = list(c1), list(c2)

    if len(c1) > 3:
        c1spl = fitting.interpolate_curve(c1, 3, centripetal=centripetal)
    else:
        c1spl = fitting.interpolate_curve(c1, len(c1) - 1, centripetal=centripetal)
    if len(c2) > 3:
        c2spl = fitting.interpolate_curve(c2, 3, centripetal=centripetal)
    else:
        c2spl = fitting.interpolate_curve(c2, len(c2) - 1, centripetal=centripetal)

    delta = 1/(max(len(c1), len(c2)) + 1)
    c1spl.delta, c2spl.delta = delta, delta
    c1, c2 = np.array(c1spl.evalpts), np.array(c2spl.evalpts)

    if np.array(ratio).ndim == 1:
        ratio = np.transpose([ratio, ratio])
    
    return ratio * c1 + (1 - ratio) * c2


def crv_dist(c1: Union[list,tuple,np.ndarray], c2: Union[list,tuple,np.ndarray]) -> np.ndarray:
    """
    Measure distance between points of two curves.

    Args:
        c1: [[x0, y0], [x1, y1], ... , [xm, ym]] the matrix containing all the point coordinates of curve 1
        c2: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2

    Returns:
        M x N matrix containing distances of all points, M and N being the number of points of curve 1 and 2

    """
    c1, c2 = np.array(c1), np.array(c2)
    m, n = np.shape(c1)[0], np.shape(c2)[0]
    c1 = np.repeat(c1[:,np.newaxis,:], n, axis=1)
    c2 = np.repeat(c2[np.newaxis,:,:], m, axis=0)
    c1x, c1y = c1[:,:,0], c1[:,:,1]
    c2x, c2y = c2[:,:,0], c2[:,:,1]
    return np.hypot(c1x-c2x, c1y-c2y)


def crv_p_dist(c: Union[list,tuple,np.ndarray], p: Union[list,tuple,np.ndarray]) -> float:
    """
    Accurately measure distance of a point from curve.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve
        p: [x, y] coordinates of the point
    
    Returns:
        distance

    """

    num = np.abs((c[1:,1] - c[0:-1,1]) * p[0] - (c[1:,0] - c[0:-1,0]) * p[1] + c[1:,0] * c[0:-1,1] - c[1:,1] * c[0:-1,0])
    den = ((c[1:,1] - c[0:-1,1])**2 + (c[1:,0] - c[0:-1,0])**2)**0.5
    lnrdist = np.min(num/den)
    vrtxdist = np.min(np.hypot(c[:,0] - p[0], c[:,1] - p[1]))
    return min(lnrdist, vrtxdist)


def crv_common(c1: Union[list,tuple,np.ndarray], c2: Union[list,tuple,np.ndarray], tol: float = 10**-4) -> np.ndarray:
    """
    Check if two curves have common points within tolerance.

    Args:
        c1: [[x0, y0], [x1, y1], ... , [xm, ym]] the matrix containing all the point coordinates of curve 1
        c2: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2
        tol: tolerance

    Returns:
        M x N matrix containing True if points are within tolerance, M and N being the number of points of curve 1 and 2

    """
    m, n = np.shape(c1)[0], np.shape(c2)[0]
    tolmat = np.ones((m, n)) * tol
    return crv_dist(c1, c2) <= tolmat


def clr_common(c1: Union[list,tuple,np.ndarray], c2: Union[list,tuple,np.ndarray], tol: float = 10**-4) -> np.ndarray:
    """
    Clears the points of c1 that are common with c2.

    Args:
        c1: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 1
        c2: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2
        tol: tolerance

    Returns:
        curve 1 cleared of common points

    """
    boolmat = crv_common(c1, c2, tol)
    if c1 == c2:
        diag = np.arange(0, np.shape(c1)[0])
        boolmat[diag, diag] = False
    inds = np.nonzero(np.sum(boolmat), axis = 0)[0]
    return np.delete(c1, inds)


def common_ends(c1: Union[list,tuple,np.ndarray], c2: Union[list,tuple,np.ndarray], end1: int = None, end2: int = None, tol: float = 10**-4) -> int:
    """
    Find two end points of two curves that are common.

    Args:
        c1: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        c2: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        end1: The end of the first list that will be patched, in case of multiple common points. 0 assigns the beggining of the list and -1 the end
        end2: The end of the second list that will be patched, in case of multiple common points. 0 assigns the beggining of the list and -1 the end
        tol: The tolerance when checking if two points are common.
    
    Returns:
        Int showing which ends meet
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


def crv_patch(c1: Union[list,tuple,np.ndarray], c2: Union[list,tuple,np.ndarray], comend_args: Union[list,tuple] = []) -> np.ndarray:
    """
    Patch two curves into one.

    Args:
        c1: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        c2: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        comend_args: list containing the arguments for the common ends function [end1, end2, tol]
    
    Returns:
        curve consisting of two curves
    
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


def DEPRECATED_crv_fillet(c1: Union[list,tuple,np.ndarray], c2: Union[list,tuple,np.ndarray], r: float, n: Union[int,list,tuple], comend_args: Union[list,tuple] = [], reflex: bool = False, crosscheck: bool = False) -> list:
    """
    ### DEPRECATED ###
    Fillet two patchable curves into one. If it fails to find a fillet circle, a simple patched curve is returned.

    Args:
        c1: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 1
        c2: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2
        r: The radius of the fillet arc
        n: Either the number of points, or a vector with the angular and longitudinal density [ad, ld], ad being the number of points in a full circle, ld being the number of points per unit of length
        comend_args: list containing the arguments for the common ends function [end1, end2, tol]
        reflex: If True, the generated arc will be "indented" rather than "smooth"
        crosscheck: If True, checks if the curves cross each other at any point. If they do, the last intersection is assigned as the new start point
    
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

            lf1 = lfp(c1[i], c1[i+1])
            lf2 = lfp(c2[j], c2[j+1])
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
        c = crv_patch(c1, c2, [0, 0])
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


def crv_fillet(c1: Union[list,tuple,np.ndarray], c2: Union[list,tuple,np.ndarray], r: float, n: Union[int,list,tuple], comend_args: Union[list,tuple] = [], reflex: bool = False, crosscheck: bool = False) -> list:
    """
    Fillet two patchable curves into one. If it fails to find a fillet circle with requested radius an aproximate radius is used. 
    Note: This fillet finds only circles tangent to diverging portions of curves, this works well for our purposes. To generalise it, one must modify the crcl_tang_2ln_rlim
    function to take a quadrant argument. But that allows the function to sometimes find unholy circles that will end up making airfoils into tear drops. Thats bad.

    Args:
        c1: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 1
        c2: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2
        r: The radius of the fillet arc
        n: Either the number of points, or a vector with the angular and longitudinal density [ad, ld], ad being the number of points in a full circle, ld being the number of points per unit of length
        comend_args: list containing the arguments for the common ends function [end1, end2, tol]
        reflex: If True, the generated arc will be "indented" rather than "smooth"
        crosscheck: If True, checks if the curves cross each other at any point. If they do, the last intersection is assigned as the new start point
    
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
    lf1 = lfp(c1[i], c1[i+1])
    lf2 = lfp(c2[j], c2[j+1])
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


# GEOSHAPE CLASS
class GeoShape:
    """
    Describes a 2d geometry. Carries data relating to the points, sequences and segments of a geometric shape.

    Attr:
        points (ndarray):  [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        squencs (list): A dictionary of lists of indexes pointing in the points matrix, describing the points each curve passes through
        shapes (list): A list of lists of indexes pointing in the squencs list, showing the sequences that each shape consists of

    Methods:
        transform: Transform the GeoShape's points
        recenter: Center the GeoShape, around point
    
    """

    def __init__(self, points: np.ndarray, squencs: list, shapes: list):
        self.points = points
        self.squencs = squencs
        self.shapes = shapes


    def __str__(self):
        return 'Points: ' + str(self.points) + '\nSequences: ' + str(self.squencs) + '\nShapes: ' + str(self.shapes)


    def plot(self, lines: bool = True, indxs: bool = True, marks: bool = True, show: bool = True) -> None:
        """
        Plot the geometric shape.

        Args:
            lines (bool): If True, the lines of the curves will be plotted
            indxs (bool): If True, the points will be numbered according to their index in the point matrix
            marks (bool): If True, the points will be marked with dots
            show (bool): If True, the plot will be shown after plotting
        
        """
        points = self.points
        squencs = self.squencs

        if lines:
            for squence in squencs:
                plt.plot(points[squence,0], points[squence,1], '-')
        
        if marks:
            plt.plot(points[:,0], points[:,1], '.r')
        
        if indxs:
            for i in range(np.shape(points)[0]):
                plt.text(points[i,0], points[i,1], i)

        plt.grid(visible=True)
        plt.axis('equal')

        if show:
            plt.show()


    def transform(self, center: Union[list,tuple,np.ndarray], theta: float, sf: Union[list,tuple,np.ndarray], tv: Union[list,tuple,np.ndarray]) -> list:
        """
        Transform the GeoShape's points.

        Args:
            center: [x, y] the coordinates of the center of scaling and rotation
            theta: the angle of rotation
            sf: [sfx, sfy] the factors of scaling
            tv: [vx, vy] the components of displacement vector
        
        """
        points = self.points
        points = rotate(points, center, theta)
        points = scale(points, center, sf)
        self.points = translate(points, tv)


    def recenter(self, point: Union[list,tuple,np.ndarray]) -> np.ndarray:
        """
        Center the GeoShape, around point.

        Args:
            point: [x, y] coordinates of the point
        
        Returns:
            [x, y] coordinates of displacement vector
        
        """
        x0 = (np.max(self.points[:,0]) + np.min(self.points[:,0])) / 2
        y0 = (np.max(self.points[:,1]) + np.min(self.points[:,1])) / 2
        tv = point - np.array([x0, y0])
        self.points = translate(self.points, tv)
        return tv


    def remove_sequence(self, squence_i: int) -> list:
        """
        Remove a sequence from the geoshape. No point deletion takes place.

        Arg:
            squence_i: the index of the sequence to be removed.
        
        Returns:
            list containing:
            - list containing all the indexes of all the shapes the sequence was used in
            - list of lists containing all the corresponding (in-shape) indexes from which the sequence was removed 

        """
        self.squencs.pop(squence_i)
        affected_shapes = []
        popindxs = []
        shapes = self.shapes
        for i in range(len(shapes)):
            shape = shapes[i]
            j = 0
            while j < len(shape):
                if shape[j] > squence_i:
                    shape[j] -= 1
                elif shape[j] == squence_i:
                    shape.pop(j)
                    affected_shapes.append(i)
                    popindxs.append(j)
                    continue
                j += 1
        return [affected_shapes, popindxs]


    def remove_point(self, point_i: int):
        """
        Remove a point from the GeoShape.

        Args:
            point_i: point index

        """
        # edit sequences
        for i in range(len(self.squencs)):
            sequence = np.array(self.squencs[i])
            greaterindxs = np.nonzero(sequence > point_i)[0]
            equalindxs = np.nonzero(sequence == point_i)[0]
            sequence[greaterindxs] -= 1
            sequence = np.delete(sequence, equalindxs)
            self.squencs[i] = list(sequence)
        # remove point
        self.points = np.delete(self.points, point_i, axis=0)


    def replace_point(self, del_i: int, new_i: int, sql: Union[list,tuple] = []):
        """
        Replace a point index in specified sequences with another.

        Args:
            del_i: index to be replaced
            new_i: index that will replace
            sql: contains the indexes of the sequences that will be researched

        """
        reflist = self.point_ref(del_i, sql)
        for i in range(len(reflist)):
            seq = np.array(self.squencs[reflist[i]])
            seq[seq==del_i] = new_i
            self.squencs[reflist[i]] = list(seq)


    def replace_sequence(self, del_i: int, new_i: int, spl: Union[list,tuple] = []):
        """
        Replace a point index in specified sequences with another.

        Args:
            del_i: index to be replaced
            new_i: index that will replace
            sql: contains the indexes of the sequences that will be researched

        """
        reflist = self.sequence_ref(del_i, spl)
        for i in range(len(reflist)):
            shape = np.array(self.shapes[reflist[i]])
            shape[shape==del_i] = new_i
            self.shapes[reflist[i]] = list(shape)


    def split_sequence(self, squence_i: int, div_i: Union[list,tuple]) -> list:
        """
        Split a sequence at given indexes.

        Args:
            squence_i: the index of the sequence
            div_i: contains the indexes where the sequence will be divided
        
        Returns:
            indexes of new sequences

        """
        sequence = self.squencs[squence_i]
        # Generate sequence parts
        div_i = np.flip(np.sort(div_i))
        seqlist = []
        for i in div_i:
            if (i < len(sequence) - 1) and (i > 0):
                seqlist.append(sequence[i:])
                sequence = sequence[0:i+1]
        seqlist.append(sequence)
        seqlist.reverse()
        lsl = len(seqlist)
        # Remove old sequence
        affected_shapes, insertindxs = self.remove_sequence(squence_i)
        # Add new sequences to squencs
        seqindxs = list(range(len(self.squencs), len(self.squencs) + lsl))
        self.squencs = self.squencs + seqlist
        # Add new sequences to shapes
        for i in range(0, len(affected_shapes)):
            if (affected_shapes[i] == affected_shapes[i-1]) and (i > 0):
                addindx += lsl
            else:
                addindx = 0

            for j in range(lsl):
                self.shapes[affected_shapes[i]].insert(j + addindx + insertindxs[i], seqindxs[j])

        return seqindxs


    def sequence_ref(self, squence_i: int, spl: Union[list,tuple] = []) -> list:
        """
        Get all the shape indexes that reference the indexed sequence.

        Args:
            squence_i: the index of the sequence
            spl: contains the indexes of the shapes that will be researched

        Returns:
            list containing all shape indexes referencing the sequence

        """
        reflist = []
        shapes = self.shapes
        if len(spl) == 0:
            spl = list(range(len(self.shapes)))
        for i in spl:
            if squence_i in shapes[i]:
                reflist.append(i)
        return reflist


    def point_ref(self, point_i: int, sql: Union[list,tuple] = []) -> list:
        """
        Get all the sequence indexes that reference the indexed point.

        Args:
            point_i: the index of the point
            sql: contains the indexes of the sequences that will be researched

        Returns:
            list containing all sequence indexes referencing the point

        """
        reflist = []
        squencs = self.squencs
        if len(sql) == 0:
            sql = list(range(len(self.squencs)))
        for i in sql:
            if point_i in squencs[i]:
                reflist.append(i)
        return reflist


    def add_crv(self, crv: Union[list,tuple,np.ndarray]):
        """
        Add a curve (as mentioned in geometrics package) to the GeoShape.

        Args:
            crv: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the curve

        """
        i1 = len(self.points)
        i2 = i1 + len(crv)
        self.squencs.append(list(range(i1,i2)))
        if i1 == 0:
            self.points = np.array([], dtype=np.int64).reshape(0,2)
        self.points = np.vstack((self.points, crv))


    def clear_duplicates(self, tol: float):
        """
        Clear all duplicate points, sequences and shapes from the GeoShape.

        Args:
            tol: the maximum distance two points must have to be considered the same point.

        """
        # Points
        i = 0 
        while i < len(self.points) - 1:
            j = i + 1
            while j < len(self.points):
                if comcheck(self.points[i], self.points[j], tol):
                    self.replace_point(j, i)
                    self.remove_point(j)
                else:
                    j += 1
            i += 1
        # Sequences
        squencs = self.squencs
        for squence in squencs:
            i = 0
            while i < len(squence) - 1:
                if squence[i] == squence[i-1]:
                    squence.pop(i)
                else:
                    i += 1

        i = 0
        while i < len(squencs) - 1:
            j = i + 1
            while j < len(squencs):
                if (squencs[i] == squencs[j]) or (squencs[i] == list(np.flip(squencs[j]))):
                    self.replace_sequence(j, i)
                    self.remove_sequence(j)
                else:
                    j += 1
            i += 1
        # Shapes
        shapes = self.shapes
        for shape in shapes:
            i = 0
            while i < len(shape) - 1:
                if shape[i] == shape[i-1]:
                    shape.pop(i)
                else:
                    i += 1

        i = 0
        while i < len(shapes) - 1:
            j = i + 1
            while j < len(shapes):
                if (shapes[i] == shapes[j]) or (shapes[i] == list(np.flip(shapes[j]))):
                    self.shapes.pop(j)
                else:
                    j += 1
            i += 1


    def clear_empties(self, length1squences: bool = True):
        """
        Remove empty sequences and shapes.

        Args:
            length1squences: if True, also remove sequences with just one index in them

        """
        if length1squences:
            limit = 2
        else:
            limit = 1

        squencs = self.squencs
        i = 0
        while i < len(squencs):
            if len(squencs[i]) < limit:
                self.remove_sequence(i)
            else:
                i += 1

        shapes = self.shapes
        i = 0
        while i < len(shapes):
            if len(shapes[i]) < limit:
                shapes.pop(i)
            else:
                i += 1


    def clear_unused(self):
        """
        Remove unused points and sequences.
        """
        i = 0
        while i < len(self.points):
            if len(self.point_ref(i)) == 0:
                self.remove_point(i)
            else:
                i += 1
        i = 0
        while i < len(self.squencs):
            if len(self.sequence_ref(i)) == 0:
                self.remove_sequence(i)
            else:
                i += 1


    def shape_clarify(self, shape_i: int, sqi: int = 0) -> list:
        """
        Get order and orientation of all sequences of a proper shape.

        Args:
            shape_i: the index of the shape
            sqi: the within shape index of the sequence to start from

        Returns:
            a list of the within-shape indexes of the ordered sequences 
            a list of integers showing if the sequence is reversed or not

        """
        shape = self.shapes[shape_i]
        squencs = self.squencs
        counter = 1
        order_sqs = [sqi]
        orient = [0]
        while counter < len(shape):
            counter += 1
            pi = squencs[shape[order_sqs[-1]]][opst_i(orient[-1])]
            reflist = self.point_ref(pi, shape)
            reflist.remove(shape[order_sqs[-1]])
            nsequence = reflist[0]
            next_sqi = shape.index(nsequence)
            next_orient = squencs[nsequence].index(pi)
            next_orient = opst_i(opst_i(next_orient))
            order_sqs.append(next_sqi)
            orient.append(next_orient)
        
        return [order_sqs, orient]


    def outer_shell(self) -> list:
        """
        Get the outer sequences that envelop the entire domain. The sequence curves should not cross each other, and the domain should not be split into disconnected shapes.

        Returns:
            list containing indexes of the sequences

        """
        points = self.points
        squencs = self.squencs
        # Trace a ray from any point ([0,0]), across the domain to get any outer sequence.
        raylen = np.max(crv_dist([[0,0]], points))
        ray = np.array([[0,0], (points[squencs[0][0]] + points[squencs[0][1]]) / 2])
        ray = raylen * ray / np.linalg.norm(ray[1])
        maxdistr = 0
        for i in range(len(squencs)):
            itr, ptr = raytrace(points[squencs[i]], ray)
            if len(ptr) > 0:
                distr = np.linalg.norm(ptr[-1])
                if distr > maxdistr:
                    maxdistr, maxitr, maxptri, maxsqi = distr, itr[-1], ptr[-1], i
        # Find anti-clock-wise direction
        angs = np.sign(vectorangle(points[squencs[maxsqi][maxitr]] - maxptri, -maxptri))
        indx = int((-angs-1)/2)                     # transfrom sign to first/last indx
        n_sqi = maxsqi
        outer_shell = []
        while n_sqi not in outer_shell:
            p_sqi = n_sqi
            outer_shell.append(p_sqi)
            indx2 = nxt_i(indx)
            pvect = points[squencs[p_sqi][indx2]] - points[squencs[p_sqi][indx]]
            pi = squencs[p_sqi][indx]
            node_sqncs = self.point_ref(pi)
            node_sqncs.remove(p_sqi)
            minang = 2*np.pi
            for sqi in node_sqncs:
                _indx = squencs[sqi].index(pi)
                _indx2 = nxt_i(_indx)           # transform index to second / second last
                nvect = points[squencs[sqi][_indx2]] - points[squencs[sqi][_indx]]
                ang = vectorangle(nvect, pvect)
                if ang < 0:
                    ang = 2*np.pi + ang
                if ang < minang:
                    minang = ang
                    n_sqi = sqi
                    indx = _indx
            indx = opst_i(indx)

        return outer_shell


    def prox_point_index(self, point: Union[list,tuple,np.ndarray], squenc_i: int) -> int:
        """
        Get the index of a point on the sequence, closest to the point given.

        Args:
            point: [x, y] coordinates of point
            squenc_i: the index of the sequence to investigate
        
        Returns:
            the sequence index of the closest point

        """
        crv = self.points[self.squencs[squenc_i]]
        return np.argmin(np.hypot(crv[:,0] - point[0], crv[:,1] - point[1]))


    def snap_intersection(self, inters_indx: int, snap_cords: Union[list,tuple]):
        """
        Snap an intersection point of sequences to new coordinates. Intersection point must only be at an end point of all sequences referencing it.

        Args:
            inters_indx: the point index of the intersection.
            snap_cords: the coordinates to snap it

        """
        points = self.points
        prevcords = np.array(points[inters_indx])
        squences = self.squencs
        sqindxs = self.point_ref(inters_indx)
        for sqi in sqindxs:
            points[inters_indx] = prevcords
            squence = squences[sqi]
            if squence[-1] == inters_indx:
                points[squence] = crv_snap(points[squence], snap_cords)
            elif squence[0] == inters_indx:
                points[squence] = np.flipud(crv_snap(np.flipud(points[squence]), snap_cords))
        self.points = points


    def multipoint_ref(self, point_i: Union[list, tuple], sql: Union[list,tuple] = []):
        """
        Get all sequences of the GeoShape that contain all of given point indexes.
        """
        refset = []
        for pi in point_i:
            reflist = self.point_ref(pi, sql)
            refset = refset + reflist
        refset = set(refset)
        reflist = []
        for sqi in refset:
            refbool = True
            for pi in point_i:
                if pi not in self.squencs[sqi]:
                    refbool = False
                    break
            if refbool:
                reflist.append(sqi)
        return reflist


def gs_merge(gsl: list) -> GeoShape:
    """
    Merge a list of GeoShapes, into one GeoShape object.

    Args:
        gsl: A list containing all the shape objects
    
    Returns:
        GeoShape object consisting of all others

    """
    # Assemble all the shapes
    points = np.array([], dtype=np.int64).reshape(0,2)
    squencs = []
    shapes = []
    i = 0
    j = 0
    for gs in gsl:
        points = np.vstack((points, gs.points))
        # Fix squencs indices
        gs.squencs = list(map(lambda y: list(map(lambda x: x+i, y)), gs.squencs))
        squencs = squencs + gs.squencs
        # Fix shapes indices
        gs.shapes = list(map(lambda y: list(map(lambda x: x+j, y)), gs.shapes))
        shapes = shapes + gs.shapes
        i = int(np.shape(points)[0])
        j = int(len(squencs))
    
    # Removing duplicate squencs and shapes
    # Squencs
    un_squencs = []
    inv = np.zeros(len(squencs), dtype=int)

    for i in range(len(squencs)):
        if squencs[i] in un_squencs:
            inv[i] = un_squencs.index(squencs[i])
        else:
            inv[i] = len(un_squencs)
            un_squencs.append(squencs[i])

    squencs = un_squencs
    for i in range(len(shapes)):
        shapes[i] = list(inv[shapes[i]])

    # Shapes
    un_shapes = []
    for i in range(len(shapes)):
        if not (shapes[i] in un_shapes):
            un_shapes.append(shapes[i])

    shapes = un_shapes
    return GeoShape(points, squencs, shapes)


def gs_copy(gs: GeoShape) -> GeoShape:
    """
    Make copy of a GeoShape and return it.

    Args:
        gs: the GeoShape to be copied

    Returns:
        copy of the GeoShape

    """
    return GeoShape(gs.points, gs.squencs, gs.shapes)


def crv2gs(curvesls: list) -> list:
    """
    Generate a GeoShape from geometric curves as defined in geometrics package. 

    Args:
        curvesls: list of the lists containing all the geometric curves, they must have common ends and be ordered the way they connect

    Returns:
        list with GeoShape objects

    """
    gsl = []
    for curves in curvesls:
        j1 = 0
        squencs = []
        points = np.array([], dtype=np.int64).reshape(0,2)

        for i in range(len(curves)-1):
            points = np.vstack((points, curves[i][0:-1]))
            j2 = np.shape(points)[0]
            squencs = squencs + [list(range(j1, j2+1))]
            j1 = j2

        points = np.vstack((points, curves[-1][0:-1]))
        squencs = squencs + [list(range(j1, np.shape(points)[0])) + [0]]
        shapes = [list(range(len(curves)))]
        gsl.append(GeoShape(points, squencs, shapes))

    return gsl


def cross_check(gs: GeoShape) -> bool:
    """
    Check if any point of any shape in a GeoShape, is inside the polygon defined by the points of another shape of the GeoShape.
    This type of check identifies *most* crossings between shapes, and this is good enough. 

    Returns:
        bool: True if there is a "cross"

    """
    points = gs.points
    squencs = gs.squencs
    polygons = []
    for shape in gs.shapes:
        indcs = np.hstack(squencs[shape])
        polygons.append(points[indcs])

    for i in range(len(polygons)):
        for j in range(len(polygons)):
            if np.any(Path(polygons[i]).contains_points(polygons[j])):
                return True
    return False
