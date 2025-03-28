#--------- READ ME----------------------------------------------------------------
# A few curve generation functions. These fucntions work by generating a curve
# based on the geometry of an already existing curve. Their logic can seem
# convoluted and their use can be unstable.
#---------------------------------------------------------------------------------

import numpy as np
import geometrics as gmt
from typing import Union, Callable


def weav(c: Union[list,tuple,np.ndarray], theta: float, weight_fun: Callable[[np.ndarray], np.ndarray], n: int, closed_brackets: bool = True) -> np.ndarray:
    """
    Generate a curve by calculating the weighted average y coordinate between the curve and a line, applying a weight function.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the curve
        theta: the angle at which to rotate the curve before calculating the weighted average
        weight_fun: the function from which the weights of the weighted average calculation are taken
        n: the number of points the generated curve will have
        closed_brackets: If True, after rotation, all curve points with x values exceeding the segment created by the first and last point of the curve, are removed
    
    Returns:
        [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the generated curve

    """
    c = gmt.rotate(c, np.array([0,0]), theta)

    if closed_brackets:
        crange = [c[0,0], c[-1,0]]
        c = c[np.argwhere(np.logical_and(c[:,0] >= min(crange), c[:,0] <= max(crange)))][:,0,:]        # Only keep curve values between cmin and cmax
    else:
        crange = [min(c[:,0]), max(c[:,0])]

    lf = gmt.lfp(c[0], c[-1])                                                                          # Calculate line factors
    x = c[:,0]
    ly = np.polyval(lf, x)                                                                             # Place line points
    wx = 2 * (x - (crange[1] + crange[0])/2) / abs(crange[1] - crange[0])                              # Map x values to the range [-1,1]
    wf = weight_fun(wx)
    mc = np.transpose([x, wf*c[:,1] + (1-wf)*ly])
    mc = gmt.rotate(mc, np.array([0,0]), -theta)
    mcs = gmt.fitting.interpolate_curve(list(mc), 3, centripetal = True)
    mcs.delta = 1/(n+1)
    return np.array(mcs.evalpts)


def arc_p(c: Union[list,tuple,np.ndarray], p: Union[list,tuple,np.ndarray], n: int) -> np.ndarray:
    """
    Generate an arc curve, arcing from the first curve point to the last, that would, if extended pass through the given point.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the curve
        p: [x, y] coordinates of the point
        n: the number of points the generated curve will have
    
    Returns:
        [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the generated curve

    """
    p0 = gmt.crcl_3p(p, c[0], c[-1])
    return gmt.arc_gen(c[0], c[-1], p0, n)


def arc_tang(c: Union[list,tuple,np.ndarray], n: int) -> np.ndarray:
    """
    Generate an arc curve, tangent of the first curve segment and arcing to the last curve point

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the curve
        n: the number of points the generated curve will have
    
    Returns:
        [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the generated curve

    """
    lf = gmt.lfp(c[0], c[1])
    p0 = gmt.crcl_tang_lnpp(lf, c[0], c[-1])
    return gmt.arc_gen(c[0], c[-1], p0, n)


def bezier(c: Union[list,tuple,np.ndarray], m: int, gens: int, n: int, w: Union[list,tuple,np.ndarray] = 1) -> np.ndarray:
    """
    Generate a bezier curve based on a curve.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xm, ym]] the matrix containing all the point coordinates of the curve
        m:  2 <= m <= np.shape(c)[0], number of points of the curve to take as control points, if m = 'all', take all points
        gens: number of generations (each generation is based on the previous curve)
        n: the number of points the generated curve will have
        w: weights of the control points, length must be equal to the (int) m argument
    
    Returns:
        [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the generated curve

    """
    for i in range(gens):
        t = np.linspace(0,1,n)

        if not (type(m) == str):
            c = c[np.array(np.round(np.linspace(0, np.shape(c)[0]-1, m)), dtype=int)]
        
        c = gmt.bezier(c, w)(t)
    return c


def marriage(c1: Union[list,tuple,np.ndarray], c2: Union[list,tuple,np.ndarray], x1: float, x2: float, x0: float, n: int, w: Union[list,tuple,np.ndarray] = 1) -> np.ndarray:
    """
    Generate a curve by mix-matching two others, through unholy means. Both should be somewhat parallel and lay wide on the x axis.
    
    Args:
        c1: [[x0, y0], [x1, y1], ... , [xm, ym]] the matrix containing all the point coordinates of curve 1
        c2: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2
        x1: the x coord of the cut of the first curve
        x2: the x coord of the cut of the second curve
        x0: the x coord of the trim off of both curves
        n: the number of points the generated curve will have
        w: the weights of the 4 points of the bezier curve used in the factor frame

    Returns:
        [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the generated curve

    """
    # Flip if needed
    if np.linalg.norm(c1[-1] - c2[0]) > np.linalg.norm(c1[0] - c2[0]):
        c2 = np.flipud(c2)
    # Generate factor frame and factor curve
    factorframe = [[x1,1], [x0,1], [x0,0], [x2,0]]
    factorcurve = gmt.bezier(factorframe, w)(np.linspace(0,1, n+2))
    argunsort1 = np.argsort(np.flipud(np.argsort(factorcurve[:,0])))
    argunsort2 = np.argsort(np.argsort(factorcurve[:,0]))
    axis = np.zeros(len(factorcurve), dtype=int)
    mcp = gmt.cbs_interp(np.vstack((c1, c2)), factorcurve[:,0], axis)
    c1p = mcp[0:int(len(mcp)/2)]
    c2p = mcp[int(len(mcp)/2):]
    c1p = c1p[argunsort1]
    c2p = c2p[argunsort2]
    mc = np.transpose([factorcurve[:,0], c1p[:,1] * factorcurve[:,1] + (1 - factorcurve[:,1]) * c2p[:,1]])
    return mc
