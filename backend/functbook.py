#--------- READ ME----------------------------------------------------------------
# A few function generating functions. One should refrain from editing
# already existing ones, but feel free to add new ones and use them.
#---------------------------------------------------------------------------------

import numpy as np
import geometrics as gmt
from typing import Callable, Union

# Function builders
def gen_linterp(y: Union[list,tuple,np.ndarray], x: Union[list,tuple,np.ndarray] = None) -> Callable[[np.ndarray],np.ndarray]:
    """
    Generate a function by interpolating points linearly.

    Args:
        y: [y0, y1, ..., yn] vector containing all the y values
        x: [x0, x1, ..., xn] vector containing all the x values, if None the x values will be evenly spaced across [-1,1]

    Returns:
        function of x

    """
    y = np.array(y)
    if x == None:
        x = np.linspace(-1,1, np.shape(y)[0])
    funct = lambda xs: np.interp(xs, x, y)
    return funct


def gen_poly(y: Union[list,tuple,np.ndarray], x: Union[list,tuple,np.ndarray] = None, deg: int = None) -> Callable[[np.ndarray],np.ndarray]:
    """
    Generate a function by fitting a polynomial onto points.

    Args:
        y: [y0, y1, ..., yn] vector containing all the y values
        x: [x0, x1, ..., xn] vector containing all the x values, if None the x values will be evenly spaced across [-1,1]
        deg: The degree of the polynomial, if None the greatest possible degree is picked

    Returns:
        function of x

    """
    y = np.array(y)
    if x == None:
        x = np.linspace(-1,1, np.shape(y)[0])
    if deg == None:
        deg = np.shape(y)[0]-1
    poly = np.polyfit(x,y,deg)
    funct = lambda xs: np.polyval(poly, xs)
    return funct

    """
    Generate a function of a fraction. The fraction is as shown:  c / (b + (a + xfunct(xs))*ys)**p

    Args:
        xfunct (function): function of x
        a (float): denominator multiplier
        b (float): denominator constant
        c (float): fraction numerator
        p (float): denominator power

    Returns:
        f (function): function of y,x

    """
    funct = lambda y,x : c / (b + (a + xfunct(x))*y)**p
    return funct


def gen_ray_crit_func(angtol: float, maxdist: float, mindist: float = 0) -> Callable[[np.ndarray, np.ndarray, np.ndarray],bool]:
    """
    Generate a basic ray trace criterion function for opposing_faces functions.

    Arg:
        angtol: the maximum difference of the angle from vertical, that ecaluates the criterion function
        maxdist: the maximum between ray casting point and traced point, that evaluates the criterion function
        mindist: the minimum between ray casting point and traced point, that evaluates the criterion function
    
    Returns:
        criterion function
    
    """
    def crit_func(tcv, p, ray):
        # angle criterion
        abool = abs(abs(gmt.vectorangle(tcv[1] - tcv[0], ray[1] - ray[0])) - np.pi/2) <= angtol
        # distance criterion
        dbool = mindist <= np.linalg.norm(ray[0] - p) <= maxdist
        return abool and dbool
    return crit_func


# Function book

