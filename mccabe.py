'''
============
McCabe-thiele 
============
 
 
==========================
Rectifying operating line 
==========================

 ============
 assumptions:
 ============
 1- constant molal overflow (CMO)
 2-total condenser
 

 ============
  equations:
 ============

1) y = mR * x + bR
2) mR = R / ( R + 1 )
3) bR = xD / (R + 1 )

where:
R: reflux ratio ( L / D )
xD: distillate composition (light key)


===============================
q-line (feed thermal condition) 
===============================

********************
it passes through (xF , xF) 

slope: mq = q / ( q - 1 )

intercept: bq = xF ( 1 - mq)
******************** 

 ================
  special cases: 
 ================

1) q = 1 vertical line (saturated liquid)
2) q = 0 horizontal line (saturated vapor)
3) q > 1 line steeper than vertical line (subcooled liquid)
4) q < 0 line negative slope (superheated vapor)

==========================
feed break point (x*,y*) 
==========================

*************************************
q-line intersects the rectifying line
*************************************

=============================
general (non-vertical q-line) 
=============================

mR * x* + bR = mq * x* + bq

x* = ( bq - bR ) / ( mR - mq )
y* = mR * x* + bR

========================================
vertical q-line q = 1 (saturated liquid) 
========================================

x* = xF
y* = mR * xF + bR

========================
stripping operating line  
========================

***********************************************
through the break point and the reboiler point

y= ms * x + bs

where:
ms = ( y* - xB) / ( x* - xB)
bs = y* - ms * x* 
***********************************************
===============================
Stepping off the stages (graph)
===============================

we will only model the graph 

1- stair step operations:

1) start from the top point (xD, xD)
2) step to the equilibrium curve (horizontal line till y=x)
3) step to the operating line (vertical line till y = m*x + b)

----------------------
stair step termination:
----------------------
 the termination is the bottom composition 
 as sson as we reach (xB ,y) we stop 

 
-----------
error check:
-----------

1- physically possible steps ( within [0,1] of x 
2- stage count
3- feed stage index
4- messages
    * reached bottoms
    *pinch/ near Rmin (rectifying and q-line nearly parallel)
    *“Unphysical step (x went out of [0,1])” → check specs

'''

# =============================================================================
# IMPORTS 
# =============================================================================

import numpy as np
from typing import Union, Tuple, Dict, List
from thermo_simple import yfxeq, xfyeq


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_fraction(value: float, name: str) -> float:
    """Validate that a value is a number between 0 and 1."""
    if not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a number")
    if not (0 <= value <= 1):
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)

def validate_positive(value: float, name: str) -> float:
    """Validate that a value is a positive number."""
    if not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a number")
    if value <0:
        raise ValueError(f"{name} must be >= 0")
    return float(value)


# =============================================================================
# GLOBAL DEFAULTS (for quick tests; main uses function args)
# =============================================================================

'''
#global values
xD=0.95
xB=0.05
xF=0.4
q=1
R=1.8
'''


# rectifying operating line
def rectifying_line(R:float ,xD: float):
    
    #type check
    if not isinstance(R, (float, int)): raise TypeError("R must be a number")
    #range check   
    if R<=0: raise ValueError("R must be >0")
    if not (0 <= xD <= 1): raise ValueError("xD must be between 0 and 1")
    
    mR = R / ( R + 1 )

    bR = xD / (R + 1 )

   #R = ( L / D )

    return mR, bR

# q-line (feed thermal condition)


def q_line(xF: float, q: float):
      
    # q-line feed
    #it passes through (xF , xF) 
    
    #========================================
    #vertical q-line q = 1 (saturated liquid) 
    #========================================
    eps=1e-12
    if abs(q-1.0) < eps:
        return {"vertical":True, "xF":xF}
    if abs(q-0.0) < eps:
        return {"horizontal":True, "xF":xF, "y" :xF}
      
        
    else :  
    
    #=============================
    # general (non-vertical q-line) 
    #=============================
        # slope:
        mq = q / ( q - 1 )
        # intercept:
        bq = xF * ( 1 - mq)

    return {"vertical":False,"horizontal":False, "mq":mq, "bq":bq}
    


def break_point(mR: float, bR: float, qline: dict):
    # if q=1
    if qline.get('vertical' ):
        xF_loc = qline['xF']
        return (xF_loc, mR * xF_loc +bR)
    #if q=0
    if qline.get('horizontal'):
        ystar = qline['y']          # = xF
        xstar = (ystar - bR) / mR
        return (xstar, ystar)
    #other values of q
    else:
         mq , bq = qline["mq"], qline["bq"]
         xstar = ( bq - bR ) / ( mR - mq )
         ystar = mR * xstar + bR
         return (xstar, ystar)
    


    

#========================
#stripping operating line  
#========================
def stripping_line(xB: float, xstar: float, ystar: float):
    #type check
    if not isinstance(xB, (float, int)): raise TypeError("xB must be a number")
    if not isinstance(xstar, (float, int)): raise TypeError("xstar must be a number")
    if not isinstance(ystar, (float, int)): raise TypeError("ystar must be a number")
    #range check   
    if not (0 <= xB <= 1): raise ValueError("xB must be between 0 and 1")
    if not (0 <= xstar <= 1): raise ValueError("xstar must be between 0 and 1")
    if not (0 <= ystar <= 1): raise ValueError("ystar must be between 0 and 1")
    
    # stripping line y=ms*x +bs
    
    ms = ( ystar - xB ) / ( xstar - xB )
    bs = ystar - ms * xstar
    
    return ms, bs


# =======================================================
#                    operations 
# =======================================================

'''
                     operating lines
--------------------------------------------------------------
* rectifying: y=mR*x + bR

* stripping:  y=ms*x + bs
 feed:	  	q-line: y=mq*x + bq
            vertical --> x = xF
            horizontal --> y = xF

--------------------------------------------------------------

 ------------------------------------------------------
 termination:
 ------------------------------------------------------ 

1) if reached bottoms:
   xB < x <= x_new ==> converge message is good

2) if not reached bottoms:
   x_new < xB (while stepping) ==> unacceptable solution then break and terminate

3) +++++++ addition +++++++++++++++++++++++++++
or maybe we should just do if (x_eq < x) & (x<x_new) --> then 
break and found solution, simply because moving vertically
means moving downwards.

'''

#==============================================
#helper function: y-value on a given straight line
#===============================================

def line_y(x: float, m: float, b: float):
    
    if not (0 <= x <= 1):
        raise ValueError("x must be between 0 and 1")
    if not isinstance(m, (float, int)):
        raise TypeError("m must be a number")
    if not isinstance(b, (float, int)):
        raise TypeError("b must be a number")
    if abs(m) < 1e-12:
        raise ValueError("m is too small")
    
    return m * x + b

def stair_stepper(xD: float, xB: float, xF: float, q: float, R: float, alpha: float,
                  max_iterations=50, tolerance=1e-12, N_max=500):
    
    
    
    '''  
                    operating lines
    --------------------------------------
    * rectifying: y=mR*x + bR (above the q-line)
    
    * stripping:  y=ms*x + bs (below the q-line)
    feed:	  	q-line: y=mq*x + bq
                vertical --> x = xF
                horizontal --> y = xF
    * the q-line is the divider line that determines which operating line 
      we will use during the stepping from both sections
    --------------------------------------
    
     * step termination and errors check:
      -----------------------------------
        1) termination: 
            if reached bottoms:
               xB < x <= x_new ==> converge message is good       
        2) error checks:
            * if not reached bottoms:
              x_new < xB (while stepping) ==> unacceptable solution then break and terminate
              
            * pinch near Rmin:
                testing that the rectifying line (above the q-line)
                and the q-line (vertical line means x = xF)
                are not almost parallel
                       
            * unphysical step: 
              raises error if we go out of range [0,1], during stepping 
              
    '''
    #validation inputs 
    xD = validate_fraction(xD, "xD")
    xB = validate_fraction(xB, "xB")
    xF = validate_fraction(xF, "xF")
    q = validate_positive(q, "q")
    R = validate_positive(R, "R")
    alpha = validate_positive(alpha, "alpha")
    if not isinstance(q, (float, int)):
        raise TypeError("q must be a number")
    
    
    # ---------------------------
    # rectifying line and q-line
    # ---------------------------

    mR, bR = rectifying_line(R, xD)
    # q-line spec
    qline = q_line(xF, q)
    # break point
    xstar, ystar = break_point(mR, bR, qline)
        
    # ---------------------------
    # Stripping line
    # ---------------------------
   
    ms, bs = stripping_line(xB, xstar, ystar)
    

    '''  
                    intials 
                    &
                    stairs-steeping algorithm 
                    '''        
    
    # vertices to plot the stairs 
    vertices = []
    #counting the number of stages 
    stage_counter = 0
    #feed stage and swtitcher (the feed position is inknown yet) it switch once from -1 to 1
    feed_stage_index= -1    
    message=""
    
    # the starting point ( xD , xD )
    current_x = xD
    current_y = xD
    #appending the current vertices 
    vertices.append((current_x, current_y))
    print(vertices)
    
    # the strating point in the rectifying section 
    section = 'rectifying'
    
    #-----------------------------------
    # near- parallel lines check (pinch)
    #-----------------------------------
    
    if not qline.get('vertical', False) and not qline.get('horizontal', False):
        
        # checking the slopes of the lines
        mq=qline['mq']
        
        if abs(mR - mq) < 1e-12:
            message="warning: rectifying and q-line nearly parallel"

    
    
    '''  
                    ittration of the stairing algorithn  
                    '''  
    
    for _ in range(N_max):
        # 1) Horizontal to equilibrium
        # get y on equilibrium for current_x: y_eq = y(x)
        # 1) Horizontal to equilibrium (keep y constant, find x on eq curve)
        try:
            x_eq = xfyeq(current_y, alpha)   # <-- use inverse: x = x(y)
        except Exception as e:
            message = f"Thermo error during stepping: {e}"
            break

        vertices.append((x_eq, current_y))

        # Check for termination: if at or below xB (within tolerance)
        if x_eq <= xB + tolerance:
            message = "Reached bottoms composition."
            break

        # 2) Vertical to the operating line at x = x_eq
        if section == 'rectifying':
            y_new_line = line_y(x_eq, mR, bR)
        else:  # stripping
            y_new_line = line_y(x_eq, ms, bs)

        # Decide if we crossed the feed (switch sections) — x-based heuristic
        if (feed_stage_index == -1) and (x_eq < xstar) and (section == 'rectifying'):
            section = 'stripping'
            feed_stage_index = stage_counter
            y_new_line = line_y(x_eq, ms, bs)


        # Append new vertex after vertical step
        vertices.append((x_eq, y_new_line))

        # Increment stage count and update current point
        stage_counter += 1
        current_x = x_eq
        current_y = y_new_line


        
        # Safety check: bounds
        if not (0 <= current_x <= 1):
            message = "Unphysical step: x went out of [0,1]. Check specs."
            break

        # Max iterations
        if stage_counter >= max_iterations:
            message = "Max iterations reached during stepping."
            break
    
    # If feed stage was never set but we did cross near xstar, set to last
    if feed_stage_index == -1:
        # best-effort: if xD > xstar, we expect at least one crossing
        if xD > xstar and xB < xstar:
            feed_stage_index = max(stage_counter - 1, 0)
    
    return {
        "vertices": vertices,
        "stage_counter": stage_counter,
        "feed_stage_index": feed_stage_index,
        "message": message
    }


def test_thermodynamics():
    from thermo_simple import yfxeq, xfyeq
    """Test the thermodynamics functions."""
    print("Testing thermodynamics functions:")
    alpha_test = 2.5
    x_test = 0.5
    y_test = yfxeq(x_test, alpha_test)
    x_back = xfyeq(y_test, alpha_test)
    print(f"alpha={alpha_test}, x={x_test} -> y={y_test:.6f}")
    print(f"invert y={y_test:.6f} -> x={x_back:.6f}")
    assert abs(x_back - x_test) < 1e-10, "Round-trip test failed"
    print("Thermodynamics test passed!\n")


def test_mccabe_thiele():
    """Test the complete McCabe-Thiele simulation."""
    print("Testing McCabe-Thiele simulation:")
    
    # Global values
    xD = 0.923
    xB = 0.071
    xF = 0.584
    q = 0
    R = 2.67
    alpha = 3.04
    
    # Run simulation
    result = stair_stepper(
        xD, xB, xF, q, R, alpha,
        max_iterations=500, tolerance=1e-10, N_max=500
    )
    
    # Extract results
    vertices = result['vertices']
    stage_counter = result['stage_counter']
    feed_stage_index = result['feed_stage_index']
    message = result['message']
    
    print(f"\n{'='*50}")
    print("SIMULATION RESULTS:")
    print(f"{'='*50}")
    print(f"vertices: Array of {len(vertices)} coordinate pairs")
    print(f"stage_counter: {stage_counter} and a partial reboiler")
    print(f"feed_stage: {feed_stage_index}")
    print(f"message: '{message}'")
    
    print(f"\nFirst 5 vertices:")
    for i in range(min(5, len(vertices))):
        print(f"  [{vertices[i][0]:.6f}, {vertices[i][1]:.6f}]")
    
    if len(vertices) > 10:
        print("  ...")
        print(f"Last 5 vertices:")
        for i in range(max(0, len(vertices)-5), len(vertices)):
            print(f"  [{vertices[i][0]:.6f}, {vertices[i][1]:.6f}]")
    
    return result


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Test thermodynamics
    test_thermodynamics()
    
    # Test complete simulation
    result = test_mccabe_thiele()
    
    print(f"\n{'='*60}")
    print("SUCCESS: McCabe-Thiele simulation completed successfully!")
    print(f"{'='*60}")