
'''
thermodynamics helper (constant relative volatility model)

in binary distillation, it is required to describe how the vapor and liquid 
compositions are related at equilibrium. (VLE relation)
 
 
 1-yfxeq
assumptions:

constant relative volatility (α) between the two components 

formula: y=(α*x)/(1+(α-1)*x)) (equilibruim curve)
- it bends upward strongly if α is larger --> better separation


2-xfyeq
inverse relation (if y known and x uknown)
 formula: x=y/(α-(α-1)*y)

'''

def yfxeq(x: float,alpha:float ):
    
    #type check
    if not isinstance(x, (float, int)):
        raise TypeError("x must be a number")
    if not isinstance(alpha, (float, int)):
        raise TypeError("alpha must be a number")
    #range check   
    if alpha<=0:
        raise ValueError("α must be >0")
    if not (0 <= x <= 1):
        raise ValueError("x must be between 0 and 1")  

    
    denominator = 1+(alpha-1)*x
    return (alpha*x)/denominator 

def xfyeq(y: float,alpha:float):
    
    #type check
    if not isinstance(y, (float, int)):
        raise TypeError("y must be a number")
    if not isinstance(alpha, (float, int)):
        raise TypeError("alpha must be a number")
    #range check    
    if alpha<=0:
        raise ValueError("α must be >0")
    eps = 1e-12
    if y < -eps or y > 1 + eps:
        raise ValueError("y must be between 0 and 1")      

    # allow tiny numerical drift, then clamp
    eps = 1e-12
    if y < -eps or y > 1 + eps:
        raise ValueError("y must be between 0 and 1")
    y = min(max(float(y), 0.0), 1.0)    
    
    denominator = alpha - (alpha - 1) * y
    
    if abs(denominator) < 1e-12:
        raise ZeroDivisionError("Denominator is very small")
        
    return y / denominator



if __name__ == "__main__":
    # Test 1: Single value
    a = 2.5
    x = 0.5
    y = yfxeq(x, a)
    x_back = xfyeq(y, a)
    print(f"alpha={a}, x={x} -> y={y:.6f}")
    print(f"invert y={y:.6f} -> x={x_back:.6f}")
    print("Single value test passed.")
    
    # Test 2: Multiple values
    print(f"\nTesting alpha = {a}")
    xs = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    for x in xs:
        y = yfxeq(x, a)
        x_back = xfyeq(y, a)
        print(f"x = {x:.2f} -> y = {y:.6f} -> back x = {x_back:.6f}")
    print("Multiple values test passed.")