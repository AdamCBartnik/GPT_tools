import numpy as np


def nicer_scale_prefix(scale, mm_cutoff=0.1):
    """
    Returns a nice factor and a SI prefix string 
    
    Example:
        scale = 2e-10
        
        f, u = nicer_scale_prefix(scale)
        
        
    """
    
    if (np.all(np.isnan(scale))):
        return 1, ''
    
    max_val = np.nanmedian(np.abs(scale))   # median is the new max
        
    if max_val < 1e-28:
        return 1, ''
        
    fudge_factor=10**(-3.0/2.0)/mm_cutoff
        
    max_power=3*np.sign(np.log10(max_val))*round(abs(np.log10(max_val*fudge_factor)/3))
    f = 10**max_power

    if (np.isnan(f)):
        f = 1.0
        
    return f, SHORT_PREFIX[f]




def nicer_array(a, mm_cutoff=0.3):
    """
    Returns a scaled array, the scaling, and a unit prefix
    
    Example: 
        nicer_array( np.array([2e-10, 3e-10]) )
    Returns:
        (array([200., 300.]), 1e-12, 'p')
    
    """
        
    if np.isscalar(a):
        x = a
    elif len(a) == 1:
        x = a[0]
    else:
        x = np.array(a)
         
    fac, prefix = nicer_scale_prefix( x, mm_cutoff=mm_cutoff )
        
    return a/fac, fac,  prefix




# Dicts for prefixes
PREFIX_FACTOR = {
    'yocto-' :1e-24,
    'zepto-' :1e-21,
    'atto-'  :1e-18,
    'femto-' :1e-15,
    'pico-'  :1e-12,
    'nano-'  :1e-9 ,
    'micro-' :1e-6,
    'milli-' :1e-3 ,
    'centi-' :1e-2 ,
    'deci-'  :1e-1,
    'deca-'  :1e+1,
    'hecto-' :1e2  ,
    'kilo-'  :1e3  ,
    'mega-'  :1e6  ,
    'giga-'  :1e9  ,
    'tera-'  :1e12 ,
    'peta-'  :1e15 ,
    'exa-'   :1e18 ,
    'zetta-' :1e21 ,
    'yotta-' :1e24
}
# Inverse
PREFIX = dict( (v,k) for k,v in PREFIX_FACTOR.items())

SHORT_PREFIX_FACTOR = {
    'y'  :1e-24,
    'z'  :1e-21,
    'a'  :1e-18,
    'f'  :1e-15,
    'p'  :1e-12,
    'n'  :1e-9 ,
    'u'  :1e-6,
    'm'  :1e-3 ,
    'c'  :1e-2 ,
    'd'  :1e-1,
    ''   : 1,
    'da' :1e+1,
    'h'  :1e2  ,
    'k'  :1e3  ,
    'M'  :1e6  ,
    'G'  :1e9  ,
    'T'  :1e12 ,
    'P'  :1e15 ,
    'E'  :1e18 ,
    'Z'  :1e21 ,
    'Y'  :1e24
}
# Inverse
SHORT_PREFIX = dict( (v,k) for k,v in SHORT_PREFIX_FACTOR.items())



