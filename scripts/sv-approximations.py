from pylab import *

from scipy import *
from scipy import signal
from scipy import misc
from scipy import ndimage
from scipy import interpolate

import matplotlib
import matplotlib.patches
from matplotlib.collections import PatchCollection

from tables import open_file

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams["text.usetex"] = True

# def approx_sv(image):
#     print image.shape
#     orig_shape = image.shape
#     h = 1./orig_shape[0]

#     # fft_img = signal.fft2(image)
#     # fft_conv = signal.ifft2(fft_img * fft_img[::-1,::-1])

#     image = 1-image

#     fft_conv = signal.fftconvolve(image, image[::-1,::-1])
#     fft_conv /= fft_conv.max()

#     cx = fft_conv.shape[0]/2
#     cy = fft_conv.shape[1]/2

#     px = 3

#     cp_val = fft_conv[cx, cy]
#     print "Center stupid check!", cp_val

#     imshow(fft_conv)
#     show()

#     hd1 = fft_conv[cx+px, cy] 
#     hd2 = fft_conv[cx-px, cy]
#     horiz_drop = (hd1+hd2)/2.

#     vd1 = fft_conv[cx, cy+px] 
#     vd2 = fft_conv[cx, cy-px]
#     verti_drop = (vd1+vd2)/2

#     dd1 = fft_conv[cx-px, cy-px] 
#     dd2 = fft_conv[cx+px, cy+px]

#     dd3 = fft_conv[cx+px, cy-px] 
#     dd4 = fft_conv[cx-px, cy+px]

#     diag_mean = (dd1+dd2+dd3+dd4)/4.

#     da_dr_diag = (1. - diag_mean ) / (0 - sqrt(2)*px)
#     da_dr_hori = (1. - horiz_drop) / (0 - px)
#     da_dr_vert = (1. - verti_drop) / (0 - px)

#     print da_dr_hori, da_dr_vert, da_dr_diag
    
#     da_dr_mean = (da_dr_diag*4 + da_dr_hori*2 +da_dr_vert*2) / 8.
    
#     x = linspace(0,50,100)
#     y = (da_dr_mean * x) + 1

#     x += cx
#     # figure()
#     # imshow(fft_conv)
#     # colorbar()
#     plot(fft_conv[cx,:])
#     plot(fft_conv[:,cy])
#     plot(x,y)

#     return -da_dr_mean / h

def erode_dilate_approx(image):
    h = 1. / image.shape[0]
    
    ero = ndimage.morphology.binary_erosion(image)
    dia = ndimage.morphology.binary_dilation(image)

    ero_vol = ero.sum()
    img_vol = image.sum()
    dia_vol = dia.sum()

    dv_dn1 = (img_vol - ero_vol) * h
    dv_dn2 = (dia_vol - img_vol) * h

    return (dv_dn1 + dv_dn2) / 2

def path_length(path): 
    return sum(map(linalg.norm, path.vertices - roll(path.vertices, 1, axis=0)))

def contour_sa_approx(image):
    f1 = figure()
    c = contour(image, levels=[0.9999999999])
    dist = sum(map(path_length, c.collections[0].get_paths()))
    clf()
    close(f1)
    return dist


def autocorrelation(f):
    '''Return the autocorrelation coefficient of a 2-D array

    '''

    fft_f = fft2(f)
    # 7.15 Erdmann dissertation:
    C_u = (real(ifft2(fft_f * conjugate(fft_f))) / f.size) - f.mean()**2
    # Erdmann dissertation 7.2 and unnumbered equation above 7.2
    rho_u = C_u / C_u[0,0] 

    return fftshift(rho_u)

def two_point_correlation(f):
    '''Return the two-point correlation S2(dr) = < f(x) * f(x+dr) >

    '''
    fft_f = fft2(f)
    S_2 = (real(ifft2(fft_f * conjugate(fft_f))) / f.size)
    return fftshift(S_2)

def approx_sv(image):
    orig_shape = image.shape
    h = 1./orig_shape[0]

    # Mean value
    phi = image.mean()
    # Zero pad
    z = zeros_like(image)
    image = r_[z, image, z]
    zzz = r_[z,z,z]
    image = c_[zzz, image, zzz]
        
    fft_conv = two_point_correlation(image)

    # Center point
    cx = fft_conv.shape[0]/2
    cy = fft_conv.shape[1]/2

    # cp_val = fft_conv[cx, cy]
    v_here = fft_conv[cx,cy]
    
    px = 1.
    s_n = (fft_conv[cx, cy+px] - v_here) / px
    s_s = (fft_conv[cx, cy-px] - v_here) / px
    s_e = (fft_conv[cx+px, cy] - v_here) / px
    s_w = (fft_conv[cx-px, cy] - v_here) / px
    mean_slope = (s_n + s_s + s_e + s_w) / 4.
    mean_slope *= (fft_conv.shape[0] / 3.0)

    return -12 * mean_slope

real_sv = []
appr1_sv = []
appr2_sv = []
appr3_sv = []

radaii  = linspace(0.01,0.49,100)
for rad in radaii:
    
    print(rad)
    elements = 100
    h = 1/elements
    x, y = mgrid[-0.5:0.5:elements*1j,-0.5:0.5:elements*1j]
    s = 1.*(sqrt(x**2+y**2) <= rad)

    sa = (2.*pi*rad)
    fl = (1.-(pi*rad**2))
    rl = sa / fl

    est1 = 2 * approx_sv(s) / fl
    est2 = contour_sa_approx(s)  / (fl * elements)
    est3 = erode_dilate_approx(s) / fl

    print(f"Real: {rl}")
    print(f"Appr1: {est1}")
    print(f"Appr2: {est2}")
    print(f"Appr3: {est3}")

    real_sv.append(rl)
    appr1_sv.append(est1)
    appr2_sv.append(est2)
    appr3_sv.append(est3)

real_sv = array(real_sv)
appr_sv1 = array(appr1_sv)
appr_sv2 = array(appr2_sv)
appr_sv3 = array(appr3_sv)

fl = 1. - (pi * radaii**2)

figure()
# title("Comparison of Methods to Estimate $S_v$")
plot(fl, real_sv, "black",alpha=0.7, label="Actual", linewidth=2)
plot(fl, appr1_sv, "r^",  alpha=0.7, label="Autocorrelation approximation")
plot(fl, appr2_sv, "bo",  alpha=0.7, label="Line-integral approximation")
plot(fl, appr3_sv, "gs",  alpha=0.7, label="Erosion-dilation approximation")

legend(loc=1)

xlabel("$f_l$")
ylabel("$S_v$")

savefig("sa-approxs.pdf")

show()

