import numpy as np
from lmfit import Parameters, Minimizer
from utils import grappa, splitslicegrappa, pnorm, normalise, fftnd, ifftnd, mpow, grappaop3d

from matplotlib import pyplot as plt

def rev_operator(Gx, Gy, Gz, data_in, params):
    a = params['a']
    b = params['b']
    c = params['c']
    a0 = params['a0']
    b0 = params['b0']
    c0 = params['c0']

    data_out = np.zeros_like(data_in)
    mltpls = [l for l in range(-int(data_in.shape[1]/2), int(np.ceil(data_in.shape[1]/2)))]

    for ln in range(len(mltpls)):
        data_out[:, ln:ln + 1, ...] = np.dot(data_in[:, ln:ln + 1, ...],
                                             np.dot(np.dot(mpow(Gx, a0 + a * mltpls[ln]),
                                                           mpow(Gy, b0 + b * mltpls[ln])),
                                                    mpow(Gz, c0 + c * mltpls[ln])))
    return data_out

def residual(params, nav_0, nav_n, operator, calib_nline=24):
    Gx = operator[0]
    Gy = operator[1]
    Gz = operator[2]

    nx = nav_0.shape[0]
    cnt = int(nx / 2)
    nav_m = rev_operator(Gx=Gx, Gy=Gy, Gz=Gz, data_in=nav_n, params={'a': params['a'],
                                         'b': params['b'],
                                         'c': params['c'],
                                         'a0': params['a0'],
                                         'b0': params['b0'],
                                         'c0': params['c0']
                                         })
    res = nav_m - nav_0

    res_lim = np.ravel(res[cnt - int(calib_nline / 2):cnt + int(calib_nline / 2), :, :])

    return np.nan_to_num(np.hstack((np.real(res_lim), np.imag(res_lim))))


def residual_im(params, im_ref=None, im_in=None):
    ny = int(im_in.shape[1])
    mltpls = np.array([l for l in range(-int(ny / 2), int(ny / 2))])
    im_xfm = np.nan_to_num(
        im_in * np.exp((params['b0']) * 1j * 2 * np.pi * mltpls / ny)[np.newaxis, :,
               np.newaxis])

    im_0_xfm = np.nan_to_num(np.ravel(normalise(pnorm(ifftnd(im_xfm), coil_axis=-1))))
    im_e = np.nan_to_num(np.corrcoef(im_ref.ravel()[im_ref.ravel() > 0.1], im_0_xfm[im_ref.ravel() > 0.1])[0, 1])
    return np.nan_to_num(1 - im_e)

def estimate_offres(nav_0,
                    nav_n,
                    operator=None,
                    solver='nelder',
                    calib_nline=24,
                    im_ref=None, data_in=None,
                    **kwargs):

    params = Parameters()
    params.add('a', 0, vary=True)#, max=1/nav_0.shape[0]/2, min=-1/nav_0.shape[0]/2)
    params.add('b', 0, vary=True)#, max=1/nav_0.shape[0]/2, min=-1/nav_0.shape[0]/2)
    params.add('a0', 0, vary=True)#, max=1/nav_0.shape[0]/2, min=-1/nav_0.shape[0]/2)
    params.add('b0', 0, vary=True)#, max=1/nav_0.shape[0]/2, min=-1/nav_0.shape[0]/2)
    if len(operator) > 2:
        params.add('c', 0, vary=True)#, max=0.5/nav_0.shape[0]/2, min=-0.5/nav_0.shape[0]/2)
        params.add('c0', 0, vary=True)#, max=0.5, min=-0.5)

    fitter = Minimizer(residual, params,
                       fcn_args=(nav_0, nav_n, operator, calib_nline))
    out = fitter.minimize(params=params,
                          method=solver, **kwargs)

    # find the intermediate image
    im_interm = ifftnd(rev_operator(Gx=ops[0], Gy=ops[1], Gz=ops[2], data_in=data_in,
                 params={'a': out.params['a'], 'b': out.params['b'], 'c': out.params['c'],
                         'a0': out.params['a0'], 'b0': out.params['b0'], 'c0': out.params['c0']}))

    # then refine b0 coefficient via brute-force
    params['a'].vary = False
    params['a'].value = out.params['a'].value.copy()
    params['a0'].vary = False
    params['a0'].value = out.params['a0'].value.copy()
    params['b'].vary = False
    params['b'].value = out.params['b'].value.copy()
    params['c0'].vary = False
    params['c0'].value = out.params['c0'].value.copy()
    params['c'].vary = False
    params['c'].value = out.params['c'].value.copy()
    params['b0'].vary = True
    params['b0'].min = out.params['b0'].value-0.2
    params['b0'].max = out.params['b0'].value+0.2
    params['b0'].brute_step = 0.02
    fitter = Minimizer(residual_im, params,
                       fcn_args=(im_ref, im_interm))
    out = fitter.minimize(params=params,
                             method='brute', **kwargs)

    b_y = out.params['b'].value# * nav_0.shape[0]/2
    b_y0 = out.params['b0'].value

    b_x = out.params['a'].value# * nav_0.shape[0]/2
    b_x0 = out.params['a0'].value

    b_z = out.params['c'].value
    b_z0 = out.params['c0'].value

    return b_x, b_y, b_z, b_x0, b_y0, b_z0

# load the data
f = np.load(open('data.npy', 'rb'), allow_pickle=True).item()
SB = f['SB']            # single-band data
data = f['data']        # multiband data with dynamic off-resonance
navs = f['navs']        # 3-line navigator data across the scan
ref_nav = f['ref_nav']  # 3-line navigator at the reference frame
calib = f['calib']      # calibration data used in PI reconstruction
slices = [0, 18]        # the excited SMS slice group
nx, ny, nc, nt = data.shape
nz = calib.shape[2]

#populating the proxy 3d k-space
refscan3d_sl = np.zeros_like(calib)
for ss in slices:
    refscan3d_sl[..., ss, :] = calib[..., ss, :]
refscan3d_sl = fftnd(ifftnd(refscan3d_sl, dims=(0, 1)), dims=(0, 1, 2))
ops = grappaop3d(refscan3d_sl, coil_axis=-1, lamda=0.001)

im_ref = normalise(pnorm(ifftnd(np.nansum(SB[:, :, slices, ...], axis=2)), coil_axis=-1))
data_corred = np.zeros_like(data)
for frm in range(nt):
    #estimating dynamic off-resonance model parameters
    pars = estimate_offres(ref_nav,
                      navs[..., 0, :, frm],
                           operator=ops,
                           im_ref=im_ref,
                           data_in=data[..., frm])
    # correcting the data
    data_corred[..., frm] = rev_operator(Gx=ops[0], Gy=ops[1], Gz=ops[2], data_in=data[..., frm], params={'a': pars[0], 'b': pars[1], 'c': pars[2],
                                                                    'a0': pars[3], 'b0': pars[4], 'c0': pars[5]})

# SMS reconstruction
rec_data = np.zeros((nx, ny, nc, 2, nt), dtype=np.cdouble)
rec_data_corr = np.zeros((nx, ny, nc, 2, nt), dtype=np.cdouble)
calib[..., int(nz/2):, :] = fftnd(np.roll(ifftnd(calib[..., int(nz/2):, :]), int(calib.shape[1]/4), 1))
for frm in range(nt):
    rec_data[..., frm] = np.squeeze(splitslicegrappa(data[..., frm][..., np.newaxis],
                                                                 np.moveaxis(SB[..., slices, :], -1, -2),
                                                                                 coil_axis=2,
                                                                                 slice_axis=-1))
    for sl in range(2):
        rec_data[..., sl, frm] = grappa(rec_data[..., sl, frm],
               calib[..., slices[sl], :], coil_axis=-1)
    rec_data[..., 1, frm] = fftnd(np.roll(ifftnd(rec_data[..., 1, frm]), -int(ny/4), 1))

for frm in range(nt):
    rec_data_corr[..., frm] = np.squeeze(splitslicegrappa(data_corred[..., frm][..., np.newaxis],
                                                                 np.moveaxis(SB[..., slices, :], -1, -2),
                                                                                 coil_axis=2,
                                                                                 slice_axis=-1))
    for sl in range(2):
        rec_data_corr[..., sl, frm] = grappa(rec_data_corr[..., sl, frm],
               calib[..., slices[sl], :], coil_axis=-1)
    rec_data_corr[..., 1, frm] = fftnd(np.roll(ifftnd(rec_data_corr[..., 1, frm]), -int(ny/4), 1))

rec_SB = np.zeros_like(SB)
for sl in range(2):
    rec_SB[..., slices[sl], :] = grappa(SB[..., slices[sl], :],
           calib[..., slices[sl], :], coil_axis=-1)
rec_SB[..., int(nz/2):, :] = fftnd(np.roll(ifftnd(rec_SB[..., int(nz/2):, :]), -int(ny/4), 1))
outline = np.rot90(normalise(pnorm(ifftnd(rec_SB[:, :, slices[1], :]), coil_axis=-1)))

# visualise the result for a representative frame
corred_disp = np.rot90(normalise(pnorm(ifftnd(rec_data_corr[..., 1, 2]), coil_axis=-1)))
corred_disp[:int(nx/4), :] = corred_disp[:int(nx/4), :]*5
corred_disp[int(np.ceil(3*nx/4)):, :] = corred_disp[int(np.ceil(3*nx/4)):, :]*5

uncorred_disp = np.rot90(normalise(pnorm(ifftnd(rec_data[..., 1, 2]), coil_axis=-1)))
uncorred_disp[:int(nx/4), :] = uncorred_disp[:int(nx/4), :]*5
uncorred_disp[int(np.ceil(3*nx/4)):, :] = uncorred_disp[int(np.ceil(3*nx/4)):, :]*5

fig, axes = plt.subplots(figsize=(8,4), nrows=1, ncols=2)
axes[0].imshow(uncorred_disp, vmin=0, vmax=1, cmap='gray')
axes[0].contour(outline, colors='r', levels=[0.1], alpha=0.5, linewidths=2)
axes[0].set_title('uncorrected')
axes[0].axis('off')
axes[1].imshow(corred_disp, vmin=0, vmax=1, cmap='gray')
axes[1].contour(outline, colors='r', levels=[0.1], alpha=0.5, linewidths=2)
axes[1].set_title('corrected')
axes[1].axis('off')
plt.show()


