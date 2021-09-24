import numpy as np
from lmfit import Parameters, Minimizer
from utils import grappa, splitslicegrappa, pnorm, normalise, fftnd, ifftnd, mpow, grappaop3d


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
    params.add('a', 0, vary=True)
    params.add('b', 0, vary=True)
    params.add('a0', 0, vary=True)
    params.add('b0', 0, vary=True)
    params.add('c', 0, vary=True)
    params.add('c0', 0, vary=True)

    fitter = Minimizer(residual, params,
                       fcn_args=(nav_0, nav_n, operator, calib_nline))
    out = fitter.minimize(params=params,
                          method=solver, **kwargs)

    # find the intermediate image
    im_interm = ifftnd(rev_operator(Gx=operator[0], Gy=operator[1], Gz=operator[2], data_in=data_in,
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

    b_y = out.params['b'].value
    b_y0 = out.params['b0'].value

    b_x = out.params['a'].value
    b_x0 = out.params['a0'].value

    b_z = out.params['c'].value
    b_z0 = out.params['c0'].value

    return b_x, b_y, b_z, b_x0, b_y0, b_z0