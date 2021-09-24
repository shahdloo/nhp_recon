import numpy as np
from helpers import estimate_offres, rev_operator
from utils import pnorm, normalise, fftnd, ifftnd, grappaop3d, grappa, splitslicegrappa

from matplotlib import pyplot as plt

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

# populating the proxy 3d k-space
refscan3d_sl = np.zeros_like(calib)
for ss in slices:
    refscan3d_sl[..., ss, :] = calib[..., ss, :]
refscan3d_sl = fftnd(ifftnd(refscan3d_sl, dims=(0, 1)), dims=(0, 1, 2))
ops = grappaop3d(refscan3d_sl, coil_axis=-1, lamda=0.001)

im_ref = normalise(pnorm(ifftnd(np.nansum(SB[:, :, slices, ...], axis=2)), coil_axis=-1))
data_corred = np.zeros_like(data)
for frm in range(nt):
    # estimating dynamic off-resonance model parameters
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
