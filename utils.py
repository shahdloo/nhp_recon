import numpy as np
import scipy as sp
from skimage.util import view_as_windows
from tempfile import NamedTemporaryFile as NTF

fftnd = lambda x, dims=(0, 1): np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=dims), axes=dims, norm='ortho'),
                                               axes=dims)
ifftnd = lambda x, dims=(0, 1): np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=dims), axes=dims, norm='ortho'),
                                                axes=dims)
normalise = lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def pnorm(img, p=2, coil_axis=-1, pair=False, csm=None):
    '''
    performs p-norm coil combination
    '''
    if pair:
        try:
            return np.float_power(np.sum(np.abs(img) ** p, axis=coil_axis), 1 / p), np.angle(
                np.sum(img * np.conj(csm)[..., np.newaxis], axis=coil_axis))
        except:
            csm = np.moveaxis(csm, -2, -1)
            return np.float_power(np.sum(np.abs(img) ** p, axis=coil_axis), 1 / p), np.angle(
                np.sum(img * np.conj(csm)[..., np.newaxis], axis=coil_axis))
    else:
        return np.float_power(np.sum(np.abs(img) ** p, axis=coil_axis), 1 / p)


def mpow(A, p, lamda=0.0):
    '''
    the matrix fractional power operator shipped with scipy does not seem to work fine for negative powers, hence, implemented.
    '''
    if p >= 0:
        out = sp.linalg.fractional_matrix_power(A, p)
    else:
        d, V = np.linalg.eig(A)
        out = np.dot((V * (d ** p)[np.newaxis, :]),
                     np.dot(np.linalg.pinv(np.conj(V).T.dot(V) + lamda * np.identity(V.shape[0])), np.conj(V).T))
    return out

def slicegrappa(
        kspace, calib, kernel_size=(5, 5), prior='sim', coil_axis=-2,
        time_axis=-1, slice_axis=-1, lamda=0.01, split=False):
    '''(Split)-Slice-GRAPPA for SMS reconstruction.

    Parameters
    ----------
    kspace : array_like
        Time frames of sum of k-space coil measurements for
        multiple slices.
    calib : array_like
        Single slice measurements for each slice present in kspace.
        Should be the same dimensions.
    kernel_size : tuple, optional
        Size of the GRAPPA kernel: (kx, ky).
    prior : { 'sim', 'kspace' }, optional
        How to construct GRAPPA sources.  GRAPPA weights are found by
        solving the least squares problem T = S W, where T are the
        targets (calib), S are the sources, and W are the weights.
        The possible options are:

            - 'sim': simulate SMS acquisition from calibration data,
              i.e., sources S = sum(calib, axis=slice_axis).  This
              presupposes that the spatial locations of the slices in
              the calibration data are the same as in the overlapped
              kspace data.  This is similar to how the k-t BLAST
              Wiener filter is constructed (see equation 1 in [2]_).
            - 'kspace': uses the first time frame of the overlapped
              data as sources, i.e., S = kspace[1st time frame].

        This option is not used for Split-Slice-GRAPPA.

    coil_axis : int, optional
        Dimension that holds the coil data.
    time_axis : int, optional
        Dimension of kspace that holds the time data.
    slice_axis : int, optional
        Dimension of calib that holds the slice information.
    lamda : float, optional
        Tikhonov regularization for the kernel calibration.
    split : bool, optional
        Uses Split-Slice-GRAPPA kernel training method.

    Returns
    -------
    res : array_like
        Reconstructed slices for each time frame.  res will always
        return the data in fixed order or shape:
        (nx, ny, num_coils, num_time_frames, num_slices).

    Raises
    ------
    NotImplementedError
        When "prior" is an invalid option.

    Notes
    -----
    This function implements both the Slice-GRAPPA algorithm as
    described in [1]_ and the Split-Slice-GRAPPA algorithm as first
    described in [3]_.

    References
    ----------
    .. [1] Setsompop, Kawin, et al. "Blipped‐controlled aliasing in
           parallel imaging for simultaneous multislice echo planar
           imaging with reduced g‐factor penalty." Magnetic resonance
           in medicine 67.5 (2012): 1210-1224.
    .. [2] Sigfridsson, Andreas, et al. "Improving temporal fidelity
           in k-t BLAST MRI reconstruction." International Conference
           on Medical Image Computing and Computer-Assisted
           Intervention. Springer, Berlin, Heidelberg, 2007.
    .. [3] Cauley, Stephen F., et al. "Interslice leakage artifact
           reduction technique for simultaneous multislice
           acquisitions." Magnetic resonance in medicine 72.1 (2014):
           93-102.
    '''

    # Make sure we know how to construct the sources:
    if prior not in ['sim', 'kspace']:
        raise NotImplementedError("Unknown 'prior' value: %s" % prior)

    # Put the axes where we expect them
    kspace = np.moveaxis(kspace, (coil_axis, time_axis), (-2, -1))
    calib = np.moveaxis(calib, (coil_axis, slice_axis), (-2, -1))
    nx, ny, nc, nt = kspace.shape[:]
    kx, ky = kernel_size[:]
    kx2, ky2 = int(kx / 2), int(ky / 2)
    _cx, _cy, nc, cs = calib.shape[:]

    # Pad kspace data
    kspace = np.pad(  # pylint: disable=E1102
        kspace, ((kx2, kx2), (ky2, ky2), (0, 0), (0, 0)),
        mode='constant')
    calib = np.pad(  # pylint: disable=E1102
        calib, ((kx2, kx2), (ky2, ky2), (0, 0), (0, 0)),
        mode='constant')

    # Figure out how to construct the sources (only relevant for
    # Slice-GRAPPA, not Split-Slice-GRAPPA):
    if not split:
        if prior == 'sim':
            # Source data from SMS simulated calibration data.  This
            # is constructing the "prior" like k-t BLAST does, using
            # the calibration data to form the aliased/overlapped
            # images.  This requires the single-band images to be in
            # the same spatial locations as the SMS data.
            S = view_as_windows(np.sum(
                calib, axis=-1), (kx, ky, nc)).reshape((-1, kx * ky * nc))
        elif prior == 'kspace':
            # Source data from the first time frame of the
            # kspace data.
            S = view_as_windows(np.ascontiguousarray(
                kspace[..., 0]), (kx, ky, nc)).reshape((-1, kx * ky * nc))

    # Train a kernel for each target slice -- use Split-Slice-GRAPPA
    # if the user asked for it, else use Slice-GRAPPA
    W = np.zeros((cs, kx * ky * nc, nc), dtype=calib.dtype)
    for sl in range(cs):

        # Train GRAPPA kernel for the current slice
        T = calib[kx2:-kx2, ky2:-ky2, :, sl].reshape((-1, nc))
        if not split:
            # Regular old Slice-GRAPPA
            ShS = S.conj().T @ S
            ShT = S.conj().T @ T
            lamda0 = lamda * np.linalg.norm(ShS) / ShS.shape[0]
            W[sl, ...] = np.linalg.solve(
                ShS + lamda0 * np.eye(ShS.shape[0]), ShT)
        else:
            # Split-Slice-GRAPPA for all your slice leakage needs!
            # Equation (7) from ref. [3]:
            MhM = np.zeros((kx * ky * nc,) * 2, dtype=calib.dtype)

            # This might be inefficient as we're getting patches
            # for every single slice for every slice kernel, but it
            # does run pretty fast and it's pretty memory intensive
            # to get all patches for all slices outside of the loop.
            # Maybe use temporary files?
            for jj in range(cs):
                calib0 = view_as_windows(np.ascontiguousarray(
                    calib[..., jj]), (kx, ky, nc)).reshape(
                    (-1, kx * ky * nc))
                MhM += calib0.conj().T @ calib0

                # Find and save the target calibration slice, Mz:
                if jj == sl:
                    Mz = calib0

            MhT = Mz.conj().T @ T
            lamda0 = lamda * np.linalg.norm(MhM) / MhM.shape[0]
            W[sl, ...] = np.linalg.solve(
                MhM + lamda0 * np.eye(MhM.shape[0]), MhT)

    # Now pull apart slices for each time frame
    res = np.zeros((nx, ny, nc, nt, cs), dtype=kspace.dtype)
    S = view_as_windows(
        kspace, (kx, ky, nc, nt)).reshape((-1, kx * ky * nc, nt))
    for tt in range(nt):  # , leave=False, desc='Slice-GRAPPA'):
        for sl in range(cs):
            res[..., tt, sl] = (
                    S[..., tt] @ W[sl, ...]).reshape((nx, ny, nc))

    # Return results in fixed order: (nx, ny, nc, nt, cs)
    return res


def splitslicegrappa(*args, **kwargs):
    '''Split-Slice-GRAPPA.

    Notes
    -----
    This is an alias for pygrappa.slicegrappa(split=True).
    See pygrappa.slicegrappa() for more information.
    '''

    # Make sure that the 'split' argument is set to True
    if 'split' not in kwargs or not kwargs['split']:
        kwargs['split'] = True
    return slicegrappa(*args, **kwargs)

def grappa(
        kspace, calib, kernel_size=(5, 5), coil_axis=-1, lamda=0.01,
        memmap=False, memmap_filename='out.memmap', silent=True):
    '''GeneRalized Autocalibrating Partially Parallel Acquisitions.

    Parameters
    ----------
    kspace : array_like
        2D multi-coil k-space data to reconstruct from.  Make sure
        that the missing entries have exact zeros in them.
    calib : array_like
        Calibration data (fully sampled k-space).
    kernel_size : tuple, optional
        Size of the 2D GRAPPA kernel (kx, ky).
    coil_axis : int, optional
        Dimension holding coil data.  The other two dimensions should
        be image size: (sx, sy).
    lamda : float, optional
        Tikhonov regularization for the kernel calibration.
    memmap : bool, optional
        Store data in Numpy memmaps.  Use when datasets are too large
        to store in memory.
    memmap_filename : str, optional
        Name of memmap to store results in.  File is only saved if
        memmap=True.
    silent : bool, optional
        Suppress messages to user.

    Returns
    -------
    res : array_like
        k-space data where missing entries have been filled in.

    Notes
    -----
    Based on implementation of the GRAPPA algorithm [1]_ for 2D
    images.

    If memmap=True, the results will be written to memmap_filename
    and nothing is returned from the function.

    References
    ----------
    .. [1] Griswold, Mark A., et al. "Generalized autocalibrating
           partially parallel acquisitions (GRAPPA)." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           47.6 (2002): 1202-1210.
    '''

    # Remember what shape the final reconstruction should be
    fin_shape = kspace.shape[:]

    # Put the coil dimension at the end
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)

    # Quit early if there are no holes
    if np.sum((np.abs(kspace[..., 0]) == 0).flatten()) == 0:
        return np.moveaxis(kspace, -1, coil_axis)

    # Get shape of kernel
    kx, ky = kernel_size[:]
    kx2, ky2 = int(kx / 2), int(ky / 2)
    nc = calib.shape[-1]

    # When we apply weights, we need to select a window of data the
    # size of the kernel.  If the kernel size is odd, the window will
    # be symmetric about the target.  If it's even, then we have to
    # decide where the window lies in relation to the target.  Let's
    # arbitrarily decide that it will be right-sided, so we'll need
    # adjustment factors used as follows:
    #     S = kspace[xx-kx2:xx+kx2+adjx, yy-ky2:yy+ky2+adjy, :]
    # Where:
    #     xx, yy : location of target
    adjx = np.mod(kx, 2)
    adjy = np.mod(ky, 2)

    # Pad kspace data
    kspace = np.pad(  # pylint: disable=E1102
        kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')
    calib = np.pad(  # pylint: disable=E1102
        calib, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')

    # Notice that all coils have same sampling pattern, so choose
    # the 0th one arbitrarily for the mask
    mask = np.ascontiguousarray(np.abs(kspace[..., 0]) > 0)

    # Store windows in temporary files so we don't overwhelm memory
    with NTF() as fP, NTF() as fA, NTF() as frecon:

        # Get all overlapping patches from the mask
        P = np.memmap(fP, dtype=mask.dtype, mode='w+', shape=(
            mask.shape[0] - 2 * kx2, mask.shape[1] - 2 * ky2, 1, kx, ky))
        P = view_as_windows(mask, (kx, ky))
        Psh = P.shape[:]  # save shape for unflattening indices later
        P = P.reshape((-1, kx, ky))

        # Find the unique patches and associate them with indices
        P, iidx = np.unique(P, return_inverse=True, axis=0)

        # Filter out geometries that don't have a hole at the center.
        # These are all the kernel geometries we actually need to
        # compute weights for.
        validP = np.argwhere(~P[:, kx2, ky2]).squeeze()

        # We also want to ignore empty patches
        invalidP = np.argwhere(np.all(P == 0, axis=(1, 2)))
        validP = np.setdiff1d(validP, invalidP, assume_unique=True)

        # Make sure validP is iterable
        validP = np.atleast_1d(validP)

        # Give P back its coil dimension
        P = np.tile(P[..., None], (1, 1, 1, nc))


        # Get all overlapping patches of ACS
        try:
            A = np.memmap(fA, dtype=calib.dtype, mode='w+', shape=(
                calib.shape[0] - 2 * kx, calib.shape[1] - 2 * ky, 1, kx, ky, nc))
            A[:] = view_as_windows(
                calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))
        except ValueError:
            A = view_as_windows(
                calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))

        # Initialize recon array
        recon = np.memmap(
            frecon, dtype=kspace.dtype, mode='w+',
            shape=kspace.shape)

        # Train weights and apply them for each valid hole we have in
        # kspace data:
        for ii in validP:
            # Get the sources by masking all patches of the ACS and
            # get targets by taking the center of each patch. Source
            # and targets will have the following sizes:
            #     S : (# samples, N possible patches in ACS)
            #     T : (# coils, N possible patches in ACS)
            # Solve the equation for the weights:
            #     WS = T
            #     WSS^H = TS^H
            #  -> W = TS^H (SS^H)^-1
            # S = A[:, P[ii, ...]].T # transpose to get correct shape
            # T = A[:, kx2, ky2, :].T
            # TSh = T @ S.conj().T
            # SSh = S @ S.conj().T
            # W = TSh @ np.linalg.pinv(SSh) # inv won't work here

            # Equivalenty, we can formulate the problem so we avoid
            # computing the inverse, use numpy.linalg.solve, and
            # Tikhonov regularization for better conditioning:
            #     SW = T
            #     S^HSW = S^HT
            #     W = (S^HS)^-1 S^HT
            #  -> W = (S^HS + lamda I)^-1 S^HT
            # Notice that this W is a transposed version of the
            # above formulation.  Need to figure out if W @ S or
            # S @ W is more efficient matrix multiplication.
            # Currently computing W @ S when applying weights.
            S = A[:, P[ii, ...]]
            T = A[:, kx2, ky2, :]
            ShS = S.conj().T @ S
            ShT = S.conj().T @ T
            lamda0 = lamda * np.linalg.norm(ShS) / ShS.shape[0]
            W = np.linalg.solve(
                ShS + lamda0 * np.eye(ShS.shape[0]), ShT).T

            # Now that we know the weights, let's apply them!  Find
            # all holes corresponding to current geometry.
            # Currently we're looping through all the points
            # associated with the current geometry.  It would be nice
            # to find a way to apply the weights to everything at
            # once.  Right now I don't know how to simultaneously
            # pull all source patches from kspace faster than a
            # for loop...

            # x, y define where top left corner is, so move to ctr,
            # also make sure they are iterable by enforcing atleast_1d
            idx = np.unravel_index(
                np.argwhere(iidx == ii), Psh[:2])
            x, y = idx[0] + kx2, idx[1] + ky2
            x = np.atleast_1d(x.squeeze())
            y = np.atleast_1d(y.squeeze())
            for xx, yy in zip(x, y):
                # Collect sources for this hole and apply weights
                S = kspace[xx - kx2:xx + kx2 + adjx, yy - ky2:yy + ky2 + adjy, :]
                S = S[P[ii, ...]]
                recon[xx, yy, :] = (W @ S[:, None]).squeeze()

        # The recon array has been zero padded, so let's crop it down
        # to size and return it either as a memmap to the correct
        # file or in memory.
        # Also fill in known data, crop, move coil axis back.
        if memmap:
            fin = np.memmap(
                memmap_filename, dtype=recon.dtype, mode='w+',
                shape=fin_shape)
            fin[:] = np.moveaxis(
                (recon + kspace)[kx2:-kx2, ky2:-ky2, :],
                -1, coil_axis)
            del fin
            return None

        return np.moveaxis(
            (recon[:] + kspace)[kx2:-kx2, ky2:-ky2, :], -1, coil_axis)

def grappaop3d(calib, coil_axis=-1, lamda=0.01):
    '''GRAPPA operator for Cartesian calibration datasets.

    Parameters
    ----------
    calib : array_like
        Calibration region data.  Usually a small portion from the
        center of kspace.
    coil_axis : int, optional
        Dimension holding coil data.
    lamda : float, optional
        Tikhonov regularization parameter.  Set to 0 for no
        regularization.

    Returns
    -------
    Gx, Gy : array_like
        GRAPPA operators for both the x and y directions.

    Notes
    -----
    Produces the unit operator described in [1]_.

    This seems to only work well when coil sensitivities are very
    well separated/distinct.  If coil sensitivities are similar,
    operators perform poorly.

    References
    ----------
    .. [1] Griswold, Mark A., et al. "Parallel magnetic resonance
           imaging using the GRAPPA operator formalism." Magnetic
           resonance in medicine 54.6 (2005): 1553-1556.
    '''

    # Coil axis in the back
    calib = np.moveaxis(calib, coil_axis, -1)
    _cx, _cy, _cz, nc = calib.shape[:]

    # We need sources (last source has no target!)
    Sx = np.reshape(calib[:-1, ...], (-1, nc))
    Sy = np.reshape(calib[:, :-1, ...], (-1, nc))
    Sz = np.reshape(calib[:, :, :-1, ...], (-1, nc))

    # And we need targets for an operator along each axis (first
    # target has no associated source!)
    Tx = np.reshape(calib[1:, ...], (-1, nc))
    Ty = np.reshape(calib[:, 1:, ...], (-1, nc))
    Tz = np.reshape(calib[:, :, 1:, :], (-1, nc))

    # Train the operators:
    Sxh = Sx.conj().T
    lamda0 = lamda * np.linalg.norm(Sxh) / Sxh.shape[0]
    Gx = np.linalg.pinv(Sxh@Sx + lamda0 * np.eye(Sxh.shape[0]))@(Sxh @ Tx)
    # Gx = np.linalg.solve(
    #     Sxh @ Sx + lamda0 * np.eye(Sxh.shape[0]), Sxh @ Tx)

    Syh = Sy.conj().T
    lamda0 = lamda * np.linalg.norm(Syh) / Syh.shape[0]
    # Gy = np.linalg.solve(
    #     Syh @ Sy + lamda0 * np.eye(Syh.shape[0]), Syh @ Ty)
    Gy = np.linalg.pinv(Syh @ Sy + lamda0 * np.eye(Syh.shape[0])) @ (Syh @ Ty)

    Szh = Sz.conj().T
    lamda0 = lamda * np.linalg.norm(Szh) / Szh.shape[0]
    # Gz = np.linalg.solve(
    #     Szh @ Sz + lamda0 * np.eye(Szh.shape[0]), Szh @ Tz)
    Gz = np.linalg.pinv(Szh @ Sz + lamda0 * np.eye(Szh.shape[0])) @ (Szh @ Tz)

    return Gx, Gy, Gz
