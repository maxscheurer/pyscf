#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Non-relativistic RKS analytical nuclear gradients'''

import time
import copy
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import rhf_grad
from pyscf.dft import numint, radi, gen_grid


def get_veff(ks_grad, mol=None, dm=None):
    '''Coulomb + XC functional
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad._scf.make_rdm1()
    t0 = (time.clock(), time.time())

    mf = ks_grad._scf
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)
    hyb = mf._numint.libxc.hybrid_coeff(mf.xc, spin=mol.spin)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc = get_vxc_full_response(mf._numint, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
    else:
        exc, vxc = get_vxc(mf._numint, mol, grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
    nao = vxc.shape[-1]
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if abs(hyb) < 1e-10:
        vj = ks_grad.get_j(mol, dm)
        vxc += vj
    else:
        vj, vk = ks_grad.get_jk(mol, dm)
        vxc += vj - vk * (hyb * .5)

    return lib.tag_array(vxc, exc1_grid=exc)


def grad_elec(grad_mf, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    mf = grad_mf._scf
    mol = grad_mf.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(grad_mf.stdout, grad_mf.verbose)

    h1 = grad_mf.get_hcore(mol)
    s1 = grad_mf.get_ovlp(mol)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    t0 = (time.clock(), time.time())
    log.debug('Computing Gradients of NR Hartree-Fock Coulomb repulsion')
    vhf = grad_mf.get_veff(mol, dm0)
    log.timer('gradients of 2e part', *t0)

    f1 = h1 + vhf
    dme0 = grad_mf.make_rdm1e(mo_energy, mo_coeff, mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    atom_slices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = atom_slices[ia]
# h1, s1, vhf are \nabla <i|h|j>, the nuclear gradients = -\nabla
        vrinv = grad_mf._grad_rinv(mol, ia)
        de[k] += numpy.einsum('xij,ij->x', f1[:,p0:p1], dm0[p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vrinv, dm0) * 2
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2
        if grad_mf.grid_response:
            de[k] += vhf.exc1_grid[ia]
    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        rhf_grad._write(log, mol, de, atmlst)
        if grad_mf.grid_response:
            log.debug('grids response contributions')
            rhf_grad._write(log, mol, vhf.exc1_grid[atmlst], atmlst)
            log.debug1('sum(de) %s', vhf.exc1_grid.sum(axis=0))
    return de


def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((nset,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[0], mask, 'LDA')
                vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1, verbose)[1]
                vrho = vxc[0]
                aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho)
                vmat[idm,0] += numint._dot_ao_ao(mol, ao[1], aow, mask, shls_slice, ao_loc)
                vmat[idm,1] += numint._dot_ao_ao(mol, ao[2], aow, mask, shls_slice, ao_loc)
                vmat[idm,2] += numint._dot_ao_ao(mol, ao[3], aow, mask, shls_slice, ao_loc)
                rho = vxc = vrho = aow = None
    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[:4], mask, 'GGA')
                vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1, verbose)[1]
                vrho, vsigma = vxc[:2]
                wv = numpy.empty_like(rho)
                wv[0]  = weight * vrho
                wv[1:] = rho[1:] * (weight * vsigma * 2)
                vmat[idm] += _gga_grad_sum(mol, ao, wv, mask, shls_slice, ao_loc)
                rho = vxc = vrho = vsigma = wv = None
    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    else:
        raise NotImplementedError('meta-GGA')

    exc = None
    if nset == 1:
        vmat = vmat.reshape(3,nao,nao)
    # - sign because nabla_X = -nabla_x
    return exc, -vmat

def _gga_grad_sum(mol, ao, wv, mask, shls_slice, ao_loc):
    ngrid, nao = ao[0].shape
    vmat = numpy.empty((3,nao,nao))
    aow = numpy.einsum('npi,np->pi', ao[:4], wv)
    vmat[0] = numint._dot_ao_ao(mol, ao[1], aow, mask, shls_slice, ao_loc)
    vmat[1] = numint._dot_ao_ao(mol, ao[2], aow, mask, shls_slice, ao_loc)
    vmat[2] = numint._dot_ao_ao(mol, ao[3], aow, mask, shls_slice, ao_loc)

    # XX, XY, XZ = 4, 5, 6
    # YX, YY, YZ = 5, 7, 8
    # ZX, ZY, ZZ = 6, 8, 9
    aow = numpy.einsum('pi,p->pi', ao[4], wv[1])
    aow+= numpy.einsum('pi,p->pi', ao[5], wv[2])
    aow+= numpy.einsum('pi,p->pi', ao[6], wv[3])
    vmat[0] += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
    aow = numpy.einsum('pi,p->pi', ao[5], wv[1])
    aow+= numpy.einsum('pi,p->pi', ao[7], wv[2])
    aow+= numpy.einsum('pi,p->pi', ao[8], wv[3])
    vmat[1] += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
    aow = numpy.einsum('pi,p->pi', ao[6], wv[1])
    aow+= numpy.einsum('pi,p->pi', ao[8], wv[2])
    aow+= numpy.einsum('pi,p->pi', ao[9], wv[3])
    vmat[2] += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
    return vmat


def get_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                          max_memory=2000, verbose=None):
    '''Full response including the response of the grids'''
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    atom_slices = mol.aoslice_by_atom()

    excsum = 0
    vmat = numpy.zeros((3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        vtmp = numpy.empty((3,nao,nao))
        for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
            ngrids = weight.size
            sh0, sh1 = atom_slices[atm_id][:2]
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask)
            rho = make_rho(0, ao[0], mask, 'LDA')
            exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1, verbose)[:2]
            vrho = vxc[0]
            aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho)
            vtmp[0] = numint._dot_ao_ao(mol, ao[1], aow, mask, shls_slice, ao_loc)
            vtmp[1] = numint._dot_ao_ao(mol, ao[2], aow, mask, shls_slice, ao_loc)
            vtmp[2] = numint._dot_ao_ao(mol, ao[3], aow, mask, shls_slice, ao_loc)
            vmat += vtmp

            # response of weights
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho, weight1)
            # response of grids coordinates
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms) * 2
            rho = vxc = vrho = aow = None

    elif xctype == 'GGA':
        ao_deriv = 2
        for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
            ngrids = weight.size
            sh0, sh1 = atom_slices[atm_id][:2]
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask)
            rho = make_rho(0, ao[:4], mask, 'GGA')
            exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1, verbose)[:2]
            vrho, vsigma = vxc[:2]
            wv = numpy.empty_like(rho)
            wv[0]  = weight * vrho
            wv[1:] = rho[1:] * (weight * vsigma * 2)
            vtmp = _gga_grad_sum(mol, ao, wv, mask, shls_slice, ao_loc)
            vmat += vtmp

            # response of weights
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho[0], weight1)
            # response of grids coordinates
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms) * 2
            rho = vxc = vrho = vsigma = wv = mat = None

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    else:
        raise NotImplementedError('meta-GGA')

    # - sign because nabla_X = -nabla_x
    return excsum, -vmat


# JCP, 98, 5612
def grids_response_cc(grids):
    mol = grids.mol
    atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid,
                                            grids.radi_method,
                                            grids.level, grids.prune)
    atm_coords = numpy.asarray(mol.atom_coords() , order='C')
    atm_dist = radi._inter_distance(mol)

    def _radii_adjust(mol, atomic_radii):
        charges = mol.atom_charges()
        if grids.radii_adjust == radi.treutler_atomic_radii_adjust:
            rad = numpy.sqrt(atomic_radii[charges]) + 1e-200
        elif grids.radii_adjust == radi.becke_atomic_radii_adjust:
            rad = atomic_radii[charges] + 1e-200
        else:
            fadjust = lambda i, j, g: g
            gadjust = lambda *args: 1
            return fadjust, gadjust

        rr = rad.reshape(-1,1) * (1./rad)
        a = .25 * (rr.T - rr)
        a[a<-.5] = -.5
        a[a>0.5] = 0.5

        def fadjust(i, j, g):
            return g + a[i,j]*(1-g**2)

        #: d[g + a[i,j]*(1-g**2)] /dg = 1 - 2*a[i,j]*g
        def gadjust(i, j, g):
            return 1 - 2*a[i,j]*g
        return fadjust, gadjust

    fadjust, gadjust = _radii_adjust(mol, grids.atomic_radii)

    def gen_grid_partition(coords, atom_id):
        ngrids = coords.shape[0]
        grid_dist = []
        grid_norm_vec = []
        for ia in range(mol.natm):
            v = (atm_coords[ia] - coords).T
            normv = numpy.linalg.norm(v,axis=0) + 1e-200
            v /= normv
            grid_dist.append(normv)
            grid_norm_vec.append(v)

        def get_du(ia, ib):  # JCP, 98, 5612 (B10)
            uab = atm_coords[ia] - atm_coords[ib]
            duab = 1./atm_dist[ia,ib] * grid_norm_vec[ia]
            duab-= uab[:,None]/atm_dist[ia,ib]**3 * (grid_dist[ia]-grid_dist[ib])
            return duab

        pbecke = numpy.ones((mol.natm,ngrids))
        dpbecke = numpy.zeros((mol.natm,mol.natm,3,ngrids))
        for ia in range(mol.natm):
            for ib in range(ia):
                g = 1/atm_dist[ia,ib] * (grid_dist[ia]-grid_dist[ib])
                p0 = fadjust(ia, ib, g)
                p1 = (3 - p0**2) * p0 * .5
                p2 = (3 - p1**2) * p1 * .5
                p3 = (3 - p2**2) * p2 * .5
                t_uab = 27./16 * (1-p2**2) * (1-p1**2) * (1-p0**2) * gadjust(ia, ib, g)

                s_uab = .5 * (1 - p3 + 1e-200)
                s_uba = .5 * (1 + p3 + 1e-200)
                pbecke[ia] *= s_uab
                pbecke[ib] *= s_uba
                pt_uab =-t_uab / s_uab
                pt_uba = t_uab / s_uba

# * When grid is on atom ia/ib, ua/ub == 0, d_uba/d_uab may have huge error
#   How to remove this error?
                duab = get_du(ia, ib)
                duba = get_du(ib, ia)
                if ia == atom_id:
                    dpbecke[ia,ia] += pt_uab * duba
                    dpbecke[ia,ib] += pt_uba * duba
                else:
                    dpbecke[ia,ia] += pt_uab * duab
                    dpbecke[ia,ib] += pt_uba * duab

                if ib == atom_id:
                    dpbecke[ib,ib] -= pt_uba * duab
                    dpbecke[ib,ia] -= pt_uab * duab
                else:
                    dpbecke[ib,ib] -= pt_uba * duba
                    dpbecke[ib,ia] -= pt_uab * duba

# * JCP, 98, 5612 (B8) (B10) miss many terms
                if ia != atom_id and ib != atom_id:
                    ua_ub = grid_norm_vec[ia] - grid_norm_vec[ib]
                    ua_ub /= atm_dist[ia,ib]
                    dpbecke[atom_id,ia] -= pt_uab * ua_ub
                    dpbecke[atom_id,ib] -= pt_uba * ua_ub

        for ia in range(mol.natm):
            dpbecke[:,ia] *= pbecke[ia]

        return pbecke, dpbecke

    natm = mol.natm
    for ia in range(natm):
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
        coords = coords + atm_coords[ia]
        pbecke, dpbecke = gen_grid_partition(coords, ia)
        z = 1./pbecke.sum(axis=0)
        w1 = dpbecke[:,ia] * z
        w1 -= pbecke[ia] * z**2 * dpbecke.sum(axis=1)
        w1 *= vol
        w0 = vol * pbecke[ia] * z
        yield coords, w0, w1


class Gradients(rhf_grad.Gradients):
    def __init__(self, mf):
        rhf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.grid_response = False
        self._keys = self._keys.union(['grid_response', 'grids'])

    def dump_flags(self):
        rhf_grad.Gradients.dump_flags(self)
        logger.info(self, 'grid_response = %s', self.grid_response)
        #if callable(self._scf.grids.prune):
        #    logger.info(self, 'Grid pruning %s may affect DFT gradients accuracy.'
        #                'Call mf.grids.run(prune=False) to mute grid pruning',
        #                self._scf.grids.prune)
        return self

    get_veff = get_veff
    grad_elec = grad_elec

Grad = Gradients


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft

    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. ,  0.757 , 0.587)] ]
    mol.basis = '631g'
    mol.build()
    mf = dft.RKS(mol)
    mf.conv_tol = 1e-14
    #mf.grids.atom_grid = (20,86)
    e0 = mf.scf()
    g = Gradients(mf)
    print(g.kernel())
#[[ -4.20040265e-16  -6.59462771e-16   2.10150467e-02]
# [  1.42178271e-16   2.81979579e-02  -1.05137653e-02]
# [  6.34069238e-17  -2.81979579e-02  -1.05137653e-02]]
    g.grid_response = True
    print(g.kernel())
    exit()

    mf.xc = 'b88,p86'
    e0 = mf.scf()
    g = Gradients(mf)
    print(g.kernel())
#[[ -8.20194970e-16  -2.04319288e-15   2.44405835e-02]
# [  4.36709255e-18   2.73690416e-02  -1.22232039e-02]
# [  3.44483899e-17  -2.73690416e-02  -1.22232039e-02]]
    g.grid_response = True
    print(g.kernel())

    mf.xc = 'b3lypg'
    e0 = mf.scf()
    g = Gradients(mf)
    print(g.kernel())
#[[ -3.59411142e-16  -2.68753987e-16   1.21557501e-02]
# [  4.04977877e-17   2.11112794e-02  -6.08181640e-03]
# [  1.52600378e-16  -2.11112794e-02  -6.08181640e-03]]


    mol = gto.Mole()
    mol.atom = [
        ['H' , (0. , 0. , 1.804)],
        ['F' , (0. , 0. , 0.   )], ]
    mol.unit = 'B'
    mol.basis = '631g'
    mol.build()

    mf = dft.RKS(mol)
    mf.conv_tol = 1e-14
    mf.kernel()
    print(Gradients(mf).kernel())
# sum over z direction non-zero, due to meshgrid response
#[[ 0  0  -2.68934738e-03]
# [ 0  0   2.69333577e-03]]
    mf = dft.RKS(mol)
    mf.grids.prune = None
    mf.grids.level = 6
    mf.conv_tol = 1e-14
    mf.kernel()
    print(Gradients(mf).kernel())
#[[ 0  0  -2.68931547e-03]
# [ 0  0   2.68911282e-03]]
