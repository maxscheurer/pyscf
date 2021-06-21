#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Analytical nuclear gradients for Polarizable Embedding
'''  # noqa: E501

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import df
from pyscf.solvent._attach_solvent import _Solvation
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import rks as rks_grad


def make_grad_object(grad_method):
    '''For grad_method in vacuum, add nuclear gradients of solvent pcmobj'''

    # Zeroth order method object must be a solvation-enabled method
    assert isinstance(grad_method.base, _Solvation)
    if grad_method.base.with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy gradients')

    grad_method_class = grad_method.__class__
    class WithSolventGrad(grad_method_class):
        def __init__(self, grad_method):
            self.__dict__.update(grad_method.__dict__)
            self.de_solvent = None
            self.de_solute = None
            self._keys = self._keys.union(['de_solvent', 'de_solute'])

        # TODO: if moving to python3, change signature to
        # def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        def kernel(self, *args, **kwargs):
            dm = kwargs.pop('dm', None)
            if dm is None:
                dm = self.base.make_rdm1(ao_repr=True)

            self.de_solvent = kernel(self.base.with_solvent, dm)
            self.de_solute = grad_method_class.kernel(self, *args, **kwargs)
            self.de = self.de_solute + self.de_solvent

            if self.verbose >= logger.NOTE:
                logger.note(self, '--------------- %s (+%s) gradients ---------------',
                            self.base.__class__.__name__,
                            self.base.with_solvent.__class__.__name__)
                rhf_grad._write(self, self.mol, self.de, self.atmlst)
                logger.note(self, '----------------------------------------------')
            return self.de

        def _finalize(self):
            # disable _finalize. It is called in grad_method.kernel method
            # where self.de was not yet initialized.
            pass

    return WithSolventGrad(grad_method)


def kernel(peobj, dm, verbose=None):
    import cppe
    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        # UHF density matrix
        dm = dm[0] + dm[1]

    mol = peobj.mol
    natoms = mol.natm
    de = numpy.zeros((natoms, 3))
    cppe_state = peobj.cppe_state
    nuc_ee_grad = cppe_state.nuclear_interaction_energy_gradient()
    elec_ee_grad = numpy.zeros_like(nuc_ee_grad)

    positions = numpy.array([p.position for p in peobj.potentials])
    moments = []
    orders = []
    for p in peobj.potentials:
        p_moments = []
        for m in p.multipoles:
            m.remove_trace()
            p_moments.append(m.values)
        orders.append(m.k)
        moments.append(p_moments)
    orders = numpy.asarray(orders)

    fakemol = gto.fakemol_for_charges(positions)
    integral0 = df.incore.aux_e2(mol, fakemol, intor='int3c2e_ip1', comp=3)
    moments_0 = numpy.array([m[0] for m in moments])
    op = numpy.einsum('cijg,ga->cij', integral0, moments_0 * cppe.prefactors(0))

    if numpy.any(orders >= 1):
        idx = numpy.where(orders >= 1)[0]
        fakemol = gto.fakemol_for_charges(positions[idx])

        integral11 = df.incore.aux_e2(mol, fakemol, intor='int3c2e_ipip1', comp=9)
        integral12 = df.incore.aux_e2(mol, fakemol, intor='int3c2e_ipvip1', comp=9)

        integral11 = integral11.reshape(3, 3, *integral11.shape[1:])
        integral12 = integral12.reshape(3, 3, *integral12.shape[1:])

        moments_1 = numpy.array([moments[i][1] for i in idx])
        v = numpy.einsum('caijg,ga,a->cij', integral11, moments_1, cppe.prefactors(1))
        v += numpy.einsum('caijg,ga,a->cij', integral12, moments_1, cppe.prefactors(1))
        op += v

    if numpy.any(orders >= 2):
        idx = numpy.where(orders >= 2)[0]
        n_sites = idx.size
        # moments_2 is the lower triangle of
        # [[XX, XY, XZ], [YX, YY, YZ], [ZX, ZY, ZZ]] i.e.
        # XX, XY, XZ, YY, YZ, ZZ = 0,1,2,4,5,8
        # symmetrize it to the upper triangle part
        # XX, YX, ZX, YY, ZY, ZZ = 0,3,6,4,7,8
        m2 = numpy.einsum('ga,a->ga', [moments[i][2] for i in idx],
                          cppe.prefactors(2))
        moments_2 = numpy.zeros((n_sites, 9))
        moments_2[:, [0, 1, 2, 4, 5, 8]]  = m2
        moments_2[:, [0, 3, 6, 4, 7, 8]] += m2
        moments_2 *= .5

        for ii, pos in enumerate(positions[idx]):
            with mol.with_rinv_orig(pos):
                int1 = mol.intor('int1e_ipipiprinv', comp=27).reshape(3, 9, mol.nao, mol.nao)
                int2 = mol.intor('int1e_ipiprinvip', comp=27).reshape(3, 9, mol.nao, mol.nao)
                int3 = int2.reshape(3, 3, 3, -1).transpose(2, 0, 1, -1).reshape(3, 9, mol.nao, mol.nao)
                op += numpy.einsum('caij,a->cij', int1, moments_2[ii])
                op += numpy.einsum('caji,a->cij', int3, moments_2[ii])
                op += 2.0 * numpy.einsum('caij,a->cij', int2, moments_2[ii])
    
    # induction part
    positions = cppe_state.positions_polarizable
    n_polsites = positions.shape[0]
    if n_polsites > 0:
        nuc_field_grad = cppe_state.nuclear_field_gradient().reshape(natoms, 3, n_polsites, 3)
        fakemol = gto.fakemol_for_charges(positions)
        j3c = df.incore.aux_e2(mol, fakemol, intor='int3c2e_ip1')
        elf = numpy.einsum('aijg,ij->ga', j3c, dm) + numpy.einsum('aijg,ji->ga', j3c, dm)
        peobj.cppe_state.update_induced_moments(elf.ravel(), False)
        induced_moments = numpy.array(cppe_state.get_induced_moments()).reshape(n_polsites, 3)
        grad_induction_nuc = -numpy.einsum("acpk,pk->ac", nuc_field_grad, induced_moments)
        de += grad_induction_nuc

        # refactor...
        integral11 = df.incore.aux_e2(mol, fakemol, intor='int3c2e_ipip1', comp=9)
        integral12 = df.incore.aux_e2(mol, fakemol, intor='int3c2e_ipvip1', comp=9)

        integral11 = integral11.reshape(3, 3, *integral11.shape[1:])
        integral12 = integral12.reshape(3, 3, *integral12.shape[1:])

        v = numpy.einsum('caijg,ga,a->cij', integral11, induced_moments, cppe.prefactors(1))
        v += numpy.einsum('caijg,ga,a->cij', integral12, induced_moments, cppe.prefactors(1))
        op += v

    ao_slices = mol.aoslice_by_atom()
    for ia in range(natoms):
        k0, k1 = ao_slices[ia, 2:]

        if peobj.do_ecp:
            op[:, k0:k1] += peobj.ecpmol.intor('ECPscalar_ipnuc', comp=3)[:, k0:k1]

        Dx_a = numpy.zeros_like(op)
        Dx_a[:, k0:k1] = op[:, k0:k1]
        Dx_a += Dx_a.transpose(0, 2, 1)
        elec_ee_grad[ia] -= numpy.einsum("xpq,pq", Dx_a, dm)
    de += nuc_ee_grad + elec_ee_grad
    return de
    