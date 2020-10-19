#!/usr/bin/env python
''' todo list
redefine score based on grid point rmsd (align first and then rmsd)
use area as size may be better (better include electrons)
-tc option use area is better
save latest match information to save time
use two edge angle and one  angle error in %
print layer infor one by one in the check layer function for the impatient
-t s function for slice surfac3
check layer function support molecule system
'''
import numpy as np
import os
import re
import sys
import math
import copy
import argparse
import pandas as pd
import itertools
from collections import defaultdict
from scipy.spatial import distance
from collections import Counter
from sklearn.cluster import MeanShift
from datetime import datetime


class LatticeMatch:
    def __init__(self, ps1, ps2, tol='0.1,0.1,0.05', size='10,20',
                 angle_range='90,61', size_tol=100):
        self.a1, self.b1, _ = np.array(ps1.cell_vect)
        self.a2, self.b2, _ = np.array(ps2.cell_vect)
        self.tol_edge, self.tol_area, self.tol_all = [float(i) for i in tol.split(',')]
        self.edge_size, self.angle_size = [int(i) for i in size.split(',')]
        self.angle_range = [float(i) for i in angle_range.split(',')]
        self.size_tol = int(size_tol)
        self.ab_switch = 'abab'
        len_a1, len_b1 = ps1.cell_param[:2]
        g1 = ps1.cell_param[-1]
        len_a2, len_b2 = ps2.cell_param[:2]
        g2 = ps2.cell_param[-1]
        delta_1 = round(abs(len_a1-len_b1), 3)
        delta_2 = round(abs(len_a2-len_b2), 3)
        print('Cell1: a1={:.4f},b1={:.4f},gamma={:.4f}'.format(len_a1, len_b1, g1))
        print('Cell2: a2={:.4f},b2={:.4f},gamma={:.4f}'.format(len_a2, len_b2, g2))
        print('Cell_1=a1b1, Cell_2=a2b2. Start with matching a1 with a2 ...')
        self.match_data = []
        self.match_all()
        if delta_1 > 0 or delta_2 > 0:
            self.b1, self.a1, _ = np.array(ps1.cell_vect)
            self.b2, self.a2, _ = np.array(ps2.cell_vect)
            self.ab_switch = 'baba'
            print('Cell_1=b1a1, Cell_2=b2a2. Start with matching b1 with b2 ...')
            self.match_all()
        if delta_1 > 0 and delta_2 > 0:
            self.a1, self.b1, _ = np.array(ps1.cell_vect)
            self.b2, self.a2, _ = np.array(ps2.cell_vect)
            self.ab_switch = 'abba'
            print('Cell_1=a1b1, Cell_2=b2a2. Start with matching a1 with b2 ...')
            self.match_all()
            self.b1, self.a1, _ = np.array(ps1.cell_vect)
            self.a2, self.b2, _ = np.array(ps2.cell_vect)
            self.ab_switch = 'baab'
            print('Cell_1=b1a1, Cell_2=a2b2. Start with matching b1 with a2 ...')
            self.match_all()
        print('Total {:d} match pairs found'.format(len(self.match_data)))

    def match_all(self):
        a = self.match_a()
        print('{:d} pairs of edge match with error < {:.2f}; '.format(len(a), self.tol_edge))
        area = self.match_area(a)
        print('{:d} pairs of area match with error < {:.2f}; '.format(len(area), self.tol_area))
        match_data = self.match_angle(area)
        print('{:d} pairs of total match with error < {:.2f};'.format(len(match_data), self.tol_all))
        self.match_data = self.match_data + match_data

    def match_a(self, n=None, threshold=None):
        if not threshold:
            threshold = self.tol_edge
        if not n:
            n = self.edge_size
        matched_a = []
        for i in range(1, n+1):
            for j in range(1, n+1):
                len_a1 = np.linalg.norm(i*self.a1)
                len_a2 = np.linalg.norm(j*self.a2)
                da = abs(len_a1 - len_a2)
                da_p = da / (len_a1+len_a2)
                matched_a.append([da_p, da, i, j])
        matched_a = sorted(matched_a, key=lambda x: x[0])
        if threshold >= 1:
            matched_a = matched_a[:int(threshold)]
        elif threshold > 0:
            matched_a = [i for i in matched_a if i[0] < threshold]
        return matched_a

    def area(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(np.cross(a, b))

    def match_area(self, matched_a, n=None, threshold=None):
        if not threshold:
            threshold = self.tol_area
        if not n:
            n = self.edge_size
        matched_area = []
        for a in matched_a:
            da_p, da, n1, n2 = a
            for i in range(1, n+1):
                for j in range(1, n+1):
                    area1 = self.area(self.a1*n1, self.b1*i)
                    area2 = self.area(self.a2*n2, self.b2*j)
                    ds = abs(area1-area2)
                    ds_p = ds / (area1+area2)
                    matched_area.append([ds_p, da_p, ds, da, n1, n2, i, j])
        matched_area = sorted(matched_area, key=lambda x: x[0]+x[1]**2)
        if threshold >= 1:
            matched_area = matched_area[:int(threshold)]
        elif threshold > 0:
            matched_area = [i for i in matched_area if i[0]**0.5 < threshold]
        return matched_area

    def match_angle(self, matched_area, n=None, threshold=None,
                    angle_range=None, size_tol=None, ab_switch=None):
        if not ab_switch:
            ab_switch = self.ab_switch
        if not threshold:
            threshold = self.tol_all
        if not n:
            n = self.angle_size
        if not angle_range:
            angle_range = self.angle_range
        if not size_tol:
            size_tol = self.size_tol
        angle_center, angle_ext = angle_range

        def angle(a, b):
            len_a = np.linalg.norm(a)
            len_b = np.linalg.norm(b)
            angle = np.arccos(np.dot(a, b)/(len_a*len_b))
            return np.rad2deg(angle)
        matched_data = []
        for a in matched_area:
            ds_p, da_p, ds, da, n1, n2, m1, m2 = a
            for i in range(-1*n, n+1):
                for j in range(-1*n, n+1):
                    b1 = i*self.a1 + m1*self.b1
                    b2 = j*self.a2 + m2*self.b2
                    # ds2=abs(area1-area2)
                    g1 = angle(self.a1, b1)
                    g2 = angle(self.a2, b2)
                    if max(g1, g2) > angle_center + angle_ext or min(g1, g2) < angle_center - angle_ext:
                        continue
                    dg = abs(g1-g2)
                    dg_p = dg / 90
                    s1 = self.area(self.a1*n1, b1)
                    s2 = self.area(self.a2*n2, b2)
                    len_b1 = np.linalg.norm(b1)
                    len_b2 = np.linalg.norm(b2)
                    len_a1 = np.linalg.norm(n1*self.a1)
                    len_a2 = np.linalg.norm(n2*self.a2)
                    db = abs(len_b1-len_b2)
                    db_p = db / (len_b1+len_b2)
                    match_score = ((da_p**2+db_p**2+dg_p**2)/3)**0.5*100
                    matched_data.append({'error': [match_score, ds_p, da_p, db_p, dg_p],
                                         'delta': [ds, da, db, dg],
                                         'trans_mat1': [n1, m1, i, ab_switch[:2]],
                                         'trans_mat2': [n2, m2, j, ab_switch[2:]],
                                         'super_lattice1': [s1, len_a1, len_b1, g1],
                                         'super_lattice2': [s2, len_a2, len_b2, g2],
                                         'size': [n1, m1, n2, m2],
                                         'angle_penalty': abs((g1+g2)/2-angle_center)})
        matched_data = sorted(matched_data, key=lambda x: x['error'][0])
        #if threshold >= 1:
        #    matched_data = matched_data[:int(threshold)]
        if threshold > 0:
            matched_data = [i for i in matched_data if i['error'][0]/100 < threshold]

        def max_size(s):
            return max([s[0]*s[1], s[2]*s[3]])
        if size_tol:
            matched_data = [i for i in matched_data if max_size(i['size']) < size_tol]
        return matched_data

    def render_data(self, num_print=100, sort_str='csh'):
        matched_data = sorted(self.match_data, key=lambda x: x['error'][3])

        def sort_func_gen(sort_str):
            def sortby(data):
                c2p = {}
                c2p['c'] = round(data['error'][0], 2)
                c2p['s'] = data['size'][0]*data['size'][1]
                c2p['h'] = abs(data['trans_mat1'][2])+abs(data['trans_mat2'][2])
                c2p['a'] = data['angle_penalty']
                return tuple([c2p[i] for i in sort_str])
            return sortby
        sort_func = sort_func_gen(sort_str)
        matched_data = sorted(self.match_data, key=sort_func)
        matched_data = matched_data[:num_print]
        self.data = matched_data
        '''print best n match'''
        def vec_str(t_mat):
            if t_mat[3] == 'ab':
                u = '{:d}a'.format(t_mat[0])
                v = '{:d}b{:+d}a'.format(t_mat[1], t_mat[2])
            if t_mat[3] == 'ba':
                u = '{:d}b'.format(t_mat[0])
                v = '{:d}a{:+d}b'.format(t_mat[1], t_mat[2])
            return u+','+v
        print('{:<4s}{:<48s}{:<48s}{:<24s}'.format('No.', 'Layer1', 'Layer2', 'Error(%)'))
        ts = '{:<8s}{:<8s}{:<12s}{:<8s}{:<12s}'.format('u', 'v', 'area', 'gamma', 'u,v')
        td = '{:<6s}{:<6s}{:<6s}{:<6s}{:<8s}'.format('u', 'v', 'gamma', 'score', 'size')
        print('    '+ts+ts+td)
        for no, i in enumerate(matched_data):
            s1, a1, b1, g1 = i['super_lattice1']
            s2, a2, b2, g2 = i['super_lattice2']
            score, ds, da, db, dg = np.array(i['error'])*100
            uv1 = vec_str(i['trans_mat1'])
            uv2 = vec_str(i['trans_mat2'])
            size1 = i['trans_mat1'][0]*i['trans_mat1'][1]
            size2 = i['trans_mat2'][0]*i['trans_mat2'][1]
            size = str(size1)+','+str(size2)
            no_str = '{:<4d}'.format(no)
            s = '{:<8.2f}{:<8.2f}{:<12.3E}{:<8.2f}{:<12s}'
            d = '{:<6.2f}{:<6.2f}{:<6.2f}{:<6.2f}{:<8s}'.format(da, db, dg, score/100, size)
            s1 = s.format(a1, b1, s1, g1, uv1)
            s2 = s.format(a2, b2, s2, g2, uv2)
            print(no_str+s1+s2+d)


class StructureWriter:

    def __init__(self):
        self.write_func = {'res': self._write_res,
                           'pdb': self._write_pdb,
                           'mol2': self._write_mol2,
                           'gro': self._write_gro,
                           'cif': self._write_cif
                           }

    def fd(self, input, f=float, d=0.0):
        '''fill default'''
        if input == '':
            input = f(d)
        return f(input)

    def get_value(self, prop_name, fill=None):
        try:
            prop_name = getattr(self.st, prop_name)
        except AttributeError:
            if fill is None:
                prop_name = fill
            elif len(fill) == len(self.st.cart_coord):
                prop_name = fill
            elif len(fill) < len(self.st.cart_coord) or isinstance(fill, str):
                prop_name = len(self.st.cart_coord) * [fill]
        return prop_name

    def write_file(self, st, basename=None, ext='pdb'):
        self.st = st
        if not basename:
            basename = self.st.basename
        self.basename = basename
        filename = basename+'.'+ext
        print('Generating file {:s}'.format(filename))
        self.file = open(filename, 'w')
        try:
            self.write_func[ext]()
        except KeyError:
            print('Format {:s} is not supported yet'.format(ext))
            sys.exit()
        self.file.close()

    def _write_res(self):
        self.file.write('TITL {:s}\n'.format(self.basename))
        self.file.write(('CELL'+7*'{:>11.6f}'+'\n').format(0.0000, *self.st.cell_param))
        self.file.write('LATT -1\n')
        elem_set = list(set(self.st.elem))
        elem_str = ' '.join(elem_set)
        self.file.write('SFAC '+elem_str+'\n')
        for i, c in enumerate(self.st.fcoord):
            elem_name = self.st.elem[i]
            atom_code = elem_set.index(self.st.elem[i])+1
            clist = [elem_name, atom_code] + c + [1.0, 0.0]
            s = '{:<6s}{:<3d}{:<14.8f}{:<14.8f}{:<14.8f}{:<11.5f}{:<10.5f}\n'.format(*clist)
            self.file.write(s)
        self.file.write('END')

    def _write_pdb(self):

        self.file.write('REMARK    Generated by Masagna\n')
        if self.st.period_flag == 1:
            self.file.write('CRYST1{:>9.3f}{:>9.3f}{:>9.3f}{:>7.2f}{:>7.2f}{:>7.2f} {:<11s}\n'
                            .format(*(self.st.cell_param+['P1'])))
            scale = np.matrix(self.st.cell_vect).I.T.tolist()
            for i in range(1, 4):
                self.file.write('SCALE{:<4d}{:>10.6f}{:>10.6f}{:>10.6f}{:5s}{:>10.5f}\n'
                                .format(*([i]+scale[i-1]+[' ']+[0.0])))
        atomname = self.st.getter('atomname')
        if not all(atomname):
            self.st.setter('atomname', self.st.elem)
        for i, a in enumerate(self.st.atoms):
            str1 = '{:<6s}{:>5d} {:<4s} {:3s} {:1s}{:>4d}    '\
                   .format('ATOM', a['sn'], a['atomname'], a['resname'],
                           a['chainid'], self.fd(a['resid'], f=int, d=1))
            str2 = '{:>8.3f}{:>8.3f}{:>8.3f}'.format(*a['coord'])
            str3 = '{:>6.2f}{:>6.2f}{:10s}'\
                   .format(self.fd(a['occupancy'], d=1), self.fd(a['bfactor']), ' ')
            str4 = '{:>2s}{:2s}\n'.format(a['elem'], a['formal_charge'])
            self.file.write(str1+str2+str3+str4)

    def _write_gro(self):
        atomname = self.st.getter('atomname')
        if not all(atomname):
            self.st.setter('atomname', self.st.elem)
        self.file.write('gro file generate by masagna, t= 0.0\n')
        self.file.write('{:d}\n'.format(len(self.st.atoms)))
        for i, a in enumerate(self.st.atoms):
            name_id = "{:5d}{:<5s}{:>5s}{:5d}"\
                      .format(self.fd(a['resid'], f=int, d=1),
                              self.fd(a['resname'], f=str, d='MOL'),
                              a['atomname'], a['sn'])
            coord = "{:8.3f}{:8.3f}{:8.3f}"\
                    .format(*[i/10 for i in self.st.coord[i]])
            if a['velocity'] == '':
                vel = '\n'
            else:
                vel = "{:8.4f}{:8.4f}{:8.4f}\n"\
                    .format(*a['velocity'])
            self.file.write(name_id+coord+vel)
        if len(self.st.cell_vect) == 3:
            v1, v2, v3 = self.st.cell_vect
            vlist = [v1[0], v2[1], v3[2], v1[1], v1[2], v2[0], v2[2], v3[0], v3[1]]
            vstr = ' '.join(['{:.5f}'.format(i/10) for i in vlist]) + '\n'
            self.file.write(vstr)

    def _write_mol2(self):
        atomname = self.st.getter('atomname')
        if not all(atomname):
            self.st.setter('atomname', self.st.elem)
        atomtype = self.st.getter('atomtype')
        if not all(atomtype):
            self.st.setter('atomtype', self.st.elem)
        self.file.write('@<TRIPOS>MOLECULE\n')
        self.file.write('{:s}\n'.format(self.st.basename))
        self.file.write('{:d} 0\n'.format(len(self.st.coord)))
        self.file.write('SMALL\n')
        self.file.write('NO_CHARGE\n')
        self.file.write('@<TRIPOS>ATOM\n')
        for i, a in enumerate(self.st.atoms):
            str1 = '{:<6d}{:<6s}'.format(a['sn'], a['atomname'])
            str2 = '{:<12.5f}{:<12.5f}{:<12.5f}'.format(*a['coord'])
            str3 = '{:<6s}'.format(a['atomtype'])
            if a['resid'] != '':
                str4 = '{:<6d}'.format(a['resid'])
                if a['resname'] != '':
                    str4 = str4 + '{:<6s}'.format(a['resname'])
                    if a['charge'] != '':
                        str4 = str4 + '{:<.6f}'.format(a['charge'])
            else:
                str4 = ''
            line = str1+str2+str3+str4+'\n'
            self.file.write(line)

    def _write_cif(self):
        self.file.write('data_'+self.st.basename+'\n')
        self.file.write('{:35s}{:.6f}\n'.format('_cell_length_a', self.st.cell_param[0]))
        self.file.write('{:35s}{:.6f}\n'.format('_cell_length_b', self.st.cell_param[1]))
        self.file.write('{:35s}{:.6f}\n'.format('_cell_length_c', self.st.cell_param[2]))
        self.file.write('{:35s}{:.6f}\n'.format('_cell_angle_alpha', self.st.cell_param[3]))
        self.file.write('{:35s}{:.6f}\n'.format('_cell_angle_beta', self.st.cell_param[4]))
        self.file.write('{:35s}{:.6f}\n'.format('_cell_angle_gamma', self.st.cell_param[5]))
        self.file.write('loop_\n'
                      '_atom_site_label\n'
                      '_atom_site_type_symbol\n'
                      '_atom_site_fract_x\n'
                      '_atom_site_fract_y\n'
                      '_atom_site_fract_z\n')
        atomname = self.st.getter('atomname')
        if not all(atomname):
            self.st.setter('atomname', self.st.elem)
        for i, a in enumerate(self.st.atoms):
            str1 = '{:7s}{:7s}'.format(a['atomname'], a['elem'])
            str2 = '{:14.8f}{:14.8f}{:14.8f}'.format(*a['fcoord'])
            self.file.write(str1+str2+'\n')

class StructureReader:

    B2A = 0.529177249

    def __init__(self):
        self.read_func = {'.res': self._read_res,
                          '.cif': self._read_cif,
                          '.xsf': self._read_xsf,
                          '.STRUCT': self._read_STRUC,
                          '.vasp': self._read_vasp,
                          '.pdb': self._read_pdb,
                          '.mol2': self._read_mol2,
                          '.gro': self._read_gro,
                          }

    def read_struc(self, inputfile):
        self.st = Structure()
        try:
            self.file = open(inputfile, 'r')
            filename = os.path.basename(inputfile)
            self.basename, ext = os.path.splitext(filename)
            self.st.basename = self.basename
        except TypeError:
            sys.exit('Error! The input {:s} is not a file'
                     .format(str(inputfile)))
        if not ext:
            if self.basename == 'CONTCAR' or self.basename == 'POSCAR':
                ext = '.vasp'
            else:
                print('File {:s} do not have extension, exit now'.format(filename))
                sys.exit()
        self.read_func[ext]()
        self.st.complete_self()
        return self.st

    def _read_res(self):
        for l in self.file:
            if 'CELL' in l:
                param = [float(i) for i in l.split()[2:]]
            if 'SFAC' in l:
                ele = l.split()[1:]
            if re.search(r'\s+\d+\s+-?\d+.\d+\s+-?\d+.\d+\s+-?\d+.\d+\s+', l):
                atom = defaultdict(str)
                coord = [float(i) for i in (l.split()[2:5])]
                atom['elem'] = ele[int(l.split()[1])-1]
                atom['fcoord'] = coord
                self.st.atoms.append(atom)
        self.st.cell_param = param

    def _read_pdb(self):
        for l in self.file:
            atom = defaultdict(str)
            if 'CRYST1' in l:
                param = [float(i) for i in l.split()[1:7]]
            if l.startswith('ATOM') or l.startswith('HETATM'):
                atom['resname'] = l[17:20].strip()
                atom['atomname'] = l[12:16].strip()
                atom['sn'] = int(l[6:11])
                try:
                    atom['resid'] = int(l[22:26].strip())
                except ValueError:
                    pass
                try:
                    atom['chainid'] = int(l[21])
                except ValueError:
                    pass
                try:
                    atom['bfactor'] = float(l[60:66].strip())
                except ValueError:
                    pass
                try:
                    atom['occupancy'] = float(l[54:60].strip())
                except ValueError:
                    pass

                x = float(l[30:38].strip())
                y = float(l[38:46].strip())
                z = float(l[46:54].strip())
                atom['elem'] = l[76:78].strip()
                atom['coord'] = [x, y, z]
                self.st.cell_param = param
                self.st.atoms.append(atom)

    def _read_cif(self):
        coord_flag = 0
        index = 0
        name_idx, ele_idx = [None, None]
        fx_idx, fy_idx, fz_idx = [None, None, None]
        # read coordinate and cell parameters from cif file
        for line in self.file:
            line = line.replace(')', '').replace('(', '')
            if re.search('_cell_length_a', line):
                a = float(line.split()[1])
            if re.search('_cell_length_b', line):
                b = float(line.split()[1])
            if re.search('_cell_length_c', line):
                c = float(line.split()[1])
            if re.search('_cell_angle_alpha', line):
                alpha = float(line.split()[1])
            if re.search('_cell_angle_beta', line):
                beta = float(line.split()[1])
            if re.search('_cell_angle_gamma', line):
                gamma = float(line.split()[1])
            if re.search('_symmetry_Int_Tables_number', line):
                self.st.sym_group = float(line.split()[1])
                if float(self.st.sym_group) != 1:
                    print('the symGroup in cif file is not 1, exit now')
                    sys.exit()
            if 'loop_' in line:
                coord_flag = 1
            if '_atom_site_' in line and coord_flag == 1:
                if 'label' in line:
                    name_idx = index
                if 'type_symbol' in line:
                    ele_idx = index
                if 'fract_x' in line:
                    fx_idx = index
                if 'fract_y' in line:
                    fy_idx = index
                if 'fract_z' in line:
                    fz_idx = index
                index = index + 1

            if re.search(r'-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+.\d+\s+', line) and coord_flag == 1:
                atom = defaultdict(str)
                fx = float(line.split()[fx_idx])
                fy = float(line.split()[fy_idx])
                fz = float(line.split()[fz_idx])
                coord = [fx, fy, fz]
                if ele_idx is not None:
                    atom['elem'] = line.split()[ele_idx]
                if name_idx is not None:
                    atom['atomname'] = line.split()[name_idx]
                atom['fcoord'] = coord
                self.st.atoms.append(atom)
        self.st.cell_param = [a, b, c, alpha, beta, gamma]

    def _read_xsf(self):
        read_vect = 0
        read_coord = 0
        for line in self.file:
            if 'PRIMVEC' in line:
                read_vect = 1
                continue
            if read_vect == 1 and len(line.split()) == 3:
                self.st.cell_vect.append([float(i) for i in line.split()])
                if len(self.st.cell_vect) == 3:
                    read_vect = 0
            if 'PRIMCOORD' in line:
                read_vect = 0
                read_coord = 1
                continue
            if read_coord == 1 and len(line.split()) == 4:
                atom = defaultdict(str)
                atom['atomnum'] = int(line.split()[0])
                atom['coord'] = [float(i) for i in line.split()[1:]]
                self.st.atoms.append(atom)
            if len(self.st.atoms) > 0 and len(line.split()) != 4:
                read_coord = 0

    def _read_gro(self):
        for l in itertools.islice(self.file, 2, None):
            if l.isupper() or l.islower():
                atom = defaultdict(str)
                atom['resid'] = int(l[:5].strip())
                atom['resname'] = l[5:10].strip()
                atom['atomname'] = l[10:15].strip()
                atom['sn'] = int(l[15:20].strip())
                x = float(l[20:28])*10
                y = float(l[28:36])*10
                z = float(l[36:44])*10
                atom['coord'] = [x, y, z]
                try:
                    vx = float(l[44:52])  # unit is nm/ps
                    vy = float(l[52:60])
                    vz = float(l[60:68])
                except ValueError:
                    vx = 0
                    vy = 0
                    vz = 0
                atom['velocity'] = [vx, vy, vz]
                self.st.atoms.append(atom)
            else:
                if len(l.split()) == 3:
                    x1, y2, z3 = [float(i)*10 for i in l.split()]
                    y1, z1, x2, z2, x3, y3 = [0, 0, 0, 0, 0, 0]
                elif len(l.split()) == 9:
                    x1, y2, z3, y1, z1, x2, z2, x3, y3 = \
                        [float(i)*10 for i in l.split()]
                cell_vect = [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]
        self.st.cell_vect = cell_vect

    def _read_STRUC(self):
        readCell = 0
        readCoord = 0
        # bohr convert to angstrom
        self.dimension = 3
        for line in self.file:
            if '$cell vectors' in line:
                readCell = 1
                continue
            if '$coordinates' in line:
                readCoord = 1
                readCell = 0
                continue
            if '$END' in line:
                readCoord = 0
                readCell = 0
                continue
            if readCell == 1:
                va = [float(i)*self.B2A for i in line.split()]
                readCell += 1
                continue
            if readCell == 2:
                vb = [float(i)*self.B2A for i in line.split()]
                if max(map(float, vb)) > 499:
                    self.dimension = self.dimension-1
                readCell += 1
                continue
            if readCell == 3:
                vc = [float(i)*self.B2A for i in line.split()]
                if max(map(float, vc)) > 499:
                    self.dimension = self.dimension-1
            if readCoord == 1:
                atom = int(line.split()[4])
                coord = np.array(float(line.split()[1:4]))*self.B2A
                self.st.cart_coord.append(coord)
                self.st.atoms.append(self.st._an2elem[atom])
        self.st.cell_vector = [va, vb, vc]

    def _read_mol2(self):
        read_flag = 0
        for l in self.file:
            if '@<TRIPOS>ATOM' in l:
                read_flag = 1
                continue
            if read_flag == 1 and '@<TRIPOS>' in l:
                read_flag = 0
            if read_flag == 1 and len(l.split()) >= 6:
                atom = defaultdict(str)
                sn, atomname, x, y, z, atom_type = l.split()[:6]
                atom['sn'] = int(sn)
                atom['atomname'] = atomname
                atom['coord'] = [float(i) for i in [x, y, z]]
                atom['atomtype'] = atom_type
                if len(l.split()) >= 7:
                    atom['resid'] = int(l.split()[6])
                if len(l.split()) >= 8:
                    atom['resname'] = l.split()[7]
                if len(l.split()) >= 9:
                    atom['charge'] = float(l.split()[8])
                self.st.atoms.append(atom)

    def _read_vasp(self):
        lines = self.file.readlines()
        scaling_factor = float(lines[1].strip())
        va = np.array([float(i) for i in lines[2].split()])*scaling_factor
        vb = np.array([float(i) for i in lines[3].split()])*scaling_factor
        vc = np.array([float(i) for i in lines[4].split()])*scaling_factor
        self.st.cell_vect = [list(va), list(vb), list(vc)]
        element = lines[5].strip().split()
        num_of_element = lines[6].strip().split()
        elem = []
        for i in range(len(element)):
            elem += [element[i]] * int(num_of_element[i])
        for i in elem:
            atom = defaultdict(str)
            atom['elem'] = i
            self.st.atoms.append(atom)
        if lines[7].strip().lower().startswith('c'):
            coord = []
            for n in lines[7:]:
                if re.search('-?\d+.\d+\s+-?\d+.\d+\s+-?\d+.\d+', n):
                    c = np.array([float(i) for i in n.strip().split()])*scaling_factor
                    coord.append(list(c))
            self.st.setter('coord', coord)
        else:
            fcoord = []
            for n in lines[7:]:
                if re.search('-?\d+.\d+\s+-?\d+.\d+\s+-?\d+.\d+', n):
                    fcoord.append([float(i) for i in  n.strip().split()[:3]])
            self.st.setter('fcoord', fcoord)

class Transformer:
    '''All methods that transforms cartersian coords are collected here'''

    def __init__(self, structure):
        self.S = structure

    def shift_abc(self, abc):
        '''abc is a list of three numbers. Means shift of fraction coords in a
        b c cell axis'''
        s = copy.deepcopy(self.S)
        s.setter('fcoord', (np.array(s.fcoord) + np.array(abc)).tolist())
        s.frac2cart()
        return s

    def shift_xyz(self, xyz, pbc=False):
        '''xyz is a list of three numbers. Means shift of cart coords in x y z'''
        s = copy.deepcopy(self.S)
        s.setter('coord', (np.array(s.coord) + np.array(xyz)).tolist())
        if s.period_flag == 1:
            s.cart2frac()
            if pbc == True:
                s.wrap_in_fcoord()
                s.frac2cart()
        return s

    def rotate(self, abc, center='g'):
        '''abc is a list of three numbers in degrees.
        Means rotation by angle a b c about x y z axis, respectively
        remeber to translate origin to 0,0,0
        '''
        mol = copy.deepcopy(self.S)
        if center == 'g':
            shift_v = np.array([0, 0, 0]) - np.array(mol.geom_center)
        elif center == 'm':
            shift_v = np.array([0, 0, 0]) - np.array(mol.mass_center)
        else:
            shift_v = np.array([0, 0, 0])
        mol = mol.T.shift_xyz(shift_v)
        s = np.sin(np.deg2rad(np.array(abc)))
        c = np.cos(np.deg2rad(np.array(abc)))
        rx = np.matrix([[1, 0, 0], [0, c[0], -1*s[0]], [0, s[0], c[0]]])
        ry = np.matrix([[c[1], 0, s[1]], [0, 1, 0], [-1*s[1], 0, c[1]]])
        rz = np.matrix([[c[2], -1*s[2], 0], [s[2], c[2], 0], [0, 0, 1]])
        coords = np.matrix(mol.coord)*rx.T*ry.T*rz.T
        mol.setter('coord', coords.tolist())
        mol = mol.T.shift_xyz(-1*shift_v)
        return mol


class CellParam:
    '''metheds that modify lattice parameters'''

    def __init__(self, structure):
        self.S = structure

    def set_cell_param(self, cell_param, keep='cart'):
        '''set new cell parameter. If keep is cart, the cartesian coord will be
        kept and if keep is frac, the fraction coord whill be kept'''
        s = copy.deepcopy(self.S)

        s.cell_param = cell_param
        s.param2vect()
        if keep.startswith('c'):
            s.cart2frac()
        if keep.startswith('f'):
            s.frac2cart()
        return s

    def switch_edge(self, order=[1, 0, 2]):
        new_frac = []
        new_vect = []
        s = copy.deepcopy(self.S)
        for i in range(3):
            new_vect.append(s.cell_vect[order[i]])
            new_frac.append(np.array(s.fcoord).T[order[i]])
        s.cell_vect = new_vect
        s.setter('fcoord', np.array(new_frac).T.tolist())
        s.vect2param()
        s.param2vect()
        s.frac2cart()
        return s

    def delete_cell(self):
        s = copy.deepcopy(self.S)
        s.setter('fcoord', '')
        s.cell_vect = []
        s.cell_param = []
        s.period_flag = 0
        return s

    def shift_cell_origin(self, frac_origin, cart_origin=None):
        '''shift cell to new_origin, new_origin is a three integer in frac coord'''
        s = copy.deepcopy(self.S)
        if not cart_origin:
            cart_origin = (np.matrix(frac_origin) * np.matrix(s.cell_vect)).tolist()
        s.cell_origin = cart_origin
        s.frac2cart()
        return s

    def super_cell(self, trans_mat):
        '''format of trans_mat:
        u = n1*a+n2*b+n3*c
        v = m1*a+m2*b+m3*c
        w = p1*a+p2*b+p3*c
        trans_mat = [[n1,n2,n3],[m1,m2,m3],[p1,p2,p3]]
        do not support negative value in trans_mat
        '''
        s = self.S
        slat = Structure()
        na = trans_mat[0][0]
        nb = trans_mat[1][1]
        nc = trans_mat[2][2]
        slat.basename = s.basename
        new_frac = []
        prim_frac = np.array(s.fcoord)
        for i in range(na):
            for j in range(nb):
                for k in range(nc):
                    new_frac.append(prim_frac + np.array([i, j, k]))
        all_frac = np.concatenate(new_frac)
        #coord = np.matrix(all_frac) * np.matrix(s.cell_vect)
        #coord = (coord.A+np.array(s.cell_origin)).tolist()
        for i in range(abs(na*nb*nc)):
            slat.atoms = slat.atoms + [copy.deepcopy(a) for a in s.atoms]
        slat.cell_vect = (np.matrix(trans_mat)*np.matrix(s.cell_vect)).tolist()
        slat.setter('fcoord', (np.matrix(all_frac)*np.matrix(trans_mat).I).tolist())
        slat.reset_sn()
        slat.complete_self(reset_vect=False)
        slat.prim_cell_param = s.cell_param
        slat.prim_cell_vect = s.cell_vect
        slat.trans_mat = trans_mat
        return slat

    def add_image_atom(self, expand_length):
        '''return cart coord in expand cell
        expand_length is the distance between the face of expanded cell and
        the origin cell. It can be one number or a list of three numbers.
        '''
        s = copy.deepcopy(self.S)

        def angle(a, b, c):
            '''calculate angle between a and norm of bc plane'''
            d = np.cross(b, c)
            len_a = np.linalg.norm(a)
            len_d = np.linalg.norm(d)
            angle = np.arccos(np.dot(a, d)/(len_a*len_d))
            return angle
        v1, v2, v3 = np.array(s.cell_vect)
        angle_abc = np.array([angle(v1, v2, v3), angle(v2, v1, v3), angle(v3, v1, v2)])
        # angle a is a to norm of bc plane; angle b is b to norm of ac plane ...
        iparam = np.abs(np.array(expand_length)/np.cos(angle_abc)/np.array(s.cell_param[:3]))
        fa, fb, fc = iparam
        ifrac = []
        iatoms = []
        isn = []
        for i, f in enumerate(s.fcoord):
            for a in range(int(np.floor(-1*fa)), int(np.ceil(1+fa))):
                for b in range(int(np.floor(-1*fb)), int(np.ceil(1+fb))):
                    for c in range(int(np.floor(-1*fc)), int(np.ceil(1+fc))):
                        new_f = np.array(f) + np.array([a, b, c])
                        if (all(new_f > iparam*-1) and all(new_f < iparam+1)  # within expaned cell
                                and (any(new_f < 0) or any(new_f >= 1))):  # not in origin cell
                            ifrac.append(new_f.tolist())
                            iatoms.append(s.atoms[i])
                            isn.append(i)
        icart = np.matrix(ifrac) * np.matrix(s.cell_vect)
        icart = (icart.A+np.array(s.cell_origin)).tolist()
        s.atoms = s.atoms + iatoms
        s.setter('fcoord', s.fcoord+ifrac)
        s.setter('sn', s.sn+isn)
        s.setter('coord', s.coord+icart)
        s.elem = s.getter('elem')
        s.atomnum = s.getter('atomnum')
        return s

    def set_plane(self):
        '''set two axis as 2D lattice. These two axis will be reassigned as a and b
        and a aixs will be aligned to x '''
        pass

    def shear_c(self):
        '''shear c axis to z direction'''
        pass


class Layer:

    def __init__(self, structure):
        self.S = structure

    def gen_layers(self, layer_th=0.1):
        '''chechk all layers in the z direction
        atoms are in the same layer if their z coordinate difference
        do not larger than layer_th'''
        s = self.S.sort_atoms(by='z')
        layers = []
        for i, c in enumerate(s.coord):
            if len(layers) == 0:
                L = Structure()
                L.cell_param = s.cell_param
                L.basename = s.basename
                L.add_atom(s.atoms[i])
                layers.append(L)
            elif abs(layers[-1].top - c[2]) < layer_th:
                layers[-1].add_atom(s.atoms[i])
            else:
                L = Structure()
                L.cell_param = s.cell_param
                L.basename = s.basename
                L.add_atom(s.atoms[i])
                layers.append(L)
        self.S.layers = layers

    def shift_btn_layer(self, z=0.1):
        '''shift button layer to z. Button layer is defined as having the largest
        distance  to the layer below it '''
        s = copy.deepcopy(self.S)
        s.L.check_layer(z_th=0.01, adist=False)
        top_idx = s.layer_data['z-dist'].idxmax()
        if top_idx < len(s.layer_data.index)-1:
            zpos = s.layer_data['z-pos'].iloc[top_idx+1]
        else:
            zpos = s.layer_data['z-pos'].iloc[0]
        dz = z - zpos
        return s.T.shift_xyz([0, 0, dz], pbc=True)

    def extract_layer(self, layer_idx):
        self.S.L.gen_layers()
        s = copy.deepcopy(self.S)
        s.atoms = []
        for i in layer_idx:
            s.atoms = s.atoms + self.S.layers[i].atoms
        s.complete_self()
        return s

    def check_layer(self, z_th=0.2, a_th=5, bw=0.2, adist=True):
        '''view the cell as layered structure and print infor of each layer
        layers' z coord speration smaller than z_th will be viewed as one layer
        interlayer atomic distance smaller than a_th will be printed
        '''
        # tobe develop for large system print layers one by one
        # for molecule system
        # this function convert atom distance dataframe to strings
        s = self.S

        def atom_dist_str(dist_df):
            count = []
            for _, df in dist_df.groupby('atoms'):
                dl = []
                for _, row in df.loc[df['dist'] < a_th].iterrows():
                    dl.append('{:.2f}'.format(row['dist'])+'*' +
                              '{:d}'.format(int(row['count'])))
                if len(dl) == 0:
                    row = df.iloc[0]
                    dl.append('{:.2f}'.format(row['dist'])+'*' +
                              '{:d}'.format(int(row['count'])))
                count.append(df['atoms'].iloc[0]+':'+','.join(dl))
            return ';'.join(count)

        # ps_up is the unit cell that on the top of the origin cell
        s_up = s.C.shift_cell_origin([0, 0, 1])
        s.L.gen_layers(layer_th=z_th)
        s_up.L.gen_layers(layer_th=z_th)
        # icz is the inter cell z distance
        icz = s_up.layers[0].button - s.layers[-1].top
        # ica is the inter cell cell atom distance
        formula = []
        dist = []
        pos = []
        thickness = []
        atom_dist = []
        for i, l in enumerate(s.layers):
            if i > 0:
                dist.append(l.button-s.layers[i-1].top)
            formula.append(l.formula)
            thickness.append(l.thick)
            pos.append(l.z)
        dist.append(icz)
        layers = pd.DataFrame({'formula': formula, 'z-dist': dist, 'z-pos': pos,
                               'thickness': thickness})
        t1 = datetime.now()
        if adist:
            # ps_ext is the unit cell with surrounding image atoms.
            s_ext = s.C.add_image_atom([10, 10, 0])
            s_ext.L.gen_layers(layer_th=z_th)
            ica = atom_dist_str(s_up.layers[0].dist_to(s_ext.layers[-1], bw=bw))
            for i, l in enumerate(s.layers):
                if i > 0:
                    atom_dist.append(atom_dist_str(l.dist_to(s_ext.layers[i-1], bw=bw)))
            atom_dist.append(ica)
            layers.loc[:, 'atom_dist'] = atom_dist
        t2 = datetime.now()
        self.S.layer_data = layers


class Molecule:
    pass


class Fragment:
    pass


class Structure:

    _elem2an = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7,
                'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13,
                'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,
                'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
                'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32,
                'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38,
                'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44,
                'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56,
                'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62,
                'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68,
                'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
                'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
                'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Bq': 0,
                }
    """element name to atomic number dictionary."""
    _LJr = {'H': 2.886, 'He': 2.362, 'Li': 2.451, 'Be': 2.745, 'B': 4.083,
            'C': 3.851, 'N': 3.660, 'O': 3.500, 'F': 3.364, 'Ne': 3.243,
            'Na': 2.983, 'Mg': 3.021, 'Al': 4.499, 'Si': 4.295, 'P': 4.147,
            'S': 4.035, 'Cl': 3.947, 'Ar': 3.868, 'K': 3.812, 'Ca': 3.399,
            'Sc': 3.295, 'Ti': 3.175, 'V': 3.144, 'Cr': 3.023, 'Mn': 2.961,
            'Fe': 2.912, 'Co': 2.872, 'Ni': 2.834, 'Cu': 3.495, 'Zn': 2.763,
            'Ga': 4.383, 'Ge': 4.280, 'As': 4.230, 'Se': 4.205, 'Br': 4.189,
            'Kr': 4.141, 'Rb': 4.114, 'Sr': 3.641, 'Y': 3.345, 'Zr': 3.124,
            'Nb': 3.165, 'Mo': 3.052, 'Tc': 2.998, 'Ru': 2.963, 'Rh': 2.929,
            'Pd': 2.899, 'Ag': 3.148, 'Cd': 2.848, 'In': 4.463, 'Sn': 4.392,
            'Sb': 4.420, 'Te': 4.470, 'I': 4.500, 'Xe': 4.404, 'Cs': 4.517,
            'Ba': 3.703, 'La': 3.522, 'Ce': 3.556, 'Pr': 3.606, 'Nd': 3.575,
            'Pm': 3.547, 'Sm': 3.520, 'Eu': 3.493, 'Gd': 3.368, 'Tb': 3.451,
            'Dy': 3.428, 'Ho': 3.409, 'Er': 3.391, 'Tm': 3.374, 'Yb': 3.355,
            'Lu': 3.640, 'Hf': 3.141, 'Ta': 3.170, 'W': 3.069, 'Re': 2.954,
            'Os': 3.120, 'Ir': 2.840, 'Pt': 2.754, 'Au': 3.293, 'Hg': 2.705,
            'Tl': 4.347, 'Pb': 4.297, 'Bi': 4.370, 'Po': 4.709, 'At': 4.750,
            'Bq': 0,
            }

    _an2elem = dict((value, key) for key, value in _elem2an.items())
    """atomic number to element name dictionary.
    iparm is three fraction number fa,fb,fc, so that image atoms are in the range
    of a*(-fa,1+fa), b*(-fb,1+fb), c*(-fc,1+fc)"""

    def __init__(self):
        self.basename = ''
        self.atoms = []  # list of atom, each atom is a defaultdict object
        self.period_flag = 0  # weather periodic structure
        self.cell_vect = []
        self.cell_param = []
        # flollowing are properties extract from self.atoms in complete_self method
        self.coord = []
        self.fcoord = []
        self.elem = []
        self.atomnum = []
        self.sn = []  # serial number of atoms. Duplicates indicate image atoms
        self.cell_origin = [0, 0, 0]
        # following are groups of tools to manipulate or analysis structure
        self.T = Transformer(self)
        self.C = CellParam(self)
        self.L = Layer(self)
        # self.M = Molecule(self)
        # self.F = Fragment(self)

    def getter(self, prop_name):
        '''get prop_name from all atoms and form a list
        also set the structure attribute
        '''
        prop = [atom[prop_name] for atom in self.atoms]
        return prop

    def setter(self, prop_name, value):
        '''set prop_name for all atoms with value'''
        if len(value) < len(self.atoms) or isinstance(value, str):
            prop_value = len(self.atoms) * [value]
        elif len(value) == len(self.atoms):
            prop_value = value
        else:
            print("Set {:s} for {:s} Error! More values({:d}) than atoms({:d})"
                  .format(prop_name, self.basename, len(value), len(self.atoms)))
        setattr(self, prop_name, prop_value)
        for i, atom in enumerate(self.atoms):
            atom[prop_name] = prop_value[i]

    def __str__(self):
        return self.formula+':{:.2f}'.format(self.z)

    def complete_self(self, wrap=True, reset_vect=True):
        '''re-Generate coords fcoords elems atomnums sn attribute from self.atoms
        reset = True means reset a to x axis and b in xy plane
        wrap = True means move atoms outside cell inside. Cannot working when atoms with
        duplicate sn exist (image atom).  
        '''
        self.coord = self.getter('coord')
        self.fcoord = self.getter('fcoord')
        self.elem = self.getter('elem')
        self.atomnum = self.getter('atomnum')
        self.sn = self.getter('sn')

        if len(self.cell_param) == 0 and len(self.cell_vect) == 0:
            self.period_flag = 0
        else:
            self.period_flag = 1
            if len(self.cell_vect) == 3 and len(self.cell_param) < 6:
                self.vect2param()
            elif len(self.cell_param) == 6 and len(self.cell_vect) < 3:
                self.param2vect()
            elif len(self.cell_param) < 6 and len(self.cell_vect) < 3:
                print('Cell_vect or cell_param is not right, the contents are:')
                print(self.cell_param)
                print(self.cell_vect)

            if reset_vect:
                self.param2vect()
            if len(self.coord[0]) == 3 and len(self.fcoord[0]) == 0:
                self.cart2frac()
            elif len(self.fcoord[0]) == 3 and len(self.coord[0]) == 0:
                self.frac2cart()
            elif len(self.fcoord[0]) == 0 and len(self.coord[0]) == 0:
                print('no coordinates found')

            if wrap and len(self.sn) == len(set(self.sn)):
                self.wrap_in_fcoord()
                self.frac2cart()

        if all(self.elem) and not all(self.atomnum):
            self.elem2an()
        elif all(self.atomnum) and not all(self.elem):
            self.an2elem()
        elif not all(self.elem) and not all(self.atomnum):
            self.atomname = self.getter('atomname')
            if all(self.atomname):
                self.name2elem()
                self.elem2an()
            else:
                print('no atoms found for {:s}'.format(self.basename))

        if not all(self.sn):
            print('Warning! Found missing atom serial number. Regenerata atom sn')
            self.reset_sn()

    def reset_sn(self):
        self.setter('sn', list(range(1, len(self.atoms)+1)))

    def elem2an(self):
        atomnum = [self._elem2an[i[0].upper()+i[1:].lower()] for i in self.elem]
        self.setter('atomnum', atomnum)

    def name2elem(self):
        cleared_elems = []
        for a in self.atomname:
            elem = ''.join([i for i in a if not i.isdigit()])
            elem = elem[0].upper()+elem[1:].lower()
            cleared_elems.append(elem)
        self.setter('elem', cleared_elems)

    def an2elem(self):
        elem = [self._an2elem[int(i)] for i in self.atomnum]
        self.setter('elem', elem)

    def param2vect(self):
        a, b, c, alpha, beta, gamma = self.cell_param
        va = [a, 0.0, 0.0]
        vb = [b*math.cos(math.radians(gamma)), b*math.sin(math.radians(gamma)), 0.0]
        angle2Y = math.acos(math.cos(math.radians(alpha))*math.cos(math.pi/2-math.radians(gamma)))
        cx = c*math.cos(math.radians(beta))
        cy = c*math.cos(angle2Y)
        cz = (c**2-cx**2-cy**2)**0.5
        vc = [cx, cy, cz]
        self.cell_vect = [va, vb, vc]

    def vect2param(self):
        va, vb, vc = self.cell_vect
        a = np.linalg.norm(va)
        b = np.linalg.norm(vb)
        c = np.linalg.norm(vc)
        alpha = np.rad2deg(np.arccos(np.dot(vc, vb)/(c*b)))
        beta = np.rad2deg(np.arccos(np.dot(va, vc)/(a*c)))
        gamma = np.rad2deg(np.arccos(np.dot(va, vb)/(a*b)))
        self.cell_param = [a, b, c, alpha, beta, gamma]

    def wrap_in_fcoord(self):
        for coord in self.fcoord:
            for i in range(len(coord)):
                if coord[i] < 0:
                    coord[i] += math.ceil(abs(coord[i]))
                if coord[i] >= 1:
                    coord[i] -= math.floor(coord[i])
        self.setter('fcoord', self.fcoord)

    def frac2cart(self):
        coord = np.matrix(self.fcoord) * np.matrix(self.cell_vect)
        coord = (coord.A+np.array(self.cell_origin)).tolist()
        self.setter('coord', coord)

    def cart2frac(self):
        coord = np.array(self.coord) - np.array(self.cell_origin)
        fcoord = (np.matrix(coord) * np.matrix(self.cell_vect).I).tolist()
        self.setter('fcoord', fcoord)

    def add_atom(self, atom):
        '''add a single atom'''
        self.atoms.append(atom)
        self.coord.append(atom['coord'])
        self.fcoord.append(atom['fcoord'])
        self.elem.append(atom['elem'])
        self.atomnum.append(atom['atomnum'])
        if atom['sn'] == '':
            atom['sn'] = max(self.sn) + 1
        self.sn.append(atom['sn'])

    def add_struc(self, struc, reset_sn=False):
        '''add structure object to the  structure
        the cell_param in structure object will be omitted'''
        s = copy.deepcopy(self)
        s.atoms = s.atoms + struc.atoms
        s.complete_self()
        if reset_sn == True:
            s.reset_sn()
        return s

    @property
    def thick(self):
        return self.top-self.button

    @property
    def z(self):
        return np.mean(np.array(self.coord).T[2])

    @property
    def top(self):
        return np.max(np.array(self.coord).T[2])

    @property
    def geom_center(self):
        x = np.mean(np.array(self.coord).T[0])
        y = np.mean(np.array(self.coord).T[1])
        z = np.mean(np.array(self.coord).T[2])
        return [x, y, z]

    @property
    def mass_center(self):
        pass

    @property
    def button(self):
        return np.min(np.array(self.coord).T[2])

    @property
    def valence_electron(self):
        nele = 0
        for d in self.atomnum:
            if d > 86:
                nele = nele + d - 86
            elif d > 54:
                nele = nele + d - 54
            elif d > 36:
                nele = nele + d - 36
            elif d > 18:
                nele = nele + d - 18
            elif d > 10:
                nele = nele + d - 10
            else:
                nele = nele + d
        return nele

    @property
    def formula(self):
        # generate formula in cell
        c = [[c, self.elem.count(c)] for c in set(self.elem)]
        sc = sorted(c, key=lambda x: self._elem2an[x[0]], reverse=True)
        formula = ''.join([i[0]+str(i[1]) for i in sc])
        return formula

    def sort_atoms(self, by='z', reset_sn=False):
        s = copy.deepcopy(self)
        indexes = list(range(len(self.atoms)))
        if by == 'z':
            indexes.sort(key=lambda x: self.coord[x][2])
        if by == 'elem':
            indexes.sort(key=self.elem.__getitem__)
        if by == 'an':
            indexes.sort(key=self.atomnum.__getitem__)
        s.atoms = list(map(self.atoms.__getitem__, indexes))
        s.complete_self()
        if reset_sn == True:
            s.reset_sn()
        return s

    def dist_to(self, struc, dist_type='normal', bw=0.2, raw=False):
        '''compute atom to atom distance from this structure to other structure.
        bw is bandwidth parameter in MeanShift cluster algorithm.
        Large bw parameter will group more inter atomic distance together.
        The default "normal" dist_type calculate the normal distance between atoms.
        If the dist_type starts with "h", means z coordinates are omitted in
        distance calculation.'''
        # comput distance matrix
        if dist_type.startswith('n'):
            dist_m = distance.cdist(self.coord, struc.coord)
        if dist_type.startswith('h'):
            dist_m = distance.cdist(np.array(self.coord)[:, [0, 1]],
                                    np.array(struc.coord)[:, [0, 1]])
        if raw == True:
            return dist_m

        sn = list(itertools.product(self.sn, struc.sn))
        atoms = [i[0]+'-'+i[1] for i in itertools.product(self.elem, struc.elem)]
        dist = dist_m.flatten()
        dist_df = pd.DataFrame({'sn': sn, 'atoms': atoms, 'dist': dist})
        df = dist_df.groupby('sn').min()  # eliminate image atoms
        if bw <= 0:
            df = df.reset_index().sort_values(['atoms', 'dist']).reset_index()[['atoms', 'dist']]
            df.insert(1, 'count', 1)
        else:
            clustering = MeanShift(bandwidth=bw).fit(df['dist'].values.reshape(-1, 1))
            df.loc[:, 'label'] = clustering.labels_
            df = df.groupby(['atoms', 'label'])['dist'].agg(['count', 'mean']).reset_index(level=1)
            df = df.reset_index().sort_values(['atoms', 'mean']).reset_index()
            df = df[['atoms', 'count', 'mean']]
            df.columns = ['atoms', 'count', 'dist']
        return df


class JunctionCompose:
    '''stack and scan'''

    def __init__(self, slat1, slat2):
        ''' slat1 is the botton cell'''
        ps = Structure()
        cp = list((np.array(slat1.cell_param) + np.array(slat2.cell_param))/2)
        ps.cell_param = cp
        bc = slat1.L.shift_btn_layer().\
            C.set_cell_param(cp[:2]+[slat1.cell_param[2]]+cp[3:], keep='frac')
        tc = slat2.L.shift_btn_layer().\
            C.set_cell_param(cp[:2]+[slat2.cell_param[2]]+cp[3:], keep='frac')
        self.bc = bc
        self.bl = bc.L.extract_layer([-1])
        self.tc = tc
        self.tl = tc.L.extract_layer([0])
        self.ps = ps

    def compose(self, zdist=3.5, vacuum=30, shift=[0, 0]):
        z_len = self.bc.thick + self.tc.thick + zdist + vacuum
        self.ps.cell_param[2] = z_len
        self.ps.basename = slat1.basename + '-' + slat2.basename
        bc = self.bc.C.set_cell_param(self.ps.cell_param, keep='cart').\
            L.shift_btn_layer(z=1.0)
        top_z = bc.top+zdist
        tc = self.tc.C.set_cell_param(self.ps.cell_param, keep='cart').\
            L.shift_btn_layer(z=top_z).\
            T.shift_xyz(shift+[0.0])
        self.ps.atoms = bc.atoms + tc.atoms
        self.ps.complete_self()
        a, b, _, _, _, g = self.ps.cell_param
        self.ps.basename += '-{:.0f}_{:.0f}_{:.0f}'.format(a, b, g)

    def scan(self, grid='0.2,0.2'):
        # def coulomb(hdist,vdist,power=1):
        #     hdist = np.array(hdist)
        #     vdist = vdist**2
        #     dist = np.square(hdist)+ vdist
        #     if power != 2:
        #         dist = np.power(dist,power/2)
        #     return np.sum(np.reciprocal(dist))
        # print(self.tc.prim_cell_vect)
        # if shift.startswith('a'):
        #     def area(a, b):
        #         a = np.array(a)
        #         b = np.array(b)
        #         return np.linalg.norm(np.cross(a, b))
        #     a,b,_ = self.bc.prim_cell_vect
        #     area_bc = area(a,b)
        #     a,b,_ = self.tc.prim_cell_vect
        #     area_tc = area(a,b)
        #     if area_bc < area_tc:
        #         shift = 'b'
        #         print('Botton layer has smaller primitive cell area (BL:{:.2f} '\
        #               'vs TL:{:.2f}) and will be scanned with '
        #               .format(area_bc,area_tc),end='')
        #     else:
        #         shift = 't'
        #         print('Top layer has smaller primitive cell area (TL:{:.2f} '\
        #               'vs BL:{:.2f}) and will be scanned with '
        #               .format(area_tc,area_bc),end='')
        # if shift.startswith('b'):
        #     print('Botton layer will be scanned with ',end='')
        #     mc,sc = self.bl,self.tl
        grid = [float(i) for i in grid.split(',')]

        def gen_mesh(grid, cv1, cv2):
            x = min(cv1[0][0], cv2[0][0])
            y = min(cv1[1][1], cv2[1][1])
            sa, sb = grid
            if sa < 1:
                na = np.ceil(x/sa)
            else:
                na = int(sa)
                sa = x/na
            if sb < 1:
                nb = np.ceil(y/sb)
            else:
                nb = int(sb)
                sb = y/nb
            mx = np.arange(na)*sa
            my = np.arange(nb)*sb
            return mx, my, int(na), int(nb)
        mx, my, na, nb = gen_mesh(grid, self.tl.prim_cell_vect, self.bl.prim_cell_vect)
        print('Top layer will be scanned with {:d} X {:d} points:'.format(na, nb))
        print('X grid: '+','.join(['{:.2f}'.format(i) for i in mx]))
        print('Y grid: '+','.join(['{:.2f}'.format(i) for i in my]))
        bl = self.bl.C.add_image_atom([10, 10, 0])
        alldist = []
        for i, x in enumerate(mx):
            for j, y in enumerate(my):
                tl = self.tl.T.shift_xyz([x, y, 0])
                dm = tl.dist_to(bl, dist_type='h', bw=0, raw=True)
                alldist.append([dm, x, y, np.min(dm)])

        alldist = sorted(alldist, key=lambda x: x[-1], reverse=True)
        r1 = self.bl._LJr[self.bl.elem[0]]
        r2 = self.tl._LJr[self.tl.elem[0]]
        r0 = (r1+r2)/2
        zdist = ((r0/2**(1/6))**2-alldist[0][-1]**2)**0.5 - 0.2
        self.zdist = zdist+0.2
        dist = []
        for i, d in enumerate(alldist):
            dm = d[0]
            r = zdist**2 + dm**2
            vdw = np.sum((r0**2/r)**6-2*(r0**2/r)**3)
            dist.append([d[1], d[2], d[3], vdw])
        dist = sorted(dist, key=lambda x: x[-1])
        return dist


parser = argparse.ArgumentParser(description='Build a two layer junction using '
                                 'two layer structure file. The symmetry '
                                 'of input structure should be maked P1. c axis should be the vacuum '
                                 'direction. Supported file formats are cif, res, xsf, pdb')
parser.add_argument(
    '-t', dest='tasktype', metavar='Task type', default='i', type=str,
    help='set the task type; '
    'i for inspect: inspect structure and convert file format.  '
    'il include layer data, ia include all infor; '
    'm for match and merge: lattice match and merge two layerd structure. '
    'First structure will be the botton layer; '
    's for slice: slice bulk structure to layered structure; '
    'ss for slice surface: slice surface. e.g. ss2,5 will generate 3 layered structrure, sliced between layer 2,3 and 5,6; '
    'p for place: place molecules on top of the slab. First structrue is '
    'slab and second structure is molecule. '
    'Default job is inspect')
parser.add_argument('-s', dest='sort', metavar='Sort match result',
                    default='csh', type=str,
                    help='Match task param. '
                    'Set how to sort match result. c stands for score; s stands for size, '
                    'h stands for shear, a stands for devidation from target angle, '
                    'Default is csh, which means sort by score first, then by size and shear.')
parser.add_argument('-c', dest='choose', metavar='Choose match cell',
                    default=None, type=int,
                    help='Merge task param. '
                    'Choose match pairs to Merge')
parser.add_argument('-v', dest='vacuum', metavar='Vacuum height',
                    default=30, type=int,
                    help='Merge task param. '
                    'Set thickness or vacuum layer in Angstrom. Default is 30 ')
parser.add_argument('-p', dest='place', metavar='How to place mol',
                    default='zd4', type=str,
                    help='Place task param. '
                    'Set x y z r of placed molecule. '
                    'x[fc] y[fc] z[fc] are absolute postion of the mol center in frac/cart coord'
                    ', and z[dD] is the mol btn/center distance to the surface top. '
                    'If not set, xf,yf or zd value will be generate randomly. '
                    'r can be set to 000 (not rotate) or 111 (default, random rotate around xyz). '
                    'format is like "xf0.5,yf0.5,zd3.5,r111", default is "zd4"')
parser.add_argument('-d', dest='displace', metavar='Top layer displace',
                    default='vdw', type=str,
                    help='Merge task param. '
                    'Set how the top layer is displaced horizontally when two layers '
                    'are combined. "no" means do not displace; "max" means displace top '
                    'layer to max vdw score (Default). "full" '
                    'means output all the scanned structures (use --grid option ot set '
                    'number of scanned structure)')
parser.add_argument('-z', dest='zdist', metavar='Interlayer dist',
                    default='auto', type=str,
                    help='Merge task param. '
                    'set distance between two layers. Default is auto set according to UFF ')
parser.add_argument('-o', dest='output', metavar='Output filename.format',
                    default=None, type=str,
                    help='Set the output file name and format, default is .pdb. '
                    'Filename:String before dot or without dot. Format:String after last dot. '
                    'If file name is not set, input structure name will be used.'
                    'Supported formats are .res .cif .xsf .pdb .')
parser.add_argument('--nprint', metavar='No. matched cell to print',
                    default=20, type=int,
                    help='Match task param. '
                    'Set the number of lattice match to print, default is 20.')
parser.add_argument('--grid', metavar='Scan grid space',
                    default='0.2,0.2', type=str,
                    help='Merge task param. '
                    'Set two float numbers seperate by comma to assign scan grid '
                    'space in x and y, or two integers to assign grid numbers. '
                    'Default is 0.2,0.2.')
parser.add_argument('--ts', metavar='Match score tolerance',
                    type=str, default='5',
                    help='Match task param. '
                    'Set the tolerance of match score. '
                    'Match score is the RMS of error of edge, area, and angle times 100 ')
parser.add_argument('--ta', metavar='Cell angle tolerance',
                    default='90,62', type=str,
                    help='Match task param. '
                    'Set the tolerance of super lattice angle range. Default is 90,62 '
                    'which means target angle is 90 and angle range between 90+62 or 90-62 '
                    'is allowed. The angle close to the target can be sorted to top '
                    'by -s a option')
parser.add_argument('--tc', metavar='Super cell size tolerance',
                    default=0, type=int,
                    help='Match task param. '
                    'Set the maximun allowed super lattice size. '
                    'Default is the square of max_edge value')
parser.add_argument('--max_edge', metavar='Max cell edge size explored',
                    default=10, type=int,
                    help='Match task param. '
                    'Set the maxinum super lattice edge size explored. Default is 10, '
                    'which means max super lattice is u=10a, v=10b')
parser.add_argument('--max_shear', metavar='Max shear number explored',
                    default=0, type=int,
                    help='Match task param. '
                    'Set the maximun shear number explored. Default is 2*max_edge, '
                    'if this number is 20, then the max sheared lattice is u=a+20b,v=b ')
parser.add_argument('--zth', metavar='Layer thickness',
                    default=0.2, type=float,
                    help='Inspect task param. '
                    'Atom z distance smaller than zth will be in the same layer')
parser.add_argument('--ath', metavar='Atom distance threshold',
                    default=5, type=float,
                    help='Inspect task param. '
                    'Atom distance smaller than ath will be printed. '
                    'At least one smallest distance will be printed. Default is 5.0 ')
parser.add_argument('--bandwidth', metavar='Group interlayer atom disance by bandwith',
                    default=0.2, type=float,
                    help='Inspect task param. '
                    'When printing atom_dist, larger bandwidth will grounp more '
                    'different distance together and the average distance will be printed')
parser.add_argument('--tol_es', type=str, default='0.1,0.1',
                    help='Match task param. '
                    'Set two float number for tolerance of edge and area. '
                    'Default value is 0.1,0.1 '
                    'Error of edge is calculated by (a1-a2)/(a1+a2). '
                    'Error of area is calculated by ((s1-s2)/(s1+s2))^2. ')

parser.add_argument('inputfile', nargs='+',
                    help='the input format could be .res .xsf .cif')
args = parser.parse_args()
tol = args.tol_es+','+str(float(args.ts)/100)
if args.max_shear == 0:
    args.max_shear = args.max_edge * 2
edge_size = ','.join(str(i) for i in [args.max_edge, args.max_shear])
if args.tc == 0:
    args.tc = args.max_edge ** 2
# parse args.output


def write_file(struc, output=args.output, append='', default=None):
    '''if output is none then use default
    if no extension is defined then use pdb'''
    if default is not None and output is None:
        output = default
    if output is not None:
        ext = 'pdb'
        flnm = ''
        if '.' in output:
            flnm = '.'.join(output.split('.')[:-1])
            ext = output.split('.')[-1]
        else:
            flnm = output
    else:
        flnm = None
        print('No filename defined, no file is generated.')
    if flnm is not None:
        if flnm == '':
            try:
                flnm = struc.basename
                sw = StructureWriter()
                sw.write_file(struc, basename=flnm+append, ext=ext)
            except AttributeError:
                print('Structure object error, no file is generated')



def print_info(st):
    if st.period_flag == 1:
        print('{:24s}Periodic sturcture'.format(st.basename+':'))
        print('Cell Parameter:         a={:.3f},b={:.3f},c={:.3f},alpha={:.3f},beta={:.3f},gamma={:.3f}'
              .format(*st.cell_param))
    elif st.period_flag == 0:
        print('{:s}: Non-periodic sturctured'.format(st.basename))
    print('{:24s}{:s}'.format('Formula: ', st.formula))
    print('{:24s}{:<d}'.format('No. of atoms:', len(st.atoms)))
    print('{:24s}{:<d}/{:<d}'
          .format('No. of Tot/Val elec:', sum(st.atomnum), st.valence_electron))
    keys = ' '.join(sorted([k for k, v in st.atoms[0].items() if v != '']))
    print('{:24s}{:s}'.format('Properties available:', keys))


if args.tasktype == 'm':
    sr = StructureReader()
    ps1 = sr.read_struc(args.inputfile[0])
    ps2 = sr.read_struc(args.inputfile[1])

    lm = LatticeMatch(ps1, ps2, tol=tol, size=edge_size, angle_range=args.ta, size_tol=args.tc)
    lm.render_data(sort_str=args.sort, num_print=args.nprint)

    def parse_trans_mat(trans_mat):
        n1, n2, n3, swith = trans_mat
        full_mat = [[n1, 0, 0], [n3, n2, 0], [0, 0, 1]]
        if swith == 'ba':
            order = [1, 0, 2]
        else:
            order = [0, 1, 2]
        return (order, full_mat)

    if args.choose is not None:
        print('Matching pair index {:d} selected'.format(args.choose))
        od1, tm1 = parse_trans_mat(lm.data[args.choose]['trans_mat1'])
        od2, tm2 = parse_trans_mat(lm.data[args.choose]['trans_mat2'])
        slat1 = ps1.C.switch_edge(order=od1).C.super_cell(tm1)
        slat2 = ps2.C.switch_edge(order=od2).C.super_cell(tm2)
        print_info(slat1)
        print_info(slat2)
        jc = JunctionCompose(slat1, slat2)
        if args.displace == 'no':
            jc.compose(zdist=args.zdist, vacuum=args.vacuum)
            write_file(jc.ps, default='.res')
        if args.displace == 'vdw':
            dist = jc.scan(args.grid)
            shift = dist[0][0:2]
            print('Mininum VDW score {:.3f} found when shifting '
                  'top layer {:.2f},{:.2f}'.format(dist[0][-1], *shift))
            if args.zdist == 'auto':
                zdist = jc.zdist
                print('Interlayer distance set to {:.3f} Angstrom automatic'.format(zdist))
            else:
                zdist = float(args.zdist)
                print('Interlayer distance set to {:.3f} Angstrom by user'.format(zdist))
            jc.compose(zdist=zdist, vacuum=args.vacuum, shift=shift)
            flnm = jc.ps.basename + '-{:.1f}_{:.1f}'.format(*shift)
            print_info(jc.ps)
            write_file(jc.ps, default=flnm+'.res')
        if args.displace == 'all':
            dist = jc.scan(args.grid)
            if args.zdist == 'auto':
                zdist = jc.zdist
                print('Interlayer distance set to {:.3f} Angstrom automatic'.format(zdist))
            else:
                zdist = float(args.zdist)
                print('Interlayer distance set to {:.3f} Angstrom by user'.format(zdist))
            for i in dist:
                shift = i[1:3]
                print('VDW score: {:3f} when shifting top layer {:.2f},{:.2f}'
                      .format(i[0], *shift))
                jc.compose(zdist=zdist, vacuum=args.vacuum, shift=shift)
                shift_str = '-{:.1f}_{:.1f}'.format(*shift)
                print_info(jc.ps)
                write_file(jc.ps, default=flnm+'.')


if args.tasktype.startswith('i'):
    sr = StructureReader()
    for i in args.inputfile:
        s1 = sr.read_struc(i)
        print_info(s1)
        if 'l' in args.tasktype:
            print('Checking layers in the cell ...')
            s1.L.check_layer(z_th=args.zth, a_th=args.ath, bw=args.bandwidth, adist=False)
            print(s1.layer_data.to_string())
        if 'a' in args.tasktype:
            for a in s1.atoms:
                print(dict(a))
        write_file(s1)

def parse_idx(idx_str):
    sub_str = idx_str.split(',')
    idx_list = []
    for i,s in enumerate(sub_str):
        if '-' in s:
            m=int(s.split('-')[0])
            n=int(s.split('-')[1])
            idx_list = idx_list + list(range(m,n+1))
        else:
            try:
                idx_list.append(int(s))
            except ValueError:
                pass
    return idx_list


if args.tasktype.startswith('ss'):
    layer_idx = [parse_idx(i) for i in args.tasktype[2:].split(';')]
    sr = StructureReader()
    s1 = sr.read_struc(args.inputfile[0])
    print_info(s1)
    print('Checking layers in the cell ...')
    s1.L.check_layer(z_th=args.zth, a_th=args.ath, bw=args.bandwidth, adist=False)
    print(s1.layer_data.to_string())
    for lay in layer_idx: 
        ls = None
        for idx, layer in enumerate(s1.layers):
            if idx in lay:
                if ls is None:
                    ls = layer
                    i=idx
                else:
                    ls = ls.add_struc(layer)  # add to existing structure
        write_file(ls, append='_'+str(i), default='.res')

 #   write_file(s1)


if args.tasktype == 'p':
    def parser_place_param(place_str, ps, mol):
        place_param = ['xf', 'xc', 'yf', 'yc', 'zf', 'zc', 'zd', 'zD', 'r']
        param_dict = {}
        for p in place_str.strip('\'').strip('\"').split(','):
            for k in place_param:
                if p.startswith(k):
                    v = p[len(k):]
                    param_dict[k] = v
        cart_list = []
        if not any([i.startswith('x') for i in param_dict.keys()]):
            param_dict['xf'] = np.random.rand()
        if not any([i.startswith('y') for i in param_dict.keys()]):
            param_dict['yf'] = np.random.rand()
        if not any([i.startswith('z') for i in param_dict.keys()]):
            param_dict['zf'] = np.random.rand()
        if 'r' not in param_dict:
            param_dict['r'] = '000'
        if 'xf' in param_dict.keys():
            x = float(param_dict['xf']) * ps.cell_vect[0][0]
            param_dict['xc'] = x
        if 'yf' in param_dict.keys():
            y = float(param_dict['yf']) * ps.cell_vect[1][1]
            param_dict['yc'] = y
        angles = []
        for i in param_dict['r']:
            if i != '1':
                d = 0
            else:
                d = np.random.rand()*360
            angles.append(d)
        param_dict['r'] = angles
        if 'zc' in param_dict.keys():
            param_dict['zc'] = float(param_dict['zc'])
        if 'zf' in param_dict.keys():
            z = float(param_dict['zf']) * ps.cell_vect[2][2]
            param_dict['zc'] = z
        if 'zd' in param_dict.keys():
            z = ps.top+float(param_dict['zd'])+mol.geom_center[2]-mol.button
            param_dict['zc'] = z
        if 'zD' in param_dict.keys():
            z = ps.top+float(param_dict['zD'])
            param_dict['zc'] = z
        return [[param_dict['xc'], param_dict['yc'], param_dict['zc']], angles]

    sr = StructureReader()
    ps = sr.read_struc(args.inputfile[0])
    print_info(ps)
    mol = sr.read_struc(args.inputfile[1])
    mol = mol.C.delete_cell()
    print_info(mol)
    ps.param2vect()  # reset vect to align a to x
    target_xyz, rot_angle = parser_place_param(args.place, ps, mol)
    mol = mol.T.shift_xyz(np.array(target_xyz)-np.array(mol.geom_center))
    mol = mol.T.rotate(rot_angle)
    print('Top position of surface is {:.2f} Angstrom'.format(ps.top))
    print('Roate molecule around x:{:.2f} y:{:.2f} z:{:.2f}'.format(*rot_angle))
    print('Place molecule at x:{:.3f} y:{:.3f} z:{:.3f}'.format(*target_xyz))
    ps = ps.add_struc(mol, reset_sn=True)
    write_file(ps, default=ps.basename+'-'+mol.basename+'.pdb')
