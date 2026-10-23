# Third-party notices and scientific sources

These sources define numerical reference behavior and data provenance.
Upstream notices are preserved below. Numerical parity does not by itself
establish whether source code was copied or translated. These notices do not
claim an exhaustive authorship audit. The project license does not relicense
upstream material or pretrained model weights. No pretrained weights are
included in this repository.

## Numerical references

- [e3nn 0.6.0](https://github.com/e3nn/e3nn/tree/2aa7f58440a06b15352a2cbce01fa4c26f824969)
  (MIT): spherical harmonics, tensor products, and convention fixtures.
- [MACE 0.3.16](https://github.com/ACEsuit/mace/tree/4d2da09413ac1407f37cdbb6b81fa28e4c15655e)
  (MIT): `mace_*` radial behavior, Agnesi transform, ZBL constants, and
  `MACE_COVALENT_RADII` numerical parity.
- [fairchem](https://github.com/FAIR-Chem/fairchem/tree/977a80328f2be44649b414a9907a1d6ef2f81e95)
  (MIT): eSEN/UMA conventions targeted by `fairchem_esen_*` kernels. This is
  the development reference revision for the license below, not a claim of
  bundled upstream-generated fairchem fixtures or pretrained weights.

## Scientific and data sources

- J. F. Ziegler, J. P. Biersack and U. Littmark, *The Stopping and Range of
  Ions in Solids*, Pergamon Press (1985): ZBL screening model. Constants here
  follow MACE's parameterization, including its screening-length parameters.
- B. Cordero et al., [Covalent radii revisited](https://doi.org/10.1039/B801115J),
  Dalton Transactions (2008), 2832–2838. The immediate reference for the
  119-entry radii table is MACE's use of
  [ASE's covalent radii](https://ase.gitlab.io/ase/ase/data.html), including
  placeholder values; not all entries are measured values from the paper.
  ASE is a separate LGPL-licensed project, not an MIT project.
- [ACEpotentials.jl, JCP (2023)](https://doi.org/10.1063/5.0158783): radial
  transformations, as cited by MACE's `AgnesiTransform`. This crate targets
  MACE's numerical defaults and derivative behavior.

## Upstream license texts

These texts are preserved from the reference revisions above as attribution;
they do not imply affiliation or endorsement.

### e3nn

```text
MIT License

Euclidean neural networks (e3nn) Copyright (c) 2020, The Regents of the
University of California, through Lawrence Berkeley National Laboratory
(subject to receipt of any required approvals from the U.S. Dept. of Energy),
Ecole Polytechnique Federale de Lausanne (EPFL), Free University of Berlin
and Kostiantyn Lapchevskyi. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MACE

```text
MIT License

Copyright (c) 2022 ACEsuit/mace

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

### fairchem

```text
# License

MIT License

Copyright (c) Meta, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
