# sparse_kmedoids

[![Build Status](https://travis-ci.com/kmayerb/sparse_kmedoids.svg?branch=master)](https://travis-ci.com/kmayerb/sparse_kmedoids)
[![PyPI version](https://badge.fury.io/py/sparse_kmedoids.svg)](https://badge.fury.io/py/sparse_kmedoids)
[![Coverage Status](https://coveralls.io/repos/github/agartland/sparse_kmedoids/badge.svg?branch=master)](https://coveralls.io/github/agartland/sparse_kmedoids?branch=master)

## SPARSE_KMEDOIDS NOT WORKING
While the non-sparse k-medoid and fuzzy c-medoid codes work well, the sparse_kmedoid code is only a draft and does not work. It seems like it may be close, but had to abandon for now. Also, the sparse numba implementation is not fast, which could be related to the bugs or could be related to the way it is implemented.

Implementation of k-medoids clustering (PAM) that can accept a `scipy.sparse.csr_matrix` distance matrix. The sparse implementation uses numba for efficiency. Also includes a numpy, non-sparse version for testing and smaller datasets.

## Install

```
pip install sparse_kmedoids
```

## Example

```python
import sparse_kmedoids

[test]
```

```
[result]
```
