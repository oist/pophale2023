cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow
from libc.stdlib cimport malloc, free
from cython.parallel import parallel, prange

#
# (C) Copyright 2015 Frankfurt Institute for Advanced Studies
# (C) Copyright 2016 Max-Planck Institute for Brain Research
#
# Author: Philipp Huelsdunk  <huelsdunk@fias.uni-frankfurt.de>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the author nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#     * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#     * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#     * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

cdef float flt_eps = np.finfo(np.float32).eps

@cython.boundscheck(False)
cdef void __similarity_detail(float[:, :] p, \
        float[:, :] q, \
        float vx, float vy, \
        float *fx, float *fy, \
        float alpha, \
        float *w) nogil:
    cdef size_t num_p = p.shape[0]
    cdef size_t j

    # Calculate weights
    cdef float w_sum = 0
    cdef int near_p = 0
    cdef float dist
    for j in range(num_p):
        dist = sqrt((vx - p[j, 0]) * (vx - p[j, 0]) \
                  + (vy - p[j, 1]) * (vy - p[j, 1]))
        if not dist > flt_eps:
            fx[0] = q[j, 0]
            fy[0] = q[j, 1]
            near_p = 1
            break
        w[j] = pow(dist, -alpha)
        w_sum += w[j]
    if near_p:
        return

    # Weighted interpolation points
    cdef float[2] p_star
    p_star[0] = 0
    p_star[1] = 0
    for j in range(num_p):
        p_star[0] += w[j] * p[j, 0]
        p_star[1] += w[j] * p[j, 1]
    p_star[0] /= w_sum
    p_star[1] /= w_sum
    cdef float[2] q_star
    q_star[0] = 0
    q_star[1] = 0
    for j in range(num_p):
        q_star[0] += w[j] * q[j, 0]
        q_star[1] += w[j] * q[j, 1]
    q_star[0] /= w_sum
    q_star[1] /= w_sum

    # From (6 + 7)
    cdef float mus = 0
    cdef float[2] f_tmp
    cdef float[2] p_hat
    cdef float[2] q_hat
    cdef float[2] v_hat
    f_tmp[0] = 0
    f_tmp[1] = 0
    for j in range(num_p):
        # From (4)
        p_hat[0] = p[j, 0] - p_star[0]
        p_hat[1] = p[j, 1] - p_star[1]
        q_hat[0] = q[j, 0] - q_star[0]
        q_hat[1] = q[j, 1] - q_star[1]
        
        f_tmp[0] += (q_hat[0] * p_hat[0] + q_hat[1] * p_hat[1]) * w[j]
        f_tmp[1] += (q_hat[0] * p_hat[1] - q_hat[1] * p_hat[0]) * w[j]

        # From (6)
        mus = mus + w[j] * (p_hat[0] * p_hat[0] + p_hat[1] * p_hat[1])
    
    v_hat[0] = vx - p_star[0]
    v_hat[1] = vy - p_star[1]

    fx[0] = (f_tmp[0] * v_hat[0] + f_tmp[1] * v_hat[1]) / mus + q_star[0]
    fy[0] = (f_tmp[0] * v_hat[1] - f_tmp[1] * v_hat[0]) / mus + q_star[1]

@cython.boundscheck(False)
cdef void __similarity(float[:, :] p, \
        float[:, :] q, \
        float[:, :] v, \
        float[:, :] f, \
        float alpha=1.) nogil:

    cdef size_t num_v = v.shape[0]
    cdef int i
    cdef float *w
    
    with parallel():
        w = <float *>malloc(p.shape[0] * sizeof(float))
        for i in prange(num_v, schedule='static'):
            __similarity_detail(p, q, \
                     v[<size_t>(i), 0],  v[<size_t>(i), 1], \
                    &f[<size_t>(i), 0], &f[<size_t>(i), 1], \
                    alpha, w)
        free(w)

cpdef similarity(float[:, :] p, \
        float[:, :] q, \
        float[:, :] v, \
        float alpha=1.):
    assert p.shape[1] == 2
    assert q.shape[1] == 2
    assert p.shape[0] == q.shape[0]
    assert v.shape[1] == 2
    cdef np.ndarray[float, ndim=2] f = np.empty([v.shape[0], 2], \
            dtype=np.float32)
    __similarity(p, q, v, f, alpha)
    return f
    
