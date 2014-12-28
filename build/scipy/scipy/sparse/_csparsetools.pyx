# -*- cython -*-
#
# Tempita-templated Cython file
#
"""
Fast snippets for sparse matrices.
"""

cimport cython
cimport cpython.list
cimport cpython.int
cimport cpython
cimport numpy as cnp
import numpy as np


def prepare_index_for_memoryview(cnp.ndarray i, cnp.ndarray j, cnp.ndarray x=None):
    """
    Convert index and data arrays to form suitable for passing to the
    Cython fancy getset routines.

    The conversions are necessary since to (i) ensure the integer
    index arrays are in one of the accepted types, and (ii) to ensure
    the arrays are writable so that Cython memoryview support doesn't
    choke on them.

    Parameters
    ----------
    i, j
        Index arrays
    x : optional
        Data arrays

    Returns
    -------
    i, j, x
        Re-formatted arrays (x is omitted, if input was None)

    """
    if i.dtype > j.dtype:
        j = j.astype(i.dtype)
    elif i.dtype < j.dtype:
        i = i.astype(j.dtype)

    if not i.flags.writeable or not i.dtype in (np.int32, np.int64):
        i = i.astype(np.intp)
    if not j.flags.writeable or not j.dtype in (np.int32, np.int64):
        j = j.astype(np.intp)

    if x is not None:
        if not x.flags.writeable:
            x = x.copy()
        return i, j, x
    else:
        return i, j


cpdef lil_get1(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
               cnp.npy_intp i, cnp.npy_intp j):
    """
    Get a single item from LIL matrix.

    Doesn't do output type conversion. Checks for bounds errors.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get

    Returns
    -------
    x
        Value at indices.

    """
    cdef list row, data

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]
    pos = bisect_left(row, j)

    if pos != len(data) and row[pos] == j:
        return data[pos]
    else:
        return 0


def lil_insert(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
               cnp.npy_intp i, cnp.npy_intp j, object x, object dtype):
    return _LIL_INSERT_DISPATCH[dtype](M, N, rows, datas, i, j, x)

cpdef _lil_insert_int32(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
                           cnp.npy_intp i, cnp.npy_intp j, cnp.npy_int32 x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data
    cdef int is_zero

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)
cpdef _lil_insert_int16(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
                           cnp.npy_intp i, cnp.npy_intp j, cnp.npy_int16 x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data
    cdef int is_zero

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)
cpdef _lil_insert_bool_(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
                           cnp.npy_intp i, cnp.npy_intp j, cnp.npy_bool x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data
    cdef int is_zero

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)
cpdef _lil_insert_longdouble(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
                           cnp.npy_intp i, cnp.npy_intp j, long double x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data
    cdef int is_zero

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)
cpdef _lil_insert_uint8(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
                           cnp.npy_intp i, cnp.npy_intp j, cnp.npy_uint8 x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data
    cdef int is_zero

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)
cpdef _lil_insert_complex64(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
                           cnp.npy_intp i, cnp.npy_intp j, float complex x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data
    cdef int is_zero

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)
cpdef _lil_insert_int8(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
                           cnp.npy_intp i, cnp.npy_intp j, cnp.npy_int8 x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data
    cdef int is_zero

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)
cpdef _lil_insert_uint64(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
                           cnp.npy_intp i, cnp.npy_intp j, cnp.npy_uint64 x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data
    cdef int is_zero

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)
cpdef _lil_insert_float64(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
                           cnp.npy_intp i, cnp.npy_intp j, cnp.npy_float64 x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data
    cdef int is_zero

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)
cpdef _lil_insert_complex128(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
                           cnp.npy_intp i, cnp.npy_intp j, double complex x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data
    cdef int is_zero

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)
cpdef _lil_insert_uint16(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
                           cnp.npy_intp i, cnp.npy_intp j, cnp.npy_uint16 x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data
    cdef int is_zero

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)
cpdef _lil_insert_int64(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
                           cnp.npy_intp i, cnp.npy_intp j, cnp.npy_int64 x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data
    cdef int is_zero

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)
cpdef _lil_insert_clongdouble(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
                           cnp.npy_intp i, cnp.npy_intp j, long double complex x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data
    cdef int is_zero

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)
cpdef _lil_insert_uint32(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
                           cnp.npy_intp i, cnp.npy_intp j, cnp.npy_uint32 x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data
    cdef int is_zero

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)
cpdef _lil_insert_float32(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
                           cnp.npy_intp i, cnp.npy_intp j, cnp.npy_float32 x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data
    cdef int is_zero

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)


cdef dict _LIL_INSERT_DISPATCH = {

np.dtype(np.int32): _lil_insert_int32,
np.dtype(np.int16): _lil_insert_int16,
np.dtype(np.bool_): _lil_insert_bool_,
np.dtype(np.longdouble): _lil_insert_longdouble,
np.dtype(np.uint8): _lil_insert_uint8,
np.dtype(np.complex64): _lil_insert_complex64,
np.dtype(np.int8): _lil_insert_int8,
np.dtype(np.uint64): _lil_insert_uint64,
np.dtype(np.float64): _lil_insert_float64,
np.dtype(np.complex128): _lil_insert_complex128,
np.dtype(np.uint16): _lil_insert_uint16,
np.dtype(np.int64): _lil_insert_int64,
np.dtype(np.clongdouble): _lil_insert_clongdouble,
np.dtype(np.uint32): _lil_insert_uint32,
np.dtype(np.float32): _lil_insert_float32,
}




def lil_fancy_get(cnp.npy_intp M, cnp.npy_intp N,
                  object[:] rows,
                  object[:] datas,
                  object[:] new_rows,
                  object[:] new_datas,
                  cnp.ndarray i_idx,
                  cnp.ndarray j_idx):
    """
    Get multiple items at given indices in LIL matrix and store to
    another LIL.

    Parameters
    ----------
    M, N, rows, data
        LIL matrix data, initially empty
    new_rows, new_idx
        Data for LIL matrix to insert to.
        Must be preallocated to shape `i_idx.shape`!
    i_idx, j_idx
        Indices of elements to insert to the new LIL matrix.

    """
    return _LIL_FANCY_GET_DISPATCH[i_idx.dtype](M, N, rows, datas, new_rows, new_datas, i_idx, j_idx)

def _lil_fancy_get_int32(cnp.npy_intp M, cnp.npy_intp N,
                            object[:] rows,
                            object[:] datas,
                            object[:] new_rows,
                            object[:] new_datas,
                            cnp.npy_int32[:,:] i_idx,
                            cnp.npy_int32[:,:] j_idx):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j
    cdef object value
    cdef list new_row
    cdef list new_data

    for x in range(i_idx.shape[0]):
        new_row = []
        new_data = []

        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]

            value = lil_get1(M, N, rows, datas, i, j)

            if value is not 0:
                # Object identity as shortcut
                new_row.append(y)
                new_data.append(value)

        new_rows[x] = new_row
        new_datas[x] = new_data
def _lil_fancy_get_int64(cnp.npy_intp M, cnp.npy_intp N,
                            object[:] rows,
                            object[:] datas,
                            object[:] new_rows,
                            object[:] new_datas,
                            cnp.npy_int64[:,:] i_idx,
                            cnp.npy_int64[:,:] j_idx):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j
    cdef object value
    cdef list new_row
    cdef list new_data

    for x in range(i_idx.shape[0]):
        new_row = []
        new_data = []

        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]

            value = lil_get1(M, N, rows, datas, i, j)

            if value is not 0:
                # Object identity as shortcut
                new_row.append(y)
                new_data.append(value)

        new_rows[x] = new_row
        new_datas[x] = new_data


cdef dict _LIL_FANCY_GET_DISPATCH = {

np.dtype(np.int32): _lil_fancy_get_int32,
np.dtype(np.int64): _lil_fancy_get_int64,
}




def lil_fancy_set(cnp.npy_intp M, cnp.npy_intp N,
                  object[:] rows,
                  object[:] data,
                  cnp.ndarray i_idx,
                  cnp.ndarray j_idx,
                  cnp.ndarray values):
    """
    Set multiple items to a LIL matrix.

    Checks for zero elements and deletes them.

    Parameters
    ----------
    M, N, rows, data
        LIL matrix data
    i_idx, j_idx
        Indices of elements to insert to the new LIL matrix.
    values
        Values of items to set.

    """
    if values.dtype == np.bool_:
        # Cython doesn't support np.bool_ as a memoryview type
        return _lil_fancy_set_generic(M, N, rows, data, i_idx, j_idx, values)
    else:
        return _LIL_FANCY_SET_DISPATCH[i_idx.dtype, values.dtype](M, N, rows, data, i_idx, j_idx, values)

cpdef _lil_fancy_set_int32_int32(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int32[:,:] i_idx,
                                           cnp.npy_int32[:,:] j_idx,
                                           cnp.npy_int32[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_int32(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int32_int16(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int32[:,:] i_idx,
                                           cnp.npy_int32[:,:] j_idx,
                                           cnp.npy_int16[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_int16(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int32_bool_(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int32[:,:] i_idx,
                                           cnp.npy_int32[:,:] j_idx,
                                           cnp.npy_bool[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_bool_(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int32_longdouble(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int32[:,:] i_idx,
                                           cnp.npy_int32[:,:] j_idx,
                                           long double[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_longdouble(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int32_uint8(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int32[:,:] i_idx,
                                           cnp.npy_int32[:,:] j_idx,
                                           cnp.npy_uint8[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_uint8(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int32_complex64(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int32[:,:] i_idx,
                                           cnp.npy_int32[:,:] j_idx,
                                           float complex[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_complex64(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int32_int8(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int32[:,:] i_idx,
                                           cnp.npy_int32[:,:] j_idx,
                                           cnp.npy_int8[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_int8(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int32_uint64(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int32[:,:] i_idx,
                                           cnp.npy_int32[:,:] j_idx,
                                           cnp.npy_uint64[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_uint64(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int32_float64(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int32[:,:] i_idx,
                                           cnp.npy_int32[:,:] j_idx,
                                           cnp.npy_float64[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_float64(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int32_complex128(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int32[:,:] i_idx,
                                           cnp.npy_int32[:,:] j_idx,
                                           double complex[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_complex128(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int32_uint16(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int32[:,:] i_idx,
                                           cnp.npy_int32[:,:] j_idx,
                                           cnp.npy_uint16[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_uint16(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int32_int64(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int32[:,:] i_idx,
                                           cnp.npy_int32[:,:] j_idx,
                                           cnp.npy_int64[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_int64(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int32_clongdouble(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int32[:,:] i_idx,
                                           cnp.npy_int32[:,:] j_idx,
                                           long double complex[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_clongdouble(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int32_uint32(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int32[:,:] i_idx,
                                           cnp.npy_int32[:,:] j_idx,
                                           cnp.npy_uint32[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_uint32(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int32_float32(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int32[:,:] i_idx,
                                           cnp.npy_int32[:,:] j_idx,
                                           cnp.npy_float32[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_float32(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int64_int32(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int64[:,:] i_idx,
                                           cnp.npy_int64[:,:] j_idx,
                                           cnp.npy_int32[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_int32(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int64_int16(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int64[:,:] i_idx,
                                           cnp.npy_int64[:,:] j_idx,
                                           cnp.npy_int16[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_int16(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int64_bool_(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int64[:,:] i_idx,
                                           cnp.npy_int64[:,:] j_idx,
                                           cnp.npy_bool[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_bool_(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int64_longdouble(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int64[:,:] i_idx,
                                           cnp.npy_int64[:,:] j_idx,
                                           long double[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_longdouble(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int64_uint8(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int64[:,:] i_idx,
                                           cnp.npy_int64[:,:] j_idx,
                                           cnp.npy_uint8[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_uint8(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int64_complex64(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int64[:,:] i_idx,
                                           cnp.npy_int64[:,:] j_idx,
                                           float complex[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_complex64(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int64_int8(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int64[:,:] i_idx,
                                           cnp.npy_int64[:,:] j_idx,
                                           cnp.npy_int8[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_int8(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int64_uint64(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int64[:,:] i_idx,
                                           cnp.npy_int64[:,:] j_idx,
                                           cnp.npy_uint64[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_uint64(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int64_float64(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int64[:,:] i_idx,
                                           cnp.npy_int64[:,:] j_idx,
                                           cnp.npy_float64[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_float64(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int64_complex128(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int64[:,:] i_idx,
                                           cnp.npy_int64[:,:] j_idx,
                                           double complex[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_complex128(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int64_uint16(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int64[:,:] i_idx,
                                           cnp.npy_int64[:,:] j_idx,
                                           cnp.npy_uint16[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_uint16(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int64_int64(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int64[:,:] i_idx,
                                           cnp.npy_int64[:,:] j_idx,
                                           cnp.npy_int64[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_int64(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int64_clongdouble(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int64[:,:] i_idx,
                                           cnp.npy_int64[:,:] j_idx,
                                           long double complex[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_clongdouble(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int64_uint32(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int64[:,:] i_idx,
                                           cnp.npy_int64[:,:] j_idx,
                                           cnp.npy_uint32[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_uint32(M, N, rows, data, i, j, values[x, y])
cpdef _lil_fancy_set_int64_float32(cnp.npy_intp M, cnp.npy_intp N,
                                           object[:] rows,
                                           object[:] data,
                                           cnp.npy_int64[:,:] i_idx,
                                           cnp.npy_int64[:,:] j_idx,
                                           cnp.npy_float32[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_float32(M, N, rows, data, i, j, values[x, y])


cdef dict _LIL_FANCY_SET_DISPATCH = {

(np.dtype(np.int32), np.dtype(np.int32)): _lil_fancy_set_int32_int32,
(np.dtype(np.int32), np.dtype(np.int16)): _lil_fancy_set_int32_int16,
(np.dtype(np.int32), np.dtype(np.bool_)): _lil_fancy_set_int32_bool_,
(np.dtype(np.int32), np.dtype(np.longdouble)): _lil_fancy_set_int32_longdouble,
(np.dtype(np.int32), np.dtype(np.uint8)): _lil_fancy_set_int32_uint8,
(np.dtype(np.int32), np.dtype(np.complex64)): _lil_fancy_set_int32_complex64,
(np.dtype(np.int32), np.dtype(np.int8)): _lil_fancy_set_int32_int8,
(np.dtype(np.int32), np.dtype(np.uint64)): _lil_fancy_set_int32_uint64,
(np.dtype(np.int32), np.dtype(np.float64)): _lil_fancy_set_int32_float64,
(np.dtype(np.int32), np.dtype(np.complex128)): _lil_fancy_set_int32_complex128,
(np.dtype(np.int32), np.dtype(np.uint16)): _lil_fancy_set_int32_uint16,
(np.dtype(np.int32), np.dtype(np.int64)): _lil_fancy_set_int32_int64,
(np.dtype(np.int32), np.dtype(np.clongdouble)): _lil_fancy_set_int32_clongdouble,
(np.dtype(np.int32), np.dtype(np.uint32)): _lil_fancy_set_int32_uint32,
(np.dtype(np.int32), np.dtype(np.float32)): _lil_fancy_set_int32_float32,
(np.dtype(np.int64), np.dtype(np.int32)): _lil_fancy_set_int64_int32,
(np.dtype(np.int64), np.dtype(np.int16)): _lil_fancy_set_int64_int16,
(np.dtype(np.int64), np.dtype(np.bool_)): _lil_fancy_set_int64_bool_,
(np.dtype(np.int64), np.dtype(np.longdouble)): _lil_fancy_set_int64_longdouble,
(np.dtype(np.int64), np.dtype(np.uint8)): _lil_fancy_set_int64_uint8,
(np.dtype(np.int64), np.dtype(np.complex64)): _lil_fancy_set_int64_complex64,
(np.dtype(np.int64), np.dtype(np.int8)): _lil_fancy_set_int64_int8,
(np.dtype(np.int64), np.dtype(np.uint64)): _lil_fancy_set_int64_uint64,
(np.dtype(np.int64), np.dtype(np.float64)): _lil_fancy_set_int64_float64,
(np.dtype(np.int64), np.dtype(np.complex128)): _lil_fancy_set_int64_complex128,
(np.dtype(np.int64), np.dtype(np.uint16)): _lil_fancy_set_int64_uint16,
(np.dtype(np.int64), np.dtype(np.int64)): _lil_fancy_set_int64_int64,
(np.dtype(np.int64), np.dtype(np.clongdouble)): _lil_fancy_set_int64_clongdouble,
(np.dtype(np.int64), np.dtype(np.uint32)): _lil_fancy_set_int64_uint32,
(np.dtype(np.int64), np.dtype(np.float32)): _lil_fancy_set_int64_float32,
}




cpdef _lil_fancy_set_generic(cnp.npy_intp M, cnp.npy_intp N,
                             object[:] rows,
                             object[:] data,
                             cnp.ndarray i_idx,
                             cnp.ndarray j_idx,
                             cnp.ndarray values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            _lil_insert_float32(M, N, rows, data, i, j, values[x, y])


cdef lil_insertat_nocheck(list row, list data, cnp.npy_intp j, object x):
    """
    Insert a single item to LIL matrix.

    Doesn't check for bounds errors. Doesn't check for zero x.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef cnp.npy_intp pos

    pos = bisect_left(row, j)
    if pos == len(row):
        row.append(j)
        data.append(x)
    elif row[pos] != j:
        row.insert(pos, j)
        data.insert(pos, x)
    else:
        data[pos] = x


cdef lil_deleteat_nocheck(list row, list data, cnp.npy_intp j):
    """
    Delete a single item from a row in LIL matrix.

    Doesn't check for bounds errors.

    Parameters
    ----------
    row, data
        Row data for LIL matrix.
    j : int
        Column index to delete at

    """
    cdef cnp.npy_intp pos
    pos = bisect_left(row, j)
    if pos < len(row) and row[pos] == j:
        del row[pos]
        del data[pos]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bisect_left(list a, cnp.npy_intp x):
    """
    Bisection search in a sorted list.

    List is assumed to contain objects castable to integers.

    Parameters
    ----------
    a
        List to search in
    x
        Value to search for

    Returns
    -------
    j : int
        Index at value (if present), or at the point to which
        it can be inserted maintaining order.

    """
    cdef cnp.npy_intp hi = len(a)
    cdef cnp.npy_intp lo = 0
    cdef cnp.npy_intp mid, v

    while lo < hi:
        mid = (lo + hi)//2
        v = a[mid]
        if v < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _fill_dtype_map(map, chars):
    """
    Fill in Numpy dtype chars for problematic types, working around
    Numpy < 1.6 bugs.
    """
    for c in chars:
        if c in "SUVO":
            continue
        dt = np.dtype(c)
        if dt not in map:
            for k, v in map.items():
                if k.kind == dt.kind and k.itemsize == dt.itemsize:
                    map[dt] = v
                    break


def _fill_dtype_map2(map):
    """
    Fill in Numpy dtype chars for problematic types, working around
    Numpy < 1.6 bugs.
    """
    for c1 in np.typecodes['Integer']:
        for c2 in np.typecodes['All']:
            if c2 in "SUVO":
                continue
            dt1 = np.dtype(c1)
            dt2 = np.dtype(c2)
            if (dt1, dt2) not in map:
                for k, v in map.items():
                    if (k[0].kind == dt1.kind and k[0].itemsize == dt1.itemsize and
                        k[1].kind == dt2.kind and k[1].itemsize == dt2.itemsize):
                        map[(dt1, dt2)] = v
                        break

_fill_dtype_map(_LIL_INSERT_DISPATCH, np.typecodes['All'])
_fill_dtype_map(_LIL_FANCY_GET_DISPATCH, np.typecodes['Integer'])
_fill_dtype_map2(_LIL_FANCY_SET_DISPATCH)
