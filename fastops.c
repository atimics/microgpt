#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>

/*
 * C accelerator for microgpt.
 * Supports both array.array('d') [zero-copy via buffer protocol]
 * and list[float] [copies to/from C buffers] inputs.
 * Forward functions return array.array('d') when available.
 */

static PyObject *array_cls = NULL; /* cached array.array type */

/* ==== DArr: unified zero-copy/copy access to array.array or list ==== */

/*
 * DArr: Dynamic Array abstraction for flexible Python input handling
 * 
 * This structure provides a unified interface for accessing numeric data from
 * either array.array('d') objects or Python list[float] objects.
 * 
 * Memory Strategy:
 * - Zero-copy path: When input is array.array('d'), we use Python's buffer protocol
 *   to directly access the underlying C array without copying. This is fast and
 *   memory-efficient for large arrays.
 * - Copy path: When input is a Python list, we allocate a temporary C buffer with
 *   malloc() and copy the list elements. This allows the same C code to work with
 *   both input types at the cost of a temporary allocation.
 * 
 * Fields:
 * - ptr: Pointer to the C double array (either from buffer protocol or malloc)
 * - n: Number of elements in the array
 * - view: Buffer protocol view (only valid when buffered == 1)
 * - buffered: 1 = zero-copy (buffer protocol), 0 = malloc'd copy from list
 */
typedef struct {
    double *ptr;
    Py_ssize_t n;
    Py_buffer view;
    int buffered; /* 1 = buffer protocol (zero-copy), 0 = malloc'd from list */
} DArr;

/*
 * darr_get: Initialize a DArr from a Python object
 * 
 * Attempts zero-copy access via buffer protocol first, falls back to copying
 * from a Python list if necessary.
 * 
 * Parameters:
 * - d: DArr structure to initialize
 * - obj: Python object (should be array.array('d') or list[float])
 * - writable: 1 if we need write access, 0 for read-only
 * 
 * Returns: 0 on success, -1 on error (with Python exception set)
 * 
 * Memory Layout:
 * - Buffer protocol path: d->ptr points directly to Python object's internal buffer
 * - List path: d->ptr points to newly malloc'd buffer containing copied values
 */
static int darr_get(DArr *d, PyObject *obj, int writable) {
    /* Try buffer protocol first (zero-copy for array.array) */
    if (PyObject_CheckBuffer(obj)) {
        int flags = PyBUF_C_CONTIGUOUS | (writable ? PyBUF_WRITABLE : 0);
        if (PyObject_GetBuffer(obj, &d->view, flags) == 0) {
            d->ptr = (double *)d->view.buf;
            d->n = (Py_ssize_t)(d->view.len / sizeof(double));
            d->buffered = 1;
            return 0;
        }
        PyErr_Clear();
    }
    /* Fallback: Python list of floats - requires copy to C buffer */
    if (PyList_Check(obj)) {
        d->n = PyList_GET_SIZE(obj);
        d->ptr = (double *)malloc(d->n * sizeof(double));
        if (!d->ptr) { PyErr_NoMemory(); return -1; }
        for (Py_ssize_t i = 0; i < d->n; i++)
            d->ptr[i] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(obj, i));
        d->buffered = 0;
        return 0;
    }
    PyErr_SetString(PyExc_TypeError, "expected array.array or list");
    return -1;
}

/*
 * darr_sync: Write modified malloc'd buffer back to Python list
 * 
 * For writable DArr objects that were created from Python lists (buffered == 0),
 * this copies the modified C buffer back to the original Python list.
 * For buffer protocol objects (buffered == 1), this is a no-op since modifications
 * were made directly to the Python object's internal buffer.
 * 
 * This is part of the cleanup sequence: modify buffer -> sync -> done
 */
/* Write modified malloc'd buffer back to list; no-op for buffer protocol */
static void darr_sync(DArr *d, PyObject *obj) {
    if (!d->buffered && PyList_Check(obj)) {
        for (Py_ssize_t i = 0; i < d->n; i++) {
            PyObject *nv = PyFloat_FromDouble(d->ptr[i]);
            Py_XDECREF(PyList_GET_ITEM(obj, i));
            PyList_SET_ITEM(obj, i, nv);
        }
    }
}

/*
 * darr_done: Release DArr resources
 * 
 * Error handling pattern: This is the cleanup function in the cascading
 * cleanup pattern used throughout this module. Always safe to call.
 * 
 * For buffer protocol (buffered == 1): Releases the Python buffer view
 * For malloc'd data (buffered == 0): Frees the allocated memory
 * 
 * This function is safe to call multiple times on the same DArr if needed,
 * though typical usage is once at the end of a function in error handling
 * or normal cleanup paths.
 */
static void darr_done(DArr *d) {
    if (d->buffered)
        PyBuffer_Release(&d->view);
    else
        free(d->ptr);
}

/*
 * darr_new: Create a new Python array.array('d') from C double buffer
 * 
 * This is used by forward functions to return results to Python.
 * 
 * Preferred path: If array.array is available (cached in array_cls), creates
 * an array.array('d') object. The 'y#' format specifier copies the data.
 * 
 * Fallback path: If array module is not available, creates a Python list.
 * 
 * Returns: New Python object (array.array or list) with reference count 1
 */
/* Create new array.array('d') from C double buffer */
static PyObject *darr_new(const double *data, Py_ssize_t n) {
    if (array_cls)
        return PyObject_CallFunction(array_cls, "sy#", "d",
                                     (const char *)data, (Py_ssize_t)(n * sizeof(double)));
    /* fallback: return list */
    PyObject *out = PyList_New(n);
    if (!out) return NULL;
    for (Py_ssize_t i = 0; i < n; i++)
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(data[i]));
    return out;
}

/*
 * rows_to_flat: Extract T rows from a Python list into a contiguous flat buffer
 * 
 * Used to convert list-of-vectors (e.g., attention keys/values across time) into
 * a single flat C array for efficient processing.
 * 
 * Memory layout: Returns a malloc'd buffer of size T * dim containing all rows
 * concatenated: [row0_elem0, row0_elem1, ..., row1_elem0, row1_elem1, ...]
 * 
 * This layout is cache-friendly for the attention computation which needs to
 * access elements from the same position across different time steps.
 * 
 * Error handling: Uses cascading cleanup - frees buffer and returns NULL on error
 */
/* Extract T rows from a Python list into a contiguous flat buffer */
static double *rows_to_flat(PyObject *rows_list, Py_ssize_t T, Py_ssize_t dim) {
    double *buf = (double *)malloc(T * dim * sizeof(double));
    if (!buf) { PyErr_NoMemory(); return NULL; }
    for (Py_ssize_t t = 0; t < T; t++) {
        DArr row;
        if (darr_get(&row, PyList_GET_ITEM(rows_list, t), 0) < 0) {
            free(buf);
            return NULL;
        }
        memcpy(buf + t * dim, row.ptr, dim * sizeof(double));
        darr_done(&row);
    }
    return buf;
}

/*
 * flat_to_rows: Write contiguous flat buffer back to T rows
 * 
 * Inverse of rows_to_flat. Distributes a flat C buffer back into a Python list
 * of vectors. Each row is updated in-place using darr_get/darr_sync pattern.
 * 
 * Memory layout: Input buffer has layout [row0, row1, ..., rowT-1] where each
 * row is 'dim' consecutive doubles.
 * 
 * Returns: 0 on success, -1 on error
 */
/* Write contiguous flat buffer back to T rows */
static int flat_to_rows(const double *buf, PyObject *rows_list, Py_ssize_t T, Py_ssize_t dim) {
    for (Py_ssize_t t = 0; t < T; t++) {
        PyObject *row_obj = PyList_GET_ITEM(rows_list, t);
        DArr row;
        if (darr_get(&row, row_obj, 1) < 0) return -1;
        memcpy(row.ptr, buf + t * dim, dim * sizeof(double));
        darr_sync(&row, row_obj);
        darr_done(&row);
    }
    return 0;
}

/*
 * vec_dot: Compute dot product of two vectors
 * 
 * Operation: Returns sum(a[i] * b[i])
 * 
 * Buffer protocol vs list: Handles both array.array and list inputs via DArr
 * 
 * Memory: No allocation, works with zero-copy or temporary buffers from darr_get
 */
/* ==== vec_dot ==== */
static PyObject* fastops_vec_dot(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    DArr a, b;
    if (darr_get(&a, a_obj, 0) < 0) return NULL;
    if (darr_get(&b, b_obj, 0) < 0) { darr_done(&a); return NULL; }
    double s = 0.0;
    for (Py_ssize_t i = 0; i < a.n; i++) s += a.ptr[i] * b.ptr[i];
    darr_done(&a); darr_done(&b);
    return PyFloat_FromDouble(s);
}

/*
 * vec_axpy: Scaled vector addition (BLAS-style)
 * 
 * Operation: y += alpha * x (in-place update of y)
 * 
 * Buffer protocol vs list: 
 * - x: read-only, accepts array.array or list
 * - y: writable, modified in-place (uses darr_sync for list inputs)
 * 
 * Memory: No allocation, works with zero-copy or temporary buffers from darr_get
 */
/* ==== vec_axpy: y += alpha * x ==== */
static PyObject* fastops_vec_axpy(PyObject* self, PyObject* args) {
    double alpha;
    PyObject *x_obj, *y_obj;
    if (!PyArg_ParseTuple(args, "dOO", &alpha, &x_obj, &y_obj)) return NULL;
    DArr x, y;
    if (darr_get(&x, x_obj, 0) < 0) return NULL;
    if (darr_get(&y, y_obj, 1) < 0) { darr_done(&x); return NULL; }
    for (Py_ssize_t i = 0; i < x.n; i++) y.ptr[i] += alpha * x.ptr[i];
    darr_sync(&y, y_obj);
    darr_done(&x); darr_done(&y);
    Py_RETURN_NONE;
}

/*
 * rmsnorm_forward: Root Mean Square Layer Normalization forward pass
 * 
 * Operation: Normalizes input by its RMS value
 *   scale = 1 / sqrt(mean(x^2) + epsilon)
 *   output = x * scale
 * 
 * Memory: Uses malloc for output buffer (cannot modify input in-place)
 * 
 * Numerical Stability: Adds epsilon (1e-5) to prevent division by zero
 * 
 * Returns: (normalized_output, scale) tuple
 *   - normalized_output: array.array('d')
 *   - scale: float (needed for backward pass)
 */
/* ==== rmsnorm forward ==== */
static PyObject* fastops_rmsnorm_forward(PyObject* self, PyObject* args) {
    PyObject *xd_obj;
    if (!PyArg_ParseTuple(args, "O", &xd_obj)) return NULL;
    DArr xd;
    if (darr_get(&xd, xd_obj, 0) < 0) return NULL;
    Py_ssize_t n = xd.n;

    double ms = 0.0;
    for (Py_ssize_t i = 0; i < n; i++) ms += xd.ptr[i] * xd.ptr[i];
    ms /= n;
    double scale = 1.0 / sqrt(ms + 1e-5);

    double *out = (double *)malloc(n * sizeof(double));
    if (!out) { darr_done(&xd); return PyErr_NoMemory(); }
    for (Py_ssize_t i = 0; i < n; i++) out[i] = xd.ptr[i] * scale;
    darr_done(&xd);

    PyObject *out_arr = darr_new(out, n);
    free(out);
    PyObject *result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, out_arr);
    PyTuple_SET_ITEM(result, 1, PyFloat_FromDouble(scale));
    return result;
}

/*
 * rmsnorm_backward: Root Mean Square Layer Normalization backward pass
 * 
 * Operation: Backpropagates gradients through RMSNorm
 * Given output gradient (og), input data (xd), scale from forward pass,
 * accumulates gradient into input gradient (xg).
 * 
 * Gradient Formula:
 *   xg[j] += og[j] * scale - (scale^3 / n) * xd[j] * dot(og, xd)
 * 
 * The second term accounts for the normalization's effect on all elements
 * (gradient of RMS statistic).
 * 
 * Memory: Works in-place on gradient buffer, no additional allocation
 */
/* ==== rmsnorm backward ==== */
static PyObject* fastops_rmsnorm_backward(PyObject* self, PyObject* args) {
    PyObject *og_obj, *xd_obj, *xg_obj;
    double scale;
    if (!PyArg_ParseTuple(args, "OOdO", &og_obj, &xd_obj, &scale, &xg_obj))
        return NULL;

    DArr og, xd, xg;
    if (darr_get(&og, og_obj, 0) < 0) return NULL;
    if (darr_get(&xd, xd_obj, 0) < 0) { darr_done(&og); return NULL; }
    if (darr_get(&xg, xg_obj, 1) < 0) { darr_done(&og); darr_done(&xd); return NULL; }

    Py_ssize_t n = xd.n;
    double dot = 0.0;
    for (Py_ssize_t i = 0; i < n; i++) dot += og.ptr[i] * xd.ptr[i];
    double s3n = scale * scale * scale / n;
    for (Py_ssize_t j = 0; j < n; j++)
        xg.ptr[j] += og.ptr[j] * scale - s3n * xd.ptr[j] * dot;

    darr_sync(&xg, xg_obj);
    darr_done(&og); darr_done(&xd); darr_done(&xg);
    Py_RETURN_NONE;
}

/*
 * squared_relu_forward: Squared ReLU activation forward pass
 * 
 * Operation: out[i] = (x[i])^2 if x[i] > 0, else 0
 * 
 * This is a variant of ReLU that squares positive values, creating
 * a smooth quadratic activation that grows faster than linear ReLU.
 * 
 * Memory: Uses malloc for output buffer
 * 
 * Returns: array.array('d')
 */
/* ==== squared_relu forward ==== */
static PyObject* fastops_squared_relu_forward(PyObject* self, PyObject* args) {
    PyObject *xd_obj;
    if (!PyArg_ParseTuple(args, "O", &xd_obj)) return NULL;
    DArr xd;
    if (darr_get(&xd, xd_obj, 0) < 0) return NULL;
    Py_ssize_t n = xd.n;

    double *out = (double *)malloc(n * sizeof(double));
    if (!out) { darr_done(&xd); return PyErr_NoMemory(); }
    for (Py_ssize_t i = 0; i < n; i++) {
        double v = xd.ptr[i];
        out[i] = v > 0.0 ? v * v : 0.0;
    }
    darr_done(&xd);
    PyObject *result = darr_new(out, n);
    free(out);
    return result;
}

/*
 * squared_relu_backward: Squared ReLU activation backward pass
 * 
 * Operation: xg += 2 * x * og where x > 0 (gradient is zero where x <= 0)
 * 
 * Derivative of squared ReLU:
 *   d/dx(x^2) = 2x for x > 0
 *   d/dx(0) = 0 for x <= 0
 * 
 * Memory: Works in-place on gradient buffer, no additional allocation
 */
/* ==== squared_relu backward ==== */
static PyObject* fastops_squared_relu_backward(PyObject* self, PyObject* args) {
    PyObject *xd_obj, *og_obj, *xg_obj;
    if (!PyArg_ParseTuple(args, "OOO", &xd_obj, &og_obj, &xg_obj)) return NULL;
    DArr xd, og, xg;
    if (darr_get(&xd, xd_obj, 0) < 0) return NULL;
    if (darr_get(&og, og_obj, 0) < 0) { darr_done(&xd); return NULL; }
    if (darr_get(&xg, xg_obj, 1) < 0) { darr_done(&xd); darr_done(&og); return NULL; }

    for (Py_ssize_t i = 0; i < xd.n; i++)
        if (xd.ptr[i] > 0.0)
            xg.ptr[i] += 2.0 * xd.ptr[i] * og.ptr[i];

    darr_sync(&xg, xg_obj);
    darr_done(&xd); darr_done(&og); darr_done(&xg);
    Py_RETURN_NONE;
}

/*
 * tensor_add: Element-wise addition of two tensors
 * 
 * Operation: out[i] = a[i] + b[i]
 * 
 * Memory: Uses malloc for output buffer
 * 
 * Returns: array.array('d')
 */
/* ==== tensor_add forward ==== */
static PyObject* fastops_tensor_add(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    DArr a, b;
    if (darr_get(&a, a_obj, 0) < 0) return NULL;
    if (darr_get(&b, b_obj, 0) < 0) { darr_done(&a); return NULL; }
    Py_ssize_t n = a.n;

    double *out = (double *)malloc(n * sizeof(double));
    if (!out) { darr_done(&a); darr_done(&b); return PyErr_NoMemory(); }
    for (Py_ssize_t i = 0; i < n; i++) out[i] = a.ptr[i] + b.ptr[i];
    darr_done(&a); darr_done(&b);

    PyObject *result = darr_new(out, n);
    free(out);
    return result;
}

/*
 * tensor_add_backward: Backprop through element-wise addition
 * 
 * Operation: ag += og, bg += og (gradient flows equally to both inputs)
 * 
 * Memory: Works in-place on gradient buffers, no additional allocation
 */
/* ==== tensor_add backward ==== */
static PyObject* fastops_tensor_add_backward(PyObject* self, PyObject* args) {
    PyObject *og_obj, *ag_obj, *bg_obj;
    if (!PyArg_ParseTuple(args, "OOO", &og_obj, &ag_obj, &bg_obj)) return NULL;
    DArr og, ag, bg;
    if (darr_get(&og, og_obj, 0) < 0) return NULL;
    if (darr_get(&ag, ag_obj, 1) < 0) { darr_done(&og); return NULL; }
    if (darr_get(&bg, bg_obj, 1) < 0) { darr_done(&og); darr_done(&ag); return NULL; }

    for (Py_ssize_t i = 0; i < og.n; i++) {
        ag.ptr[i] += og.ptr[i];
        bg.ptr[i] += og.ptr[i];
    }

    darr_sync(&ag, ag_obj); darr_sync(&bg, bg_obj);
    darr_done(&og); darr_done(&ag); darr_done(&bg);
    Py_RETURN_NONE;
}

/*
 * cross_entropy forward: Compute softmax cross-entropy loss
 * 
 * Operation: Given logits and a target class index, computes:
 * 1. Softmax probabilities: prob[i] = exp(logit[i]) / sum(exp(logit[j]))
 * 2. Cross-entropy loss: -log(prob[target])
 * 
 * Numerical Stability:
 * - Softmax uses max-normalization to prevent overflow: exp(x - max) instead of exp(x)
 * - Division by zero prevented with fmax(total, 1e-30) where 1e-30 is chosen as:
 *   * Large enough to avoid denormal/underflow ranges
 *   * Small enough not to materially affect typical probability values
 * - Log(0) prevented with fmax(prob[target], 1e-30)
 * 
 * Memory: Uses malloc for temporary softmax buffer (need to modify logits in-place)
 * 
 * Returns: (loss, probs) tuple where probs is array.array('d') of softmax values
 */
/* ==== cross_entropy forward ==== */
static PyObject* fastops_cross_entropy_forward(PyObject* self, PyObject* args) {
    PyObject *logits_obj;
    Py_ssize_t target;
    if (!PyArg_ParseTuple(args, "On", &logits_obj, &target)) return NULL;
    DArr lg;
    if (darr_get(&lg, logits_obj, 0) < 0) return NULL;
    Py_ssize_t n = lg.n;

    double *buf = (double *)malloc(n * sizeof(double));
    if (!buf) { darr_done(&lg); return PyErr_NoMemory(); }
    memcpy(buf, lg.ptr, n * sizeof(double));
    darr_done(&lg);

    if (target < 0 || target >= n) {
        free(buf);
        PyErr_Format(PyExc_IndexError, "target index %zd out of range [0, %zd)", target, n);
        return NULL;
    }

    /* softmax */
    double max_val = buf[0];
    for (Py_ssize_t i = 1; i < n; i++)
        if (buf[i] > max_val) max_val = buf[i];
    double total = 0.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        buf[i] = exp(buf[i] - max_val);
        total += buf[i];
    }
    /* Ensure total is never zero to prevent division by zero.
     * Use 1e-30 (rather than DBL_MIN or a larger epsilon) as a pragmatic floor:
     * it is safely above denormal/underflow ranges while being small enough
     * that it does not materially affect typical probability or loss values. */
    total = fmax(total, 1e-30);
    for (Py_ssize_t i = 0; i < n; i++) buf[i] /= total;

    /* Apply the same epsilon when clamping buf[target] to avoid log(0). */
    double loss = -log(fmax(buf[target], 1e-30));
    PyObject *probs = darr_new(buf, n);
    free(buf);

    PyObject *result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(loss));
    PyTuple_SET_ITEM(result, 1, probs);
    return result;
}

/*
 * cross_entropy_backward: Cross-entropy loss backward pass
 * 
 * Operation: Backpropagates gradient through softmax and negative log likelihood
 * 
 * For the target class: lg[target] += g * (prob[target] - 1)
 * For other classes: lg[i] += g * prob[i]
 * 
 * This is the gradient of -log(softmax(logits)[target]) w.r.t. logits,
 * which simplifies to (softmax - one_hot_target) scaled by upstream gradient g.
 * 
 * Memory: Works in-place on logits gradient buffer, no additional allocation
 */
/* ==== cross_entropy backward ==== */
static PyObject* fastops_cross_entropy_backward(PyObject* self, PyObject* args) {
    PyObject *probs_obj, *lg_obj;
    double g;
    Py_ssize_t target;
    if (!PyArg_ParseTuple(args, "dOnO", &g, &probs_obj, &target, &lg_obj))
        return NULL;

    DArr probs, lg;
    if (darr_get(&probs, probs_obj, 0) < 0) return NULL;
    if (darr_get(&lg, lg_obj, 1) < 0) { darr_done(&probs); return NULL; }

    for (Py_ssize_t i = 0; i < probs.n; i++) {
        double delta = (i == target) ? (probs.ptr[i] - 1.0) : probs.ptr[i];
        lg.ptr[i] += g * delta;
    }

    darr_sync(&lg, lg_obj);
    darr_done(&probs); darr_done(&lg);
    Py_RETURN_NONE;
}

/*
 * attention_forward: Multi-head self-attention forward pass
 * 
 * Operation: Computes attention output = softmax(Q @ K^T / sqrt(d)) @ V
 * for each attention head independently.
 * 
 * Algorithm (per head):
 * 1. Compute attention scores: score[t] = dot(query_head, key[t]_head) / sqrt(head_dim)
 * 2. Softmax over scores (with max-normalization for numerical stability)
 * 3. Weighted sum of values: output_head = sum(attention_weights[t] * value[t]_head)
 * 
 * Memory Layout:
 * - Input qd: flattened query vector of shape (n_head * head_dim)
 * - Input keys/vals: Python lists of T vectors, each of shape (n_head * head_dim)
 * - Internal kbuf/vbuf: flat buffers of size T * dim for cache-friendly access
 * - Output: single vector of shape (n_head * head_dim)
 * 
 * Memory Management:
 * - calloc for output (needs to start at zero for accumulation)
 * - malloc for all_aw (attention weights, returned to Python for backward pass)
 * - malloc for kbuf/vbuf (temporary flattened key/value buffers)
 * - Cascading cleanup on error: darr_done, then free in reverse allocation order
 * 
 * Numerical Stability:
 * - Attention scores normalized by sqrt(head_dim) to prevent large dot products
 * - Softmax uses max-normalization: exp(score - max_score) to prevent overflow
 * 
 * Returns: (output_vector, attention_weights_per_head) tuple
 *   - output_vector: array.array('d') of shape (dim,)
 *   - attention_weights_per_head: list of n_head lists, each with T weights
 */
/* ==== attention forward ==== */
static PyObject* fastops_attention_forward(PyObject* self, PyObject* args) {
    PyObject *qd_obj, *keys_list, *vals_list;
    int n_head, head_dim;
    if (!PyArg_ParseTuple(args, "OOOii", &qd_obj, &keys_list, &vals_list,
                          &n_head, &head_dim))
        return NULL;

    Py_ssize_t T = PyList_GET_SIZE(keys_list);
    Py_ssize_t dim = n_head * head_dim;

    DArr qd;
    if (darr_get(&qd, qd_obj, 0) < 0) return NULL;

    double *kbuf = rows_to_flat(keys_list, T, dim);
    if (!kbuf) { darr_done(&qd); return NULL; }
    double *vbuf = rows_to_flat(vals_list, T, dim);
    if (!vbuf) { darr_done(&qd); free(kbuf); return NULL; }

    double *out = (double *)calloc(dim, sizeof(double));
    double *all_aw = (double *)malloc(n_head * T * sizeof(double));
    if (!out || !all_aw) {
        darr_done(&qd); free(kbuf); free(vbuf); free(out); free(all_aw);
        return PyErr_NoMemory();
    }

    /* Scale factor for numerical stability: 1/sqrt(head_dim) */
    double scale = sqrt((double)head_dim);

    /* Process each attention head independently */
    for (int h = 0; h < n_head; h++) {
        Py_ssize_t hs = h * head_dim;  /* head start offset */
        
        /* Step 1: Compute attention scores for this head across all time steps */
        double max_val = -1e30;  /* track max for softmax stability */
        for (Py_ssize_t t = 0; t < T; t++) {
            /* Dot product: query_head · key[t]_head */
            double s = 0.0;
            for (int j = 0; j < head_dim; j++)
                s += qd.ptr[hs + j] * kbuf[t * dim + hs + j];
            s /= scale;  /* scale by 1/sqrt(head_dim) */
            all_aw[h * T + t] = s;
            if (s > max_val) max_val = s;
        }
        
        /* Step 2: Softmax - convert scores to probabilities */
        double total = 0.0;
        for (Py_ssize_t t = 0; t < T; t++) {
            all_aw[h * T + t] = exp(all_aw[h * T + t] - max_val);  /* max normalization */
            total += all_aw[h * T + t];
        }
        /* Normalize to probabilities */
        for (Py_ssize_t t = 0; t < T; t++)
            all_aw[h * T + t] /= total;
            
        /* Step 3: Weighted sum of values using attention probabilities */
        for (int j = 0; j < head_dim; j++) {
            double s = 0.0;
            for (Py_ssize_t t = 0; t < T; t++)
                s += all_aw[h * T + t] * vbuf[t * dim + hs + j];
            out[hs + j] = s;  /* output for this head dimension */
        }
    }

    darr_done(&qd);
    PyObject *out_arr = darr_new(out, dim);

    /* Package attention weights as list of lists (one list per head) for backward pass */
    PyObject *aw_list = PyList_New(n_head);
    for (int h = 0; h < n_head; h++) {
        PyObject *hw = PyList_New(T);
        for (Py_ssize_t t = 0; t < T; t++)
            PyList_SET_ITEM(hw, t, PyFloat_FromDouble(all_aw[h * T + t]));
        PyList_SET_ITEM(aw_list, h, hw);
    }

    free(kbuf); free(vbuf); free(out); free(all_aw);
    PyObject *result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, out_arr);
    PyTuple_SET_ITEM(result, 1, aw_list);  /* attention weights needed for backward */
    return result;
}

/*
 * attention_backward: Multi-head self-attention backward pass
 * 
 * Operation: Backpropagates gradients through the attention mechanism.
 * Given gradient of loss w.r.t. attention output (og), computes gradients
 * w.r.t. query (qg), keys (kgrad_list), and values (vgrad_list).
 * 
 * Gradient Flow (per head):
 * 1. Value gradients: d_value[t][j] = attention_weight[t] * d_output[j]
 * 2. Attention weight gradients: d_attn[t] = sum_j(d_output[j] * value[t][j])
 * 3. Softmax backward: Convert d_attn gradients through softmax jacobian
 * 4. Score gradients: d_score[t] = attention_weight[t] * (d_attn[t] - dot_product) / scale
 * 5. Query/Key gradients: Backprop through dot products
 * 
 * Memory Layout:
 * - Inputs: og, qd, keys/vals lists, attention weights (from forward), n_head, head_dim
 * - Outputs: qg, kgrad_list, vgrad_list (gradients accumulated in-place)
 * - Temporary buffers: kbuf, vbuf, kgbuf, vgbuf (flat), aw (attention weights), d_attn
 * 
 * Memory Management:
 * - All gradient buffers initialized via rows_to_flat (reads existing gradients)
 * - Uses malloc for all temporary buffers (aw, d_attn)
 * - Cascading cleanup pattern: free in reverse allocation order on error
 * - memset used to zero d_attn buffer for each head (reused across heads)
 * 
 * Numerical Considerations:
 * - Softmax backward includes the dot product term for proper jacobian
 * - Zero-gradient short-circuits (if g == 0.0 or dl == 0.0) for efficiency
 * - Scale factor (sqrt(head_dim)) applied to attention score gradients
 * 
 * Error Handling: Cascading darr_done() cleanup for all DArr objects,
 * followed by free() for malloc'd buffers
 */
/* ==== attention backward ==== */
static PyObject* fastops_attention_backward(PyObject* self, PyObject* args) {
    PyObject *og_obj, *qd_obj, *keys_list, *vals_list, *aw_list;
    PyObject *qg_obj, *kgrad_list, *vgrad_list;
    int n_head, head_dim;
    if (!PyArg_ParseTuple(args, "OOOOOOOOii",
            &og_obj, &qd_obj, &keys_list, &vals_list, &aw_list,
            &qg_obj, &kgrad_list, &vgrad_list,
            &n_head, &head_dim))
        return NULL;

    Py_ssize_t T = PyList_GET_SIZE(keys_list);
    Py_ssize_t dim = n_head * head_dim;
    double scale = sqrt((double)head_dim);

    DArr og, qd, qg;
    if (darr_get(&og, og_obj, 0) < 0) return NULL;
    if (darr_get(&qd, qd_obj, 0) < 0) { darr_done(&og); return NULL; }
    if (darr_get(&qg, qg_obj, 1) < 0) { darr_done(&og); darr_done(&qd); return NULL; }

    double *kbuf = rows_to_flat(keys_list, T, dim);
    double *vbuf = rows_to_flat(vals_list, T, dim);
    double *kgbuf = rows_to_flat(kgrad_list, T, dim);  /* existing gradients */
    double *vgbuf = rows_to_flat(vgrad_list, T, dim);  /* existing gradients */
    if (!kbuf || !vbuf || !kgbuf || !vgbuf) {
        darr_done(&og); darr_done(&qd); darr_done(&qg);
        free(kbuf); free(vbuf); free(kgbuf); free(vgbuf);
        return PyErr_NoMemory();
    }

    /* Extract attention weights from forward pass (needed for backward) */
    double *aw = (double *)malloc(n_head * T * sizeof(double));
    double *d_attn = (double *)malloc(T * sizeof(double));  /* temp buffer per head */
    if (!aw || !d_attn) {
        darr_done(&og); darr_done(&qd); darr_done(&qg);
        free(kbuf); free(vbuf); free(kgbuf); free(vgbuf);
        free(aw); free(d_attn);
        return PyErr_NoMemory();
    }
    for (int h = 0; h < n_head; h++) {
        PyObject *hw = PyList_GET_ITEM(aw_list, h);
        for (Py_ssize_t t = 0; t < T; t++)
            aw[h * T + t] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(hw, t));
    }

    /* Process each attention head independently */
    for (int h = 0; h < n_head; h++) {
        Py_ssize_t hs = h * head_dim;  /* head start offset */
        memset(d_attn, 0, T * sizeof(double));  /* clear temp buffer for this head */
        
        /* Step 1: Backprop through weighted sum of values */
        for (int j = 0; j < head_dim; j++) {
            double g = og.ptr[hs + j];  /* gradient w.r.t. output[head][j] */
            if (g == 0.0) continue;  /* skip if no gradient */
            Py_ssize_t idx = hs + j;
            for (Py_ssize_t t = 0; t < T; t++) {
                /* d_value[t][j] += attention_weight[t] * d_output[j] */
                vgbuf[t * dim + idx] += g * aw[h * T + t];
                /* d_attn[t] += d_output[j] * value[t][j] */
                d_attn[t] += g * vbuf[t * dim + idx];
            }
        }
        
        /* Step 2: Backprop through softmax to get d_score */
        /* Softmax Jacobian: d_score = attention_weight * (d_attn - dot_product) */
        double dot = 0.0;
        for (Py_ssize_t t = 0; t < T; t++)
            dot += aw[h * T + t] * d_attn[t];  /* sum of weighted gradients */
        
        /* Step 3: Backprop through attention scores (Q·K^T / scale) */
        for (Py_ssize_t t = 0; t < T; t++) {
            double dl = aw[h * T + t] * (d_attn[t] - dot) / scale;  /* gradient w.r.t. score[t] */
            if (dl == 0.0) continue;  /* skip if no gradient */
            for (int j = 0; j < head_dim; j++) {
                Py_ssize_t idx = hs + j;
                /* d_query[j] += d_score[t] * key[t][j] */
                qg.ptr[idx] += dl * kbuf[t * dim + idx];
                /* d_key[t][j] += d_score[t] * query[j] */
                kgbuf[t * dim + idx] += dl * qd.ptr[idx];
            }
        }
    }

    /* Write gradients back to Python objects */
    darr_sync(&qg, qg_obj);
    darr_done(&og); darr_done(&qd); darr_done(&qg);
    flat_to_rows(kgbuf, kgrad_list, T, dim);  /* write key gradients */
    flat_to_rows(vgbuf, vgrad_list, T, dim);  /* write value gradients */
    free(kbuf); free(vbuf); free(kgbuf); free(vgbuf);
    free(aw); free(d_attn);
    Py_RETURN_NONE;
}

/*
 * embedding_flat: Extract a row from a flat embedding matrix
 * 
 * Operation: Returns embeddings[idx] where embeddings is stored as a flat 1D array
 * 
 * Memory Layout:
 * - data: flat array of shape (vocab_size * dim)
 * - idx: row index to extract
 * - Returns: array.array('d') of shape (dim,) pointing to data[idx*dim : (idx+1)*dim]
 * 
 * This avoids storing embeddings as a 2D structure, using flat layout instead.
 * 
 * Returns: array.array('d') view of the requested embedding row
 */
/* ==== embedding_flat: extract row idx from flat buffer ==== */
static PyObject* fastops_embedding_flat(PyObject* self, PyObject* args) {
    PyObject *data_obj;
    Py_ssize_t idx, dim;
    if (!PyArg_ParseTuple(args, "Onn", &data_obj, &idx, &dim)) return NULL;
    DArr d;
    if (darr_get(&d, data_obj, 0) < 0) return NULL;
    if (idx < 0 || (idx + 1) * dim > d.n) {
        darr_done(&d);
        PyErr_Format(PyExc_IndexError, "embedding index %zd out of range for buffer of %zd elements with dim %zd", idx, d.n, dim);
        return NULL;
    }
    PyObject *result = darr_new(d.ptr + idx * dim, dim);
    darr_done(&d);
    return result;
}

/*
 * matvec_flat: Matrix-vector multiplication with flat weight matrix
 * 
 * Operation: out = W @ x where W is stored as a flat 1D array
 * 
 * Memory Layout:
 * - W: flat array of shape (nout * nin) in row-major order
 *   W[i*nin + j] is the weight connecting input j to output i
 * - x: input vector of shape (nin,)
 * - out: output vector of shape (nout,)
 * 
 * This flat layout is cache-friendly and avoids pointer indirection.
 * 
 * Memory: Uses malloc for output buffer
 * 
 * Returns: array.array('d') of shape (nout,)
 */
/* ==== matvec_flat: out = W @ x, W is flat (nout*nin) ==== */
static PyObject* fastops_matvec_flat(PyObject* self, PyObject* args) {
    PyObject *W_obj, *x_obj;
    int nout, nin;
    if (!PyArg_ParseTuple(args, "OOii", &W_obj, &x_obj, &nout, &nin)) return NULL;

    DArr W, x;
    if (darr_get(&W, W_obj, 0) < 0) return NULL;
    if (darr_get(&x, x_obj, 0) < 0) { darr_done(&W); return NULL; }

    double *out = (double *)malloc(nout * sizeof(double));
    if (!out) { darr_done(&W); darr_done(&x); return PyErr_NoMemory(); }

    for (int i = 0; i < nout; i++) {
        double s = 0.0;
        const double *wi = W.ptr + i * nin;
        for (int j = 0; j < nin; j++)
            s += wi[j] * x.ptr[j];
        out[i] = s;
    }

    darr_done(&W); darr_done(&x);
    PyObject *result = darr_new(out, nout);
    free(out);
    return result;
}

/*
 * linear_backward_flat: Backprop through linear layer with flat weights
 * 
 * Operation: Given output gradient (og), computes and accumulates:
 *   - Weight gradient: Wgrad[i,j] += og[i] * x[j]
 *   - Input gradient: xg[j] += sum_i(og[i] * W[i,j])
 * 
 * Memory Layout:
 * - W, Wgrad: flat arrays of shape (nout * nin) in row-major order
 * - xd: input data of shape (nin,)
 * - og: output gradient of shape (nout,)
 * - xg: input gradient of shape (nin,) - accumulated in-place
 * 
 * Optimization: Skips computation when og[i] == 0 (no gradient from that output)
 * 
 * Memory: Works in-place on gradient buffers, no additional allocation
 */
/* ==== linear_backward_flat: W and Wgrad are flat (nout*nin) ==== */
static PyObject* fastops_linear_backward_flat(PyObject* self, PyObject* args) {
    PyObject *og_obj, *W_obj, *Wgrad_obj, *xd_obj, *xg_obj;
    int nout, nin;
    if (!PyArg_ParseTuple(args, "OOOOOii", &og_obj, &W_obj, &Wgrad_obj,
                          &xd_obj, &xg_obj, &nout, &nin))
        return NULL;

    DArr og, W, Wg, xd, xg;
    if (darr_get(&og, og_obj, 0) < 0) return NULL;
    if (darr_get(&W, W_obj, 0) < 0) { darr_done(&og); return NULL; }
    if (darr_get(&Wg, Wgrad_obj, 1) < 0) { darr_done(&og); darr_done(&W); return NULL; }
    if (darr_get(&xd, xd_obj, 0) < 0) { darr_done(&og); darr_done(&W); darr_done(&Wg); return NULL; }
    if (darr_get(&xg, xg_obj, 1) < 0) { darr_done(&og); darr_done(&W); darr_done(&Wg); darr_done(&xd); return NULL; }

    for (int i = 0; i < nout; i++) {
        double gi = og.ptr[i];
        if (gi == 0.0) continue;
        const double *wi = W.ptr + i * nin;
        double *wgi = Wg.ptr + i * nin;
        for (int j = 0; j < nin; j++) {
            wgi[j] += gi * xd.ptr[j];
            xg.ptr[j] += gi * wi[j];
        }
    }

    darr_sync(&Wg, Wgrad_obj);
    darr_sync(&xg, xg_obj);
    darr_done(&og); darr_done(&W); darr_done(&Wg); darr_done(&xd); darr_done(&xg);
    Py_RETURN_NONE;
}

/*
 * adam_update_flat: Adam optimizer update for flat parameter arrays
 * 
 * Operation: Updates parameters using the Adam optimization algorithm
 *   m = beta1 * m + (1 - beta1) * grad        [first moment]
 *   v = beta2 * v + (1 - beta2) * grad^2      [second moment]
 *   param -= lr_t * (m / bc1) / (sqrt(v / bc2) + eps)
 * 
 * Parameters:
 * - lr_t: learning rate (typically lr * sqrt(1-beta2^t) / (1-beta1^t))
 * - beta1, beta2: exponential decay rates for moment estimates
 * - bc1, bc2: bias correction factors (1-beta1^t, 1-beta2^t)
 * - eps: small constant for numerical stability (typically 1e-8)
 * 
 * Memory Layout: All arrays are flat 1D of same length
 * - pdata: parameter values (updated in-place)
 * - pgrad: parameter gradients (read-only)
 * - pm: first moment estimates (updated in-place)
 * - pv: second moment estimates (updated in-place)
 * 
 * Memory: Works in-place, no additional allocation
 * 
 * Note: Bias correction is applied via bc1, bc2 rather than correcting
 * moments directly, which is mathematically equivalent but more efficient.
 */
/* ==== adam_update_flat: all buffers are flat 1D arrays ==== */
static PyObject* fastops_adam_update_flat(PyObject* self, PyObject* args) {
    PyObject *pdata_obj, *pgrad_obj, *pm_obj, *pv_obj;
    double lr_t, beta1, beta2, bc1, bc2, eps;
    if (!PyArg_ParseTuple(args, "OOOOdddddd",
            &pdata_obj, &pgrad_obj, &pm_obj, &pv_obj,
            &lr_t, &beta1, &beta2, &bc1, &bc2, &eps))
        return NULL;

    DArr pd, pg, pmr, pvr;
    if (darr_get(&pd, pdata_obj, 1) < 0) return NULL;
    if (darr_get(&pg, pgrad_obj, 0) < 0) { darr_done(&pd); return NULL; }
    if (darr_get(&pmr, pm_obj, 1) < 0) { darr_done(&pd); darr_done(&pg); return NULL; }
    if (darr_get(&pvr, pv_obj, 1) < 0) { darr_done(&pd); darr_done(&pg); darr_done(&pmr); return NULL; }

    double one_m_b1 = 1.0 - beta1;
    double one_m_b2 = 1.0 - beta2;
    Py_ssize_t n = pd.n;

    for (Py_ssize_t j = 0; j < n; j++) {
        double g = pg.ptr[j];
        pmr.ptr[j] = beta1 * pmr.ptr[j] + one_m_b1 * g;
        pvr.ptr[j] = beta2 * pvr.ptr[j] + one_m_b2 * g * g;
        pd.ptr[j] -= lr_t * (pmr.ptr[j] / bc1) / (sqrt(pvr.ptr[j] / bc2) + eps);
    }

    darr_sync(&pd, pdata_obj);
    darr_sync(&pmr, pm_obj);
    darr_sync(&pvr, pv_obj);
    darr_done(&pd); darr_done(&pg); darr_done(&pmr); darr_done(&pvr);
    Py_RETURN_NONE;
}

/*
 * zero_grad_flat: Zero out a flat gradient array
 * 
 * Operation: Sets all elements of gradient buffer to 0.0
 * 
 * Used to clear gradients before a backward pass (gradient accumulation).
 * 
 * Memory: Uses memset for efficient zeroing, works in-place
 */
/* ==== zero_grad_flat: zero a flat 1D gradient array ==== */
static PyObject* fastops_zero_grad_flat(PyObject* self, PyObject* args) {
    PyObject *grad_obj;
    if (!PyArg_ParseTuple(args, "O", &grad_obj)) return NULL;
    DArr d;
    if (darr_get(&d, grad_obj, 1) < 0) return NULL;
    memset(d.ptr, 0, d.n * sizeof(double));
    darr_sync(&d, grad_obj);
    darr_done(&d);
    Py_RETURN_NONE;
}

/* ==== method table ==== */
static PyMethodDef FastopsMethods[] = {
    {"vec_dot",              fastops_vec_dot,              METH_VARARGS, "Dot product"},
    {"vec_axpy",             fastops_vec_axpy,             METH_VARARGS, "y += alpha * x"},
    {"rmsnorm_forward",      fastops_rmsnorm_forward,      METH_VARARGS, "RMSNorm forward"},
    {"rmsnorm_backward",     fastops_rmsnorm_backward,     METH_VARARGS, "RMSNorm backward"},
    {"squared_relu_forward", fastops_squared_relu_forward, METH_VARARGS, "Squared ReLU forward"},
    {"squared_relu_backward",fastops_squared_relu_backward,METH_VARARGS, "Squared ReLU backward"},
    {"tensor_add",           fastops_tensor_add,           METH_VARARGS, "Element-wise add"},
    {"tensor_add_backward",  fastops_tensor_add_backward,  METH_VARARGS, "Add backward"},
    {"cross_entropy_forward", fastops_cross_entropy_forward,METH_VARARGS, "Cross-entropy forward"},
    {"cross_entropy_backward",fastops_cross_entropy_backward,METH_VARARGS,"Cross-entropy backward"},
    {"attention_forward",    fastops_attention_forward,     METH_VARARGS, "Attention forward"},
    {"attention_backward",   fastops_attention_backward,    METH_VARARGS, "Attention backward"},
    {"embedding_flat",       fastops_embedding_flat,        METH_VARARGS, "Extract row from flat buffer"},
    {"matvec_flat",          fastops_matvec_flat,           METH_VARARGS, "Flat W @ x"},
    {"linear_backward_flat", fastops_linear_backward_flat,  METH_VARARGS, "Flat linear backward"},
    {"adam_update_flat",     fastops_adam_update_flat,       METH_VARARGS, "Flat Adam update"},
    {"zero_grad_flat",       fastops_zero_grad_flat,        METH_VARARGS, "Zero flat gradient"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastopsmodule = {
    PyModuleDef_HEAD_INIT, "fastops", NULL, -1, FastopsMethods
};

PyMODINIT_FUNC PyInit_fastops(void) {
    PyObject *mod = PyModule_Create(&fastopsmodule);
    if (!mod) return NULL;
    /* Cache array.array type for fast output array creation */
    PyObject *arr_mod = PyImport_ImportModule("array");
    if (arr_mod) {
        array_cls = PyObject_GetAttrString(arr_mod, "array");
        Py_DECREF(arr_mod);
    }
    return mod;
}
