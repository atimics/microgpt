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

typedef struct {
    double *ptr;
    Py_ssize_t n;
    Py_buffer view;
    int buffered; /* 1 = buffer protocol (zero-copy), 0 = malloc'd from list */
} DArr;

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
    /* Fallback: Python list of floats */
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

static void darr_done(DArr *d) {
    if (d->buffered)
        PyBuffer_Release(&d->view);
    else
        free(d->ptr);
}

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
    for (Py_ssize_t i = 0; i < n; i++) buf[i] /= total;

    double loss = -log(buf[target]);
    PyObject *probs = darr_new(buf, n);
    free(buf);

    PyObject *result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(loss));
    PyTuple_SET_ITEM(result, 1, probs);
    return result;
}

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

    double scale = sqrt((double)head_dim);

    for (int h = 0; h < n_head; h++) {
        Py_ssize_t hs = h * head_dim;
        double max_val = -1e30;
        for (Py_ssize_t t = 0; t < T; t++) {
            double s = 0.0;
            for (int j = 0; j < head_dim; j++)
                s += qd.ptr[hs + j] * kbuf[t * dim + hs + j];
            s /= scale;
            all_aw[h * T + t] = s;
            if (s > max_val) max_val = s;
        }
        double total = 0.0;
        for (Py_ssize_t t = 0; t < T; t++) {
            all_aw[h * T + t] = exp(all_aw[h * T + t] - max_val);
            total += all_aw[h * T + t];
        }
        for (Py_ssize_t t = 0; t < T; t++)
            all_aw[h * T + t] /= total;
        for (int j = 0; j < head_dim; j++) {
            double s = 0.0;
            for (Py_ssize_t t = 0; t < T; t++)
                s += all_aw[h * T + t] * vbuf[t * dim + hs + j];
            out[hs + j] = s;
        }
    }

    darr_done(&qd);
    PyObject *out_arr = darr_new(out, dim);

    /* build attention weights as list of lists (per head) for backward */
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
    PyTuple_SET_ITEM(result, 1, aw_list);
    return result;
}

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
    double *kgbuf = rows_to_flat(kgrad_list, T, dim);
    double *vgbuf = rows_to_flat(vgrad_list, T, dim);
    if (!kbuf || !vbuf || !kgbuf || !vgbuf) {
        darr_done(&og); darr_done(&qd); darr_done(&qg);
        free(kbuf); free(vbuf); free(kgbuf); free(vgbuf);
        return PyErr_NoMemory();
    }

    /* extract attention weights */
    double *aw = (double *)malloc(n_head * T * sizeof(double));
    double *d_attn = (double *)malloc(T * sizeof(double));
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

    for (int h = 0; h < n_head; h++) {
        Py_ssize_t hs = h * head_dim;
        memset(d_attn, 0, T * sizeof(double));
        for (int j = 0; j < head_dim; j++) {
            double g = og.ptr[hs + j];
            if (g == 0.0) continue;
            Py_ssize_t idx = hs + j;
            for (Py_ssize_t t = 0; t < T; t++) {
                vgbuf[t * dim + idx] += g * aw[h * T + t];
                d_attn[t] += g * vbuf[t * dim + idx];
            }
        }
        double dot = 0.0;
        for (Py_ssize_t t = 0; t < T; t++)
            dot += aw[h * T + t] * d_attn[t];
        for (Py_ssize_t t = 0; t < T; t++) {
            double dl = aw[h * T + t] * (d_attn[t] - dot) / scale;
            if (dl == 0.0) continue;
            for (int j = 0; j < head_dim; j++) {
                Py_ssize_t idx = hs + j;
                qg.ptr[idx] += dl * kbuf[t * dim + idx];
                kgbuf[t * dim + idx] += dl * qd.ptr[idx];
            }
        }
    }

    /* write back grads */
    darr_sync(&qg, qg_obj);
    darr_done(&og); darr_done(&qd); darr_done(&qg);
    flat_to_rows(kgbuf, kgrad_list, T, dim);
    flat_to_rows(vgbuf, vgrad_list, T, dim);
    free(kbuf); free(vbuf); free(kgbuf); free(vgbuf);
    free(aw); free(d_attn);
    Py_RETURN_NONE;
}

/* ==== embedding_flat: extract row idx from flat buffer ==== */
static PyObject* fastops_embedding_flat(PyObject* self, PyObject* args) {
    PyObject *data_obj;
    Py_ssize_t idx, dim;
    if (!PyArg_ParseTuple(args, "Onn", &data_obj, &idx, &dim)) return NULL;
    DArr d;
    if (darr_get(&d, data_obj, 0) < 0) return NULL;
    PyObject *result = darr_new(d.ptr + idx * dim, dim);
    darr_done(&d);
    return result;
}

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
