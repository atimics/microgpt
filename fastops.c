#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>

/*
 * C accelerator for microgpt's inner loops.
 * Covers the major hot paths: linear, attention, rmsnorm, activation,
 * cross-entropy, tensor_add, and Adam optimizer.
 */

/* ---- helpers ---- */
static inline double* list_to_buf(PyObject *list, Py_ssize_t n) {
    double *buf = (double *)malloc(n * sizeof(double));
    if (!buf) return NULL;
    for (Py_ssize_t i = 0; i < n; i++)
        buf[i] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(list, i));
    return buf;
}

static inline void buf_to_list(const double *buf, PyObject *list, Py_ssize_t n) {
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *old = PyList_GET_ITEM(list, i);
        PyObject *nv = PyFloat_FromDouble(buf[i]);
        Py_DECREF(old);
        Py_INCREF(nv);
        PyList_SET_ITEM(list, i, nv);
    }
}

static inline void buf_to_new_list(const double *buf, PyObject *list, Py_ssize_t n) {
    for (Py_ssize_t i = 0; i < n; i++)
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(buf[i]));
}

/* ---- dot product ---- */
static PyObject* fastops_vec_dot(PyObject* self, PyObject* args) {
    PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "OO", &a, &b))
        return NULL;

    Py_ssize_t n = PyList_GET_SIZE(a);
    double s = 0.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        double ai = PyFloat_AS_DOUBLE(PyList_GET_ITEM(a, i));
        double bi = PyFloat_AS_DOUBLE(PyList_GET_ITEM(b, i));
        s += ai * bi;
    }
    return PyFloat_FromDouble(s);
}

/* ---- scaled add: y[i] += alpha * x[i] ---- */
static PyObject* fastops_vec_axpy(PyObject* self, PyObject* args) {
    double alpha;
    PyObject *x, *y;
    if (!PyArg_ParseTuple(args, "dOO", &alpha, &x, &y))
        return NULL;

    Py_ssize_t n = PyList_GET_SIZE(x);
    for (Py_ssize_t i = 0; i < n; i++) {
        double xi = PyFloat_AS_DOUBLE(PyList_GET_ITEM(x, i));
        double yi = PyFloat_AS_DOUBLE(PyList_GET_ITEM(y, i));
        PyObject *new_val = PyFloat_FromDouble(yi + alpha * xi);
        PyList_SET_ITEM(y, i, new_val);
    }
    Py_RETURN_NONE;
}

/* ---- matrix-vector: out[i] = dot(W[i], x) ---- */
static PyObject* fastops_matvec(PyObject* self, PyObject* args) {
    PyObject *W, *x;
    if (!PyArg_ParseTuple(args, "OO", &W, &x))
        return NULL;

    Py_ssize_t nrow = PyList_GET_SIZE(W);
    Py_ssize_t ncol = PyList_GET_SIZE(x);

    double *xbuf = list_to_buf(x, ncol);
    if (!xbuf) return PyErr_NoMemory();

    PyObject *out = PyList_New(nrow);
    if (!out) { free(xbuf); return NULL; }

    for (Py_ssize_t i = 0; i < nrow; i++) {
        PyObject *wi = PyList_GET_ITEM(W, i);
        double s = 0.0;
        for (Py_ssize_t j = 0; j < ncol; j++)
            s += PyFloat_AS_DOUBLE(PyList_GET_ITEM(wi, j)) * xbuf[j];
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(s));
    }

    free(xbuf);
    return out;
}

/* ---- fused linear backward ---- */
static PyObject* fastops_linear_backward(PyObject* self, PyObject* args) {
    PyObject *og, *wd, *wgrad, *xd, *xgrad;
    if (!PyArg_ParseTuple(args, "OOOOO", &og, &wd, &wgrad, &xd, &xgrad))
        return NULL;

    Py_ssize_t n_out = PyList_GET_SIZE(og);
    Py_ssize_t n_in = PyList_GET_SIZE(xd);

    double *xbuf = list_to_buf(xd, n_in);
    if (!xbuf) return PyErr_NoMemory();

    double *xgbuf = list_to_buf(xgrad, n_in);
    if (!xgbuf) { free(xbuf); return PyErr_NoMemory(); }

    /* Temporary buffers for one row of W and wgrad to avoid per-element PyObject ops */
    double *wrow = (double *)malloc(n_in * sizeof(double));
    double *wgrow = (double *)malloc(n_in * sizeof(double));
    if (!wrow || !wgrow) { free(xbuf); free(xgbuf); free(wrow); free(wgrow); return PyErr_NoMemory(); }

    for (Py_ssize_t i = 0; i < n_out; i++) {
        double gi = PyFloat_AS_DOUBLE(PyList_GET_ITEM(og, i));
        if (gi == 0.0) continue;

        PyObject *wi = PyList_GET_ITEM(wd, i);
        PyObject *wgi = PyList_GET_ITEM(wgrad, i);

        /* Extract row to C buffer */
        for (Py_ssize_t j = 0; j < n_in; j++) {
            wrow[j] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(wi, j));
            wgrow[j] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(wgi, j));
        }

        /* Pure C inner loop - no Python API calls */
        for (Py_ssize_t j = 0; j < n_in; j++) {
            wgrow[j] += gi * xbuf[j];
            xgbuf[j] += gi * wrow[j];
        }

        /* Write wgrad row back in bulk */
        buf_to_list(wgrow, wgi, n_in);
    }

    /* write xgrad back */
    buf_to_list(xgbuf, xgrad, n_in);

    free(xbuf);
    free(xgbuf);
    free(wrow);
    free(wgrow);
    Py_RETURN_NONE;
}

/* ---- RMSNorm forward: returns (out_list, scale_float) ---- */
static PyObject* fastops_rmsnorm_forward(PyObject* self, PyObject* args) {
    PyObject *xd;
    if (!PyArg_ParseTuple(args, "O", &xd))
        return NULL;

    Py_ssize_t n = PyList_GET_SIZE(xd);
    double ms = 0.0;
    double *xbuf = list_to_buf(xd, n);
    if (!xbuf) return PyErr_NoMemory();

    for (Py_ssize_t i = 0; i < n; i++)
        ms += xbuf[i] * xbuf[i];
    ms /= n;
    double scale = 1.0 / sqrt(ms + 1e-5);

    PyObject *out = PyList_New(n);
    if (!out) { free(xbuf); return NULL; }
    for (Py_ssize_t i = 0; i < n; i++)
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(xbuf[i] * scale));

    free(xbuf);
    PyObject *result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, out);
    PyTuple_SET_ITEM(result, 1, PyFloat_FromDouble(scale));
    return result;
}

/* ---- RMSNorm backward: mutates x.grad in-place ---- */
static PyObject* fastops_rmsnorm_backward(PyObject* self, PyObject* args) {
    PyObject *out_grad, *xd, *xgrad;
    double scale;
    if (!PyArg_ParseTuple(args, "OOdO", &out_grad, &xd, &scale, &xgrad))
        return NULL;

    Py_ssize_t n = PyList_GET_SIZE(xd);
    double *ogbuf = list_to_buf(out_grad, n);
    double *xbuf = list_to_buf(xd, n);
    double *xgbuf = list_to_buf(xgrad, n);
    if (!ogbuf || !xbuf || !xgbuf) {
        free(ogbuf); free(xbuf); free(xgbuf);
        return PyErr_NoMemory();
    }

    double dot = 0.0;
    for (Py_ssize_t i = 0; i < n; i++)
        dot += ogbuf[i] * xbuf[i];

    double s3n = scale * scale * scale / n;
    for (Py_ssize_t j = 0; j < n; j++)
        xgbuf[j] += ogbuf[j] * scale - s3n * xbuf[j] * dot;

    buf_to_list(xgbuf, xgrad, n);
    free(ogbuf); free(xbuf); free(xgbuf);
    Py_RETURN_NONE;
}

/* ---- Squared ReLU forward: returns new list ---- */
static PyObject* fastops_squared_relu_forward(PyObject* self, PyObject* args) {
    PyObject *xd;
    if (!PyArg_ParseTuple(args, "O", &xd))
        return NULL;

    Py_ssize_t n = PyList_GET_SIZE(xd);
    PyObject *out = PyList_New(n);
    if (!out) return NULL;

    for (Py_ssize_t i = 0; i < n; i++) {
        double xi = PyFloat_AS_DOUBLE(PyList_GET_ITEM(xd, i));
        double v = xi > 0.0 ? xi * xi : 0.0;
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(v));
    }
    return out;
}

/* ---- Squared ReLU backward: mutates x.grad in-place ---- */
static PyObject* fastops_squared_relu_backward(PyObject* self, PyObject* args) {
    PyObject *xd, *out_grad, *xgrad;
    if (!PyArg_ParseTuple(args, "OOO", &xd, &out_grad, &xgrad))
        return NULL;

    Py_ssize_t n = PyList_GET_SIZE(xd);
    for (Py_ssize_t i = 0; i < n; i++) {
        double xi = PyFloat_AS_DOUBLE(PyList_GET_ITEM(xd, i));
        if (xi > 0.0) {
            double gi = PyFloat_AS_DOUBLE(PyList_GET_ITEM(out_grad, i));
            double old = PyFloat_AS_DOUBLE(PyList_GET_ITEM(xgrad, i));
            PyObject *nv = PyFloat_FromDouble(old + 2.0 * xi * gi);
            PyList_SET_ITEM(xgrad, i, nv);
        }
    }
    Py_RETURN_NONE;
}

/* ---- tensor_add forward: returns new list ---- */
static PyObject* fastops_tensor_add(PyObject* self, PyObject* args) {
    PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "OO", &a, &b))
        return NULL;

    Py_ssize_t n = PyList_GET_SIZE(a);
    PyObject *out = PyList_New(n);
    if (!out) return NULL;

    for (Py_ssize_t i = 0; i < n; i++) {
        double ai = PyFloat_AS_DOUBLE(PyList_GET_ITEM(a, i));
        double bi = PyFloat_AS_DOUBLE(PyList_GET_ITEM(b, i));
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(ai + bi));
    }
    return out;
}

/* ---- tensor_add backward: mutates a.grad and b.grad in-place ---- */
static PyObject* fastops_tensor_add_backward(PyObject* self, PyObject* args) {
    PyObject *out_grad, *agrad, *bgrad;
    if (!PyArg_ParseTuple(args, "OOO", &out_grad, &agrad, &bgrad))
        return NULL;

    Py_ssize_t n = PyList_GET_SIZE(out_grad);
    for (Py_ssize_t i = 0; i < n; i++) {
        double g = PyFloat_AS_DOUBLE(PyList_GET_ITEM(out_grad, i));
        double ag = PyFloat_AS_DOUBLE(PyList_GET_ITEM(agrad, i));
        double bg = PyFloat_AS_DOUBLE(PyList_GET_ITEM(bgrad, i));
        PyList_SET_ITEM(agrad, i, PyFloat_FromDouble(ag + g));
        PyList_SET_ITEM(bgrad, i, PyFloat_FromDouble(bg + g));
    }
    Py_RETURN_NONE;
}

/* ---- Cross entropy forward: returns (loss, probs_list) ---- */
static PyObject* fastops_cross_entropy_forward(PyObject* self, PyObject* args) {
    PyObject *logits;
    Py_ssize_t target;
    if (!PyArg_ParseTuple(args, "On", &logits, &target))
        return NULL;

    Py_ssize_t n = PyList_GET_SIZE(logits);
    double *buf = list_to_buf(logits, n);
    if (!buf) return PyErr_NoMemory();

    /* softmax */
    double max_val = buf[0];
    for (Py_ssize_t i = 1; i < n; i++)
        if (buf[i] > max_val) max_val = buf[i];

    double total = 0.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        buf[i] = exp(buf[i] - max_val);
        total += buf[i];
    }
    for (Py_ssize_t i = 0; i < n; i++)
        buf[i] /= total;

    double loss = -log(buf[target]);

    PyObject *probs = PyList_New(n);
    if (!probs) { free(buf); return NULL; }
    buf_to_new_list(buf, probs, n);

    free(buf);
    PyObject *result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(loss));
    PyTuple_SET_ITEM(result, 1, probs);
    return result;
}

/* ---- Cross entropy backward: mutates logits.grad in-place ---- */
static PyObject* fastops_cross_entropy_backward(PyObject* self, PyObject* args) {
    PyObject *probs, *logits_grad;
    double g;
    Py_ssize_t target;
    if (!PyArg_ParseTuple(args, "dOnO", &g, &probs, &target, &logits_grad))
        return NULL;

    Py_ssize_t n = PyList_GET_SIZE(probs);
    for (Py_ssize_t i = 0; i < n; i++) {
        double pi = PyFloat_AS_DOUBLE(PyList_GET_ITEM(probs, i));
        double old = PyFloat_AS_DOUBLE(PyList_GET_ITEM(logits_grad, i));
        double delta = (i == target) ? (pi - 1.0) : pi;
        PyList_SET_ITEM(logits_grad, i, PyFloat_FromDouble(old + g * delta));
    }
    Py_RETURN_NONE;
}

/*
 * Adam optimizer update for one Param (2D weight matrix).
 * Mutates data, m, v in-place.  Reads grad.
 * adam_update(data, grad, m, v, lr_t, beta1, beta2, bc1, bc2, eps)
 *   data, grad, m, v are list-of-lists (nout x nin).
 */
static PyObject* fastops_adam_update(PyObject* self, PyObject* args) {
    PyObject *pdata, *pgrad, *pm, *pv;
    double lr_t, beta1, beta2, bc1, bc2, eps;
    if (!PyArg_ParseTuple(args, "OOOOdddddd",
            &pdata, &pgrad, &pm, &pv,
            &lr_t, &beta1, &beta2, &bc1, &bc2, &eps))
        return NULL;

    Py_ssize_t nout = PyList_GET_SIZE(pdata);
    double one_m_b1 = 1.0 - beta1;
    double one_m_b2 = 1.0 - beta2;

    /* Pre-determine row size from first row */
    if (nout == 0) Py_RETURN_NONE;
    Py_ssize_t nin = PyList_GET_SIZE(PyList_GET_ITEM(pdata, 0));

    /* Allocate row buffers once */
    double *db = (double *)malloc(nin * sizeof(double));
    double *gb = (double *)malloc(nin * sizeof(double));
    double *mb = (double *)malloc(nin * sizeof(double));
    double *vb = (double *)malloc(nin * sizeof(double));
    if (!db || !gb || !mb || !vb) {
        free(db); free(gb); free(mb); free(vb);
        return PyErr_NoMemory();
    }

    for (Py_ssize_t i = 0; i < nout; i++) {
        PyObject *pd = PyList_GET_ITEM(pdata, i);
        PyObject *pg = PyList_GET_ITEM(pgrad, i);
        PyObject *pmr = PyList_GET_ITEM(pm, i);
        PyObject *pvr = PyList_GET_ITEM(pv, i);

        /* Extract to C buffers */
        for (Py_ssize_t j = 0; j < nin; j++) {
            db[j] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(pd, j));
            gb[j] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(pg, j));
            mb[j] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(pmr, j));
            vb[j] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(pvr, j));
        }

        /* Pure C computation - no Python API calls in inner loop */
        for (Py_ssize_t j = 0; j < nin; j++) {
            double g = gb[j];
            mb[j] = beta1 * mb[j] + one_m_b1 * g;
            vb[j] = beta2 * vb[j] + one_m_b2 * g * g;
            db[j] -= lr_t * (mb[j] / bc1) / (sqrt(vb[j] / bc2) + eps);
        }

        /* Write back in bulk */
        buf_to_list(db, pd, nin);
        buf_to_list(mb, pmr, nin);
        buf_to_list(vb, pvr, nin);
    }

    free(db); free(gb); free(mb); free(vb);
    Py_RETURN_NONE;
}

/*
 * Fused attention forward.
 * attention_forward(q_data, key_data_list, val_data_list, n_head, head_dim)
 *   q_data     : list[float] of length n_head * head_dim
 *   key_data_list : list of list[float], each of length n_head * head_dim
 *   val_data_list : list of list[float], each of length n_head * head_dim
 * Returns (out_data: list[float], attn_weights: list of list[float] per head)
 */
static PyObject* fastops_attention_forward(PyObject* self, PyObject* args) {
    PyObject *qd_obj, *keys_list, *vals_list;
    int n_head, head_dim;
    if (!PyArg_ParseTuple(args, "OOOii", &qd_obj, &keys_list, &vals_list, &n_head, &head_dim))
        return NULL;

    Py_ssize_t T = PyList_GET_SIZE(keys_list);
    Py_ssize_t dim = n_head * head_dim;

    double *qd = list_to_buf(qd_obj, dim);
    if (!qd) return PyErr_NoMemory();

    /* extract all key and value data into contiguous C arrays */
    double *kbuf = (double *)malloc(T * dim * sizeof(double));
    double *vbuf = (double *)malloc(T * dim * sizeof(double));
    if (!kbuf || !vbuf) {
        free(qd); free(kbuf); free(vbuf);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t t = 0; t < T; t++) {
        PyObject *krow = PyList_GET_ITEM(keys_list, t);
        PyObject *vrow = PyList_GET_ITEM(vals_list, t);
        for (Py_ssize_t j = 0; j < dim; j++) {
            kbuf[t * dim + j] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(krow, j));
            vbuf[t * dim + j] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(vrow, j));
        }
    }

    double *out = (double *)calloc(dim, sizeof(double));
    /* store all attention weights for backward */
    double *all_aw = (double *)malloc(n_head * T * sizeof(double));
    if (!out || !all_aw) {
        free(qd); free(kbuf); free(vbuf); free(out); free(all_aw);
        return PyErr_NoMemory();
    }

    double scale = sqrt((double)head_dim);

    for (int h = 0; h < n_head; h++) {
        Py_ssize_t hs = h * head_dim;
        /* compute attention logits */
        double max_val = -1e30;
        for (Py_ssize_t t = 0; t < T; t++) {
            double s = 0.0;
            for (int j = 0; j < head_dim; j++)
                s += qd[hs + j] * kbuf[t * dim + hs + j];
            s /= scale;
            all_aw[h * T + t] = s;
            if (s > max_val) max_val = s;
        }
        /* softmax */
        double total = 0.0;
        for (Py_ssize_t t = 0; t < T; t++) {
            all_aw[h * T + t] = exp(all_aw[h * T + t] - max_val);
            total += all_aw[h * T + t];
        }
        for (Py_ssize_t t = 0; t < T; t++)
            all_aw[h * T + t] /= total;
        /* weighted sum of values */
        for (int j = 0; j < head_dim; j++) {
            double s = 0.0;
            for (Py_ssize_t t = 0; t < T; t++)
                s += all_aw[h * T + t] * vbuf[t * dim + hs + j];
            out[hs + j] = s;
        }
    }

    /* build output list */
    PyObject *out_list = PyList_New(dim);
    if (!out_list) { free(qd); free(kbuf); free(vbuf); free(out); free(all_aw); return NULL; }
    buf_to_new_list(out, out_list, dim);

    /* build attention weights as list of lists (per head) */
    PyObject *aw_list = PyList_New(n_head);
    for (int h = 0; h < n_head; h++) {
        PyObject *hw = PyList_New(T);
        for (Py_ssize_t t = 0; t < T; t++)
            PyList_SET_ITEM(hw, t, PyFloat_FromDouble(all_aw[h * T + t]));
        PyList_SET_ITEM(aw_list, h, hw);
    }

    free(qd); free(kbuf); free(vbuf); free(out); free(all_aw);

    PyObject *result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, out_list);
    PyTuple_SET_ITEM(result, 1, aw_list);
    return result;
}

/*
 * Fused attention backward.
 * attention_backward(out_grad, q_data, key_data_list, val_data_list,
 *                    attn_weights, q_grad, key_grad_list, val_grad_list,
 *                    n_head, head_dim)
 * Mutates q_grad, key_grad_list entries, val_grad_list entries in-place.
 */
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

    /* extract to C buffers */
    double *og = list_to_buf(og_obj, dim);
    double *qd = list_to_buf(qd_obj, dim);
    double *qg = list_to_buf(qg_obj, dim);
    if (!og || !qd || !qg) {
        free(og); free(qd); free(qg);
        return PyErr_NoMemory();
    }

    double *kbuf = (double *)malloc(T * dim * sizeof(double));
    double *vbuf = (double *)malloc(T * dim * sizeof(double));
    double *kgbuf = (double *)malloc(T * dim * sizeof(double));
    double *vgbuf = (double *)malloc(T * dim * sizeof(double));
    if (!kbuf || !vbuf || !kgbuf || !vgbuf) {
        free(og); free(qd); free(qg);
        free(kbuf); free(vbuf); free(kgbuf); free(vgbuf);
        return PyErr_NoMemory();
    }

    for (Py_ssize_t t = 0; t < T; t++) {
        PyObject *krow = PyList_GET_ITEM(keys_list, t);
        PyObject *vrow = PyList_GET_ITEM(vals_list, t);
        PyObject *kgrow = PyList_GET_ITEM(kgrad_list, t);
        PyObject *vgrow = PyList_GET_ITEM(vgrad_list, t);
        for (Py_ssize_t j = 0; j < dim; j++) {
            kbuf[t * dim + j] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(krow, j));
            vbuf[t * dim + j] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(vrow, j));
            kgbuf[t * dim + j] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(kgrow, j));
            vgbuf[t * dim + j] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(vgrow, j));
        }
    }

    /* extract attention weights */
    double *aw = (double *)malloc(n_head * T * sizeof(double));
    if (!aw) {
        free(og); free(qd); free(qg); free(kbuf); free(vbuf); free(kgbuf); free(vgbuf);
        return PyErr_NoMemory();
    }
    for (int h = 0; h < n_head; h++) {
        PyObject *hw = PyList_GET_ITEM(aw_list, h);
        for (Py_ssize_t t = 0; t < T; t++)
            aw[h * T + t] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(hw, t));
    }

    double *d_attn = (double *)malloc(T * sizeof(double));
    if (!d_attn) {
        free(og); free(qd); free(qg); free(kbuf); free(vbuf);
        free(kgbuf); free(vgbuf); free(aw);
        return PyErr_NoMemory();
    }

    for (int h = 0; h < n_head; h++) {
        Py_ssize_t hs = h * head_dim;
        memset(d_attn, 0, T * sizeof(double));

        /* d_attn[t] += g * v_data[t][idx]  and  v_grad[t][idx] += g * aw[t] */
        for (int j = 0; j < head_dim; j++) {
            double g = og[hs + j];
            if (g == 0.0) continue;
            Py_ssize_t idx = hs + j;
            for (Py_ssize_t t = 0; t < T; t++) {
                vgbuf[t * dim + idx] += g * aw[h * T + t];
                d_attn[t] += g * vbuf[t * dim + idx];
            }
        }

        /* softmax backward */
        double dot = 0.0;
        for (Py_ssize_t t = 0; t < T; t++)
            dot += aw[h * T + t] * d_attn[t];

        for (Py_ssize_t t = 0; t < T; t++) {
            double dl = aw[h * T + t] * (d_attn[t] - dot) / scale;
            if (dl == 0.0) continue;
            for (int j = 0; j < head_dim; j++) {
                Py_ssize_t idx = hs + j;
                qg[idx] += dl * kbuf[t * dim + idx];
                kgbuf[t * dim + idx] += dl * qd[idx];
            }
        }
    }

    /* write back */
    buf_to_list(qg, qg_obj, dim);
    for (Py_ssize_t t = 0; t < T; t++) {
        PyObject *kgrow = PyList_GET_ITEM(kgrad_list, t);
        PyObject *vgrow = PyList_GET_ITEM(vgrad_list, t);
        buf_to_list(kgbuf + t * dim, kgrow, dim);
        buf_to_list(vgbuf + t * dim, vgrow, dim);
    }

    free(og); free(qd); free(qg);
    free(kbuf); free(vbuf); free(kgbuf); free(vgbuf);
    free(aw); free(d_attn);
    Py_RETURN_NONE;
}

/*
 * Fused zero_grad: sets all elements of a 2D list-of-lists to 0.0
 * zero_grad(grad) where grad is list[list[float]]
 */
static PyObject* fastops_zero_grad(PyObject* self, PyObject* args) {
    PyObject *grad;
    if (!PyArg_ParseTuple(args, "O", &grad))
        return NULL;

    Py_ssize_t nout = PyList_GET_SIZE(grad);
    for (Py_ssize_t i = 0; i < nout; i++) {
        PyObject *row = PyList_GET_ITEM(grad, i);
        Py_ssize_t nin = PyList_GET_SIZE(row);
        /* Replace entire row with a fresh list of zeros — faster than per-element */
        PyObject *new_row = PyList_New(nin);
        if (!new_row) return NULL;
        for (Py_ssize_t j = 0; j < nin; j++)
            PyList_SET_ITEM(new_row, j, PyFloat_FromDouble(0.0));
        /* Replace in the outer list */
        Py_INCREF(new_row);
        PyObject *old = PyList_GET_ITEM(grad, i);
        Py_DECREF(old);
        PyList_SET_ITEM(grad, i, new_row);
    }
    Py_RETURN_NONE;
}


static PyMethodDef FastopsMethods[] = {
    {"vec_dot",             fastops_vec_dot,             METH_VARARGS, "Dot product of two lists"},
    {"vec_axpy",            fastops_vec_axpy,            METH_VARARGS, "y += alpha * x (in-place)"},
    {"matvec",              fastops_matvec,              METH_VARARGS, "Matrix-vector multiply W @ x"},
    {"linear_backward",     fastops_linear_backward,     METH_VARARGS, "Fused linear backward pass"},
    {"rmsnorm_forward",     fastops_rmsnorm_forward,     METH_VARARGS, "RMSNorm forward pass"},
    {"rmsnorm_backward",    fastops_rmsnorm_backward,    METH_VARARGS, "RMSNorm backward pass"},
    {"squared_relu_forward", fastops_squared_relu_forward, METH_VARARGS, "Squared ReLU forward"},
    {"squared_relu_backward", fastops_squared_relu_backward, METH_VARARGS, "Squared ReLU backward"},
    {"tensor_add",          fastops_tensor_add,          METH_VARARGS, "Element-wise add"},
    {"tensor_add_backward", fastops_tensor_add_backward, METH_VARARGS, "Element-wise add backward"},
    {"cross_entropy_forward", fastops_cross_entropy_forward, METH_VARARGS, "Cross-entropy forward"},
    {"cross_entropy_backward", fastops_cross_entropy_backward, METH_VARARGS, "Cross-entropy backward"},
    {"adam_update",         fastops_adam_update,          METH_VARARGS, "Adam optimizer update"},
    {"attention_forward",   fastops_attention_forward,    METH_VARARGS, "Fused attention forward"},
    {"attention_backward",  fastops_attention_backward,   METH_VARARGS, "Fused attention backward"},
    {"zero_grad",           fastops_zero_grad,            METH_VARARGS, "Zero out 2D gradient"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastopsmodule = {
    PyModuleDef_HEAD_INIT, "fastops", NULL, -1, FastopsMethods
};

PyMODINIT_FUNC PyInit_fastops(void) {
    return PyModule_Create(&fastopsmodule);
}
