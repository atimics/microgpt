#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

/*
 * Minimal C accelerator for microgpt's inner loops.
 * Three operations that cover ~65% of training runtime:
 *   vec_dot  - dot product of two Python lists
 *   vec_axpy - y[i] += alpha * x[i] (in-place scaled add)
 *   matvec   - out[i] = dot(W[i], x) for matrix W (list of lists)
 */

/* dot product: sum(a[i] * b[i] for i in range(n)) */
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

/* scaled add: y[i] += alpha * x[i] for all i (mutates y in-place) */
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
        PyList_SET_ITEM(y, i, new_val);  /* steals ref */
    }
    Py_RETURN_NONE;
}

/* matrix-vector: out[i] = dot(W[i], x) — returns new list */
static PyObject* fastops_matvec(PyObject* self, PyObject* args) {
    PyObject *W, *x;
    if (!PyArg_ParseTuple(args, "OO", &W, &x))
        return NULL;

    Py_ssize_t nrow = PyList_GET_SIZE(W);
    Py_ssize_t ncol = PyList_GET_SIZE(x);

    /* pre-extract x into a C array for speed */
    double *xbuf = (double *)malloc(ncol * sizeof(double));
    if (!xbuf) return PyErr_NoMemory();
    for (Py_ssize_t j = 0; j < ncol; j++)
        xbuf[j] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(x, j));

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

/*
 * linear_backward: the fused backward pass for linear layers.
 * Given output gradient og, weight matrix W, input data xd:
 *   wgrad[i] += og[i] * xd   (weight gradient)
 *   xgrad    += og[i] * W[i]  (input gradient)
 * Skips rows where og[i] == 0.0.
 */
static PyObject* fastops_linear_backward(PyObject* self, PyObject* args) {
    PyObject *og, *wd, *wgrad, *xd, *xgrad;
    if (!PyArg_ParseTuple(args, "OOOOO", &og, &wd, &wgrad, &xd, &xgrad))
        return NULL;

    Py_ssize_t n_out = PyList_GET_SIZE(og);
    Py_ssize_t n_in = PyList_GET_SIZE(xd);

    /* pre-extract xd into C array */
    double *xbuf = (double *)malloc(n_in * sizeof(double));
    if (!xbuf) return PyErr_NoMemory();
    for (Py_ssize_t j = 0; j < n_in; j++)
        xbuf[j] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(xd, j));

    /* pre-extract xgrad into C array */
    double *xgbuf = (double *)malloc(n_in * sizeof(double));
    if (!xgbuf) { free(xbuf); return PyErr_NoMemory(); }
    for (Py_ssize_t j = 0; j < n_in; j++)
        xgbuf[j] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(xgrad, j));

    for (Py_ssize_t i = 0; i < n_out; i++) {
        double gi = PyFloat_AS_DOUBLE(PyList_GET_ITEM(og, i));
        if (gi == 0.0) continue;

        PyObject *wi = PyList_GET_ITEM(wd, i);
        PyObject *wgi = PyList_GET_ITEM(wgrad, i);

        for (Py_ssize_t j = 0; j < n_in; j++) {
            double wij = PyFloat_AS_DOUBLE(PyList_GET_ITEM(wi, j));
            double wgij = PyFloat_AS_DOUBLE(PyList_GET_ITEM(wgi, j));
            /* wgrad[i][j] += gi * xd[j] */
            PyObject *new_wg = PyFloat_FromDouble(wgij + gi * xbuf[j]);
            PyList_SET_ITEM(wgi, j, new_wg);
            /* xgrad[j] += gi * W[i][j] */
            xgbuf[j] += gi * wij;
        }
    }

    /* write xgrad back */
    for (Py_ssize_t j = 0; j < n_in; j++) {
        PyObject *new_xg = PyFloat_FromDouble(xgbuf[j]);
        PyList_SET_ITEM(xgrad, j, new_xg);
    }

    free(xbuf);
    free(xgbuf);
    Py_RETURN_NONE;
}

static PyMethodDef FastopsMethods[] = {
    {"vec_dot", fastops_vec_dot, METH_VARARGS, "Dot product of two lists"},
    {"vec_axpy", fastops_vec_axpy, METH_VARARGS, "y += alpha * x (in-place)"},
    {"matvec", fastops_matvec, METH_VARARGS, "Matrix-vector multiply W @ x"},
    {"linear_backward", fastops_linear_backward, METH_VARARGS, "Fused linear backward pass"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastopsmodule = {
    PyModuleDef_HEAD_INIT, "fastops", NULL, -1, FastopsMethods
};

PyMODINIT_FUNC PyInit_fastops(void) {
    return PyModule_Create(&fastopsmodule);
}
