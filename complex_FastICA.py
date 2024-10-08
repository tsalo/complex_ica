"""
Author: Alex Bujan
Adapted from: Ella Bingham, 1999

Original article citation:
Ella Bingham and Aapo Hyvaerinen, "A fast fixed-point algorithm for
independent component analysis of complex valued signals",
International Journal of Neural Systems, Vol. 10, No. 1 (February, 2000) 1-8

Original code url:
http://users.ics.aalto.fi/ella/publications/cfastica_public.m

Date: 12/11/2015

TODO: include arbitrary contrast functions
"""

import warnings

import numpy as np


def abs_sqr(W, X):
    return abs(W.conj().T.dot(X)) ** 2


def _custom_contrast(x, kwargs):
    epsilon = kwargs.get("epsilon", 0.1)
    # derivative of the contrast function
    g_ = 1 / (epsilon + x)
    # derivative of g
    dg_ = -1 / (epsilon + x) ** 2
    return g_, dg_


def _ica_par(X, tol, g, fun_args, max_iter, w_init):
    """Parallel FastICA.

    Used internally by FastICA --main loop

    """
    n_components = w_init.shape[0]
    W = w_init

    # cache the covariance matrix
    C = np.cov(X)

    for ii in range(max_iter):
        # NOTE: Instead of dot product, we use abs(W.conj().T.dot(X)) ** 2
        gwtx, g_wtx = g(abs_sqr(W, X), fun_args)
        W1 = (X * (W.conj().T.dot(X)).conj() * gwtx).mean(1).reshape(
            (n_components, 1)
        ) - (gwtx + abs_sqr(W, X) * g_wtx).mean() * W
        # was W1 = _sym_decorrelation(fast_dot(gwtx, X.T) / p_
        #                             - g_wtx[:, np.newaxis] * W)

        del gwtx, g_wtx

        # Symmetric decorrelation
        Uw, Sw = np.linalg.eig(W1.conj().T.dot(C.dot(W1)))
        W1 = W1.dot(Sw.dot(np.linalg.inv(np.sqrt(np.diag(Uw))).dot(Sw.conj().T)))
        del Uw, Sw

        lim = np.abs(np.abs((W1 * W).sum()) - 1)
        W = W1
        if lim < tol:
            break

    if (ii + 1) == max_iter and lim > tol:
        warnings.warn(
            "FastICA did not converge. Consider increasing "
            "tolerance or the maximum number of iterations."
        )

    return W, ii + 1


def _ica_def(X, tol, g, fun_args, max_iter, w_init):
    """Deflationary FastICA using fun approx to neg-entropy function

    Used internally by FastICA.
    """

    n_components = w_init.shape[0]
    W = np.zeros((n_components, n_components), dtype=X.dtype)
    n_iter = []

    # j is the index of the extracted component
    for j in range(n_components):
        w = w_init[j, :].copy()
        w = w[:, None]  # needs to be 2D

        w /= np.linalg.norm(w)
        # was w /= np.sqrt((w ** 2).sum())

        for i in range(max_iter):
            # NOTE: Instead of dot product, we use abs(W.conj().T.dot(X)) ** 2
            gwtx, g_wtx = g(abs_sqr(w, X), fun_args)

            w1 = (X * (w.conj().T.dot(X)).conj() * gwtx).mean(1).reshape(
                (n_components, 1)
            ) - (gwtx + abs_sqr(w, X) * g_wtx).mean() * w
            # was w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w1

            del gwtx, g_wtx

            w1 /= np.linalg.norm(w1)
            # was w1 /= np.sqrt((w1 ** 2).sum())

            # Decorrelation (complex version only?)
            w1 -= W.dot(W.conj().T).dot(w1)
            w1 /= np.linalg.norm(w1)

            lim = np.abs(np.abs((w1 * w).sum()) - 1)
            w = w1
            if lim < tol:
                break

        n_iter.append(i + 1)
        W[j, :] = np.squeeze(w)

        if n_iter == max_iter and lim > tol:
            warnings.warn(
                "FastICA did not converge. Consider increasing "
                "tolerance or the maximum number of iterations."
            )

    return W, max(n_iter)


def complex_FastICA(
    X,
    n_components=None,
    algorithm="parallel",
    whiten=True,
    fun="custom",
    fun_args=None,
    max_iter=100,
    tol=1e-4,
    w_init=None,
    random_state=None,
    return_X_mean=False,
    compute_sources=True,
    return_n_iter=False,
    epsilon=0.1,
):
    """Performs Fast Independent Component Analysis of complex-valued signals.

    Parameters
    ----------
    X : array, shape (n_features,n_samples)
        Input signal X = A S, where A is the mixing
        matrix and S the latent sources.

    epsilon : float, optional
        Arbitrary constant in the contrast G function
        used in the approximation to neg-entropy.

    algorithm : {'parallel', 'deflation'}, optional
        Apply a parallel or deflational FASTICA algorithm.

    w_init : (n_components, n_components) array, optional
        Initial un-mixing array.If None (default) then an
        array of normally distributed r.v.s is used.

    tol: float, optional
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged.

    max_iter : int, optional
        Maximum number of iterations.

    whiten : boolean, optional
        If True, perform an initial whitening of the data.
        If False, the data is assumed to be already white.

    n_components : int, optional
        Number of components to extract. If None,
        n_components = n_features.

    Returns
    -------
    W : array, shape (n_components, n_components)
        Estimated un-mixing matrix.

    K : array, shape (n_components, n_features)
        If whiten is 'True', K is the pre-whitening matrix
        projecting the data onto the principal components.
        If whiten is 'False', K is 'None'.

    EG : array, shape(n_components, max_iter)
        Expectation of the contrast function E[G(|W'*X|^2)].
        This array may be padded with NaNs at the end.

    S : array, shape (n_samples, n_components)
        Estimated sources (S = W K X).
    """
    # random_state = check_random_state(random_state)
    fun_args = {} if fun_args is None else fun_args
    # make interface compatible with other decompositions
    # a copy is required only for non whitened data
    # X = check_array(X, copy=whiten, dtype=FLOAT_DTYPES).T

    # alpha = fun_args.get('alpha', 1.0)
    # if not 1 <= alpha <= 2:
    #     raise ValueError('alpha must be in [1,2]')

    if fun == "custom":
        g = _custom_contrast
    elif callable(fun):

        def g(x, w, fun_args):
            return fun(x, w, **fun_args)
    else:
        exc = ValueError if isinstance(fun, str) else TypeError
        raise exc("Unknown function %r; should be one of 'custom' or callable", fun)

    n, m = X.shape

    if w_init is None:
        w_init = np.random.normal(size=(n, n)) + 1j * np.random.normal(size=(n, n))

    if n_components is not None:
        n = n_components

    if whiten:
        X -= X.mean(1, keepdims=True)
        Ux, Sx = np.linalg.eig(np.cov(X))
        K = np.sqrt(np.linalg.inv(np.diag(Ux))).dot(Sx.conj().T)[:n]
        X1 = K.dot(X)
        del Ux, Sx
    else:
        K = None
        X1 = X.copy()

    kwargs = {
        "tol": tol,
        "g": g,
        "fun_args": fun_args,
        "max_iter": max_iter,
        "w_init": w_init,
    }
    if algorithm == "parallel":
        W, n_iter = _ica_par(X1, **kwargs)
    elif algorithm == "deflation":
        W, n_iter = _ica_def(X1, **kwargs)
    else:
        raise ValueError("Invalid algorithm")
    del X1

    # compute sources
    S = W.conj().T.dot(X)

    return K, W, S, n_iter
