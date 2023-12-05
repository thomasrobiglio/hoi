from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from hoi.core.entropies import prepare_for_entropy
from hoi.core.cmi import get_cmi, compute_cmi_comb
from hoi.utils.progressbar import get_pbar


class dOinfo(HOIEstimator):

    """dynamical O-information (dOinfo).

    Dynamical O-information is defined to generalize the O-information
    to the scope of handling multivariate time series which,
    apart from equal-time correlations, takes into account also
    lagged correlations.

    Consider a multivariate time series :math:`X^n = \{X_{1}, ..., X_{n}\}`
    with :math:`n` components. On this time series, we are able to compute
    the O-information :math: `\Omega(X^n)`. Upon adding a target variable
    :math:`Y`, the O-information becomes:

    .. math::

        \Omega(X^n, Y) = \Omega(X^n) + \Delta_n

    where:

    .. math::

        \Delta_n = (1-n)I(Y;X^n)+\sum_{j=1}^n I(Y;X^n_{-j})

    The additional term in :math: `\Delta_n` is the variation of the total
    O-information when the new variable :math: `Y` is added.

    In order to remove shared information due to common history and input signals
    the dynamical O-information is defined by conditioning the gradient presented
    above on the past history of the target variable :math: `Y`:

    .. math::

        d\Omega(Y;X^n)\equiv (1-n)\mathcal{I}(Y;X^n|Y_0)+\sum_{j=1}^n \mathcal{I}(Y;X^n\setminus X_j|Y_0)

    where :math: `Y_0(t)=\left(y(t),y(t-1),...,y(t-m+1)\right)` and
    :math: `Y=Y(t)=y(t+1)` are the samples of :math: `Y` divided into
    what we consider to be the past of the variable and the last instance.
    The parameter :math: `m` is the temporal order of the time series,
    corresponding to what we consider to be the relevant time scale of the process.

    .. warning::

        * :math:`d\Omega(Y;X^n) > 0 \Rightarrow Synergy`
        * :math:`d\Omega(Y;X^n) < 0 \Rightarrow Redundancy`

        Parameters
    ----------
    x : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    y : array_like
        The feature of shape (n_samples,)
    m : int
        The temporal order of the time series.
    multiplets : list | None
        List of multiplets to compute. Should be a list of multiplets, for
        example [(0, 1, 2), (2, 7, 8, 9)]. By default, all multiplets are
        going to be computed.

    References
    ----------
    Stramaglia et al. 2021 :cite:`stramaglia2021quantifying`
    """

    __name__ = "dynamical O-information"
    _encoding = True
    _positive = "synergy"
    _negative = "redundancy"
    _symmetric = True

    def __init__(self, x, y, m, multiplets=None, verbose=None):
        HOIEstimator.__init__(
            self, x=x, y=y, multiplets=multiplets, verbose=verbose
        )
        self._m = m
        self._z = y[:-m]

    def fit(self, minsize=2, maxsize=None, method="gcmi", **kwargs):
        """Compute dOinfo.

        Parameters
        ----------
        minsize, maxsize : int | 2, None
            Minimum and maximum size of the multiplets
        method : {'gcmi'}
            Name of the method to compute mutual-information. Use either :

                * 'gcmi': gaussian copula MI [default]. See
                  :func:`hoi.core.mi_gcmi_gg`

        kwargs : dict | {}
            Additional arguments are sent to each CMI function

        Returns
        -------
        hoi : array_like
            The NumPy array containing values of higher-rder interactions of
            shape (n_multiplets, n_variables)
        """
        # ________________________________ I/O ________________________________
        # check minsize and maxsize
        minsize, maxsize = self._check_minmax(max(minsize, 2), maxsize)

        # prepare the x for computing mi
        x, kwargs = prepare_for_entropy(self._x, method, **kwargs)
        x, y = self._split_xy(x)

        # prepare cmi functions
        cmi_fcn = jax.vmap(get_cmi(method=method, **kwargs))
        compute_cmi = partial(compute_cmi_comb, cmi=cmi_fcn)

        # get multiplet indices and order
        h_idx, order = self.get_combinations(minsize, maxsize=maxsize)

        # get progress bar
        pbar = get_pbar(
            iterable=range(order.min(), order.max() + 1), leave=False
        )

        # _______________________________ HOI _________________________________

        # compute mi I(x_{1}y y), ..., I(x_{n}; y)
        _, i_xiy = jax.lax.scan(
            compute_cmi, (x, y, self._z), jnp.arange(x.shape[1]).reshape(-1, 1)
        )

        offset = 0
        hoi = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)
        for msize in pbar:
            pbar.set_description(desc="dOinfo order %s" % msize, refresh=False)

            # combinations of features
            _h_idx = h_idx[order == msize, 0:msize]

            # compute I({x_{1}, ..., x_{n}}; S)
            _, _i_xy = jax.lax.scan(compute_cmi, (x, y, self._z), _h_idx)

            # compute hoi
            _hoi = _i_xy - i_xiy[_h_idx, :].sum(1)

            # fill variables
            n_combs = _h_idx.shape[0]
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)

            # updates
            offset += n_combs

        return np.asarray(hoi)
