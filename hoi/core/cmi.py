from functools import partial

import jax
import jax.numpy as jnp

from .entropies import prepare_for_entropy, get_entropy

###############################################################################
###############################################################################
#                                 SWITCHER
###############################################################################
###############################################################################


def get_cmi(method="gcmi", **kwargs):
    """Get the conditional mutual information function.

    Parameters
    ----------
    method : {'gcmi', 'binning', 'knn', 'kernel'}
        Name of the method to compute conditional mutual information.

    kwargs : dict | {}
        Additional arguments sent to the conditional mutual information function.

    Returns
    -------
    fcn : callable
        Function to compute mutual information on variables of shapes
        (n_features, n_samples)
    """
    _entropy = get_entropy(method=method, **kwargs)
    return partial(cmi_fcn, entropy_fcn=_entropy)


###############################################################################
###############################################################################
#                             PREPROCESSING
###############################################################################
###############################################################################


def prepare_for_cmi(x, y, z, method, **kwargs):
    """Prepare the data before computing conditional mutual information."""
    x, _ = prepare_for_entropy(x, method, **kwargs.copy())
    x, kwargs = prepare_for_entropy(_, method, **kwargs.copy())

    return x, y, z, kwargs


@partial(jax.jit, static_argnums=(2))
def compute_cmi_comb(inputs, comb, cmi=None):
    x, y, z = inputs
    x_c = x[:, comb, :]
    return inputs, cmi(x_c, y, z)


###############################################################################
###############################################################################
#                                 FUNCTIONS
###############################################################################
###############################################################################


@partial(jax.jit, static_argnums=(3,))
def cmi_fcn(x, y, z, entropy_fcn=None):
    """Compute the mutual information of two variables conditioned
    on a third by providing an entropy function.

    Parameters
    ----------
    x, y, z : array_like
        Arrays to consider for computing the Mutual Information. The two variables
        on which we compute the mutual information x and y should have a shape of
        (n_features_x, n_samples) and (n_features_y, n_samples).
        z is the conditioning variable and should have a shape of (n_features_z, n_samples).
    entropy_fcn : function | None
        Function to use for computing the entropy.

    Returns
    -------
    cmi : float
        Floating value describing the mutual information between x and y
        conditioned on z.
    """
    hz = entropy_fcn(z)
    hxz = entropy_fcn(jnp.concatenate((x, z), axis=0))
    hyz = entropy_fcn(jnp.concatenate((y, z), axis=0))
    hxyz = entropy_fcn(jnp.concatenate((x, y, z), axis=0))

    cmi = hxz + hyz - hxyz - hz
    return cmi
