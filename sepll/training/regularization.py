from optax._src import linear_algebra


def l2_lf_matrix(params):
    x = params["lf_classifier"]
    return linear_algebra.global_norm(x)

