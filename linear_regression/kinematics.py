
import torch as tc

def pos_eq(p0, v0, a, t, noise=False):
    p = p0 + v0*t + 0.5*a*t**2

    if noise:
        p += tc.rand(t.shape)
    return p

def limit_vals_to_ground(t, x, y):
    idcs = tc.where(y[y > 0])[0]
    t, x, y = t[idcs], x[idcs], y[idcs]

    assert t.shape == x.shape == y.shape, "Check shape"
    return t, x, y