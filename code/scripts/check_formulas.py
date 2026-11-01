import numpy as np
rng = np.random.default_rng(1)

# Proposition 1(b): cross-sectional moments of the RPNB model
def rnb(mu, r, size=None):
    lam = rng.gamma(r, mu / r)          # gamma mixing with mean mu
    return rng.poisson(lam)

x = np.array([1.0, 1.7, 1.0])            # covariates (x3 binary = 1)
beta = np.array([0.3, -0.2, 0.4]); sig = np.array([0.5, 0.3, 0.6]); r = 2.0; alpha = -0.5
M = 2_000_000
z = rng.standard_normal((M, 3))
eta = alpha + x @ beta
lam = np.exp(eta + (z * sig * x).sum(1))
y = rnb(lam, r)
v = (sig**2 * x**2).sum()
mu_th = np.exp(eta + v / 2)
var_th = mu_th + mu_th**2 * ((1 + 1/r) * np.exp(v) - 1)
print("Prop1 mean  MC %.4f  theory %.4f" % (y.mean(), mu_th))
print("Prop1 var   MC %.3f  theory %.3f" % (y.var(), var_th))

# Proposition 2: within-site covariance, independent of r
xt = np.array([1.0, 0.4, 1.0]); xs = np.array([1.0, 2.0, 0.0])
for r2 in (0.7, 5.0):
    zz = rng.standard_normal((M, 3))
    lt = np.exp(alpha + xt @ beta + (zz * sig * xt).sum(1))
    ls = np.exp(alpha + xs @ beta + (zz * sig * xs).sum(1))
    yt, ys = rnb(lt, r2), rnb(ls, r2)
    D = np.diag(sig**2)
    mut = np.exp(alpha + xt @ beta + xt @ D @ xt / 2); mus = np.exp(alpha + xs @ beta + xs @ D @ xs / 2)
    cov_th = mut * mus * (np.exp(xt @ D @ xs) - 1)
    print("Prop2 r=%.1f cov MC %.4f  theory %.4f" % (r2, np.cov(yt, ys)[0, 1], cov_th))

# Proposition 3: collapsed marginal evidence, checked against brute-force integration over z
T = 4
w = rng.uniform(0.2, 2.0, T); xj = rng.normal(1, 1, T); e = rng.normal(0, 1, T); s = 0.8
q = (w * xj**2).sum(); b = (w * xj * e).sum()
closed = -0.5 * np.log(1 + s**2 * q) + s**2 * b**2 / (2 * (1 + s**2 * q))
zg = np.linspace(-12, 12, 200001); dz = zg[1] - zg[0]
logl = lambda zz: -0.5 * (w[:, None] * (e[:, None] - s * xj[:, None] * zz[None, :])**2).sum(0)
num = np.log((np.exp(logl(zg) - logl(np.zeros(1))[0]) * np.exp(-zg**2 / 2) / np.sqrt(2 * np.pi)).sum() * dz)
print("Prop3 ell_j  closed %.6f  numeric %.6f" % (closed, num))

# Fixed-effect Bayes factor (eq. bf-beta)
tau = 0.7; S = (w * xj**2).sum(); B = (w * xj * e).sum()
closed_b = -0.5 * np.log(1 + tau**2 * S) + B**2 / (2 * (S + tau**-2))
bg = np.linspace(-15, 15, 300001); db = bg[1] - bg[0]
num_b = np.log((np.exp(B * bg - 0.5 * S * bg**2) * np.exp(-bg**2 / (2 * tau**2)) / np.sqrt(2 * np.pi * tau**2)).sum() * db)
print("BF_beta      closed %.6f  numeric %.6f" % (closed_b, num_b))
