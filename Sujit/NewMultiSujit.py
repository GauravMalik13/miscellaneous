import os
import numpy as np
import pandas as pd
from scipy.integrate import quad
import scipy.linalg as la
from multiprocess import Pool, cpu_count
import emcee
from tqdm import tqdm
from getdist import plots, MCSamples

# ——— Your data and model definitions ——————————————————————————————

df = pd.read_csv(r'E:\Pantheon+SH0ES.dat', delim_whitespace=True, comment='#')
flat_cov = np.loadtxt(r'E:\Pantheon+SH0ES_STAT+SYS.cov', skiprows=1)
cov_matrix = flat_cov.reshape((1701, 1701))
cov_tri = np.linalg.cholesky(cov_matrix)

mb = np.array(df["m_b_corr"])
z  = np.array(df["zHD"], dtype=np.float64)
c  = 299792.45

def E_CDM(z, Om):
    return np.sqrt(Om * (1 + z) ** 3 + (1 - Om))

def chi_square(H0, Om, M):
    residuals = []
    for zi, mbi in zip(z, mb):
        integral, _ = quad(lambda zp: 1.0 / E_CDM(zp, Om), 0, zi)
        d_L = c * (1 + zi) / H0 * integral
        mu  = 5 * np.log10(d_L) + 25
        residuals.append(mbi - M - mu)
    res = la.solve_triangular(cov_tri, np.array(residuals), lower=True, check_finite=False)
    return np.sum(res ** 2)

def log_prior(H0, Om, M):
    return 0.0 if (50 < H0 < 100 and 0.01 < Om < 1 and -21 < M < -17) else -np.inf

def log_likelihood(H0, Om, M):
    return -0.5 * chi_square(H0, Om, M) - 0.5 * 1701 * np.log(2 * np.pi)

def log_posterior(theta):
    H0, Om, M = theta
    lp = log_prior(H0, Om, M)
    return lp + log_likelihood(H0, Om, M) if np.isfinite(lp) else -np.inf

# ——— Main routine ————————————————————————————————————————————————

def main():
    ndim, nwalkers = 3, 50
    nburn, nsteps = 2000, 10000

    np.random.seed(0)
    p0 = np.array([70, 0.1, -18]) + 1e-2 * np.random.randn(nwalkers, ndim)

    # limit OpenMP threads per worker
    os.environ["OMP_NUM_THREADS"] = str(min(32, cpu_count()))
    print("OMP_NUM_THREADS =", os.environ["OMP_NUM_THREADS"])

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
        state   = p0
        for sample in tqdm(
            sampler.sample(state, iterations=nsteps),
            total=nsteps,
            desc="MCMC sampling",
        ):
            state = sample[0]

    flat_samples = sampler.get_chain(discard=nburn, thin=15, flat=True)
    print("Done: post‑burn samples =", flat_samples.shape[0])

    names, labels = ['H0','Om','M'], ['H_0', r'\Omega_m', 'M']
    g_samples = MCSamples(samples=flat_samples, names=names, labels=labels)
    g = plots.getSubplotPlotter()
    g.triangle_plot(g_samples, filled=True)
    import matplotlib.pyplot as plt
    plt.show()

if __name__ == "__main__":
    main()
