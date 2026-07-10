import numpy as np


def ET(A, bhat):
    p = np.linalg.pinv(A @ A.T) @ bhat
    d = np.sum(A * A, axis=1) + bhat**2
    return float(np.sum(p**2 * d))


def ET_batch_diag(a, Bhat):
    p = Bhat / (a**2)[None, :]
    d = (a**2)[None, :] + Bhat**2
    return np.sum(p**2 * d, axis=1)


def unit_sphere(S, M, rng):
    g = rng.standard_normal((S, M))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def exp_graded(rho=0.5, S=300_000, seed=0):
    print("EXP 1  graded family  A = diag(rho^i), rho = 1/2,  %.0e samples" % S)
    print(f"{'N':>3} {'kappa':>10} {'E[ET] iso':>11} {'+-se':>7} "
          f"{'E[ET] unif':>12} {'kappa^4/N^2':>12}")
    rows = []
    for N in [6, 10, 14, 18]:
        rng = np.random.default_rng(seed + N)
        a = rho ** np.arange(N)
        kappa = rho ** (-(N - 1))
        g = rng.standard_normal((S, N))
        Ag = g * a[None, :]
        iso = ET_batch_diag(a, Ag / np.linalg.norm(Ag, axis=1, keepdims=True))
        unif = ET_batch_diag(a, unit_sphere(S, N, rng))
        row = (N, kappa, iso.mean(), iso.std() / np.sqrt(S),
               unif.mean(), kappa**4 / N**2)
        rows.append(row)
        print(f"{N:>3} {kappa:>10.3g} {row[2]:>11.1f} {row[3]:>7.1f} "
              f"{row[4]:>12.3g} {row[5]:>12.3g}")
    return rows

def exp_newton(N=20, S=40_000, seed=1):
    print("\nEXP 2  Newton-structured  M = Abar diag(d) Abar^T,  N=%d" % N)
    print(f"{'mu':>8} {'kappa(M)':>11} {'sqrt(E[ET])':>12} {'E[sqrt(ET)]':>12}")
    rows = []
    for mu in [1e-1, 1e-2, 1e-3]:
        rng = np.random.default_rng(seed)
        Abar = rng.standard_normal((N, N))
        d = np.concatenate([np.full(N // 2, mu), np.full(N - N // 2, 1 / mu)])
        M = Abar @ np.diag(d) @ Abar.T
        M = 0.5 * (M + M.T)
        M = M / np.linalg.norm(M, 2)
        kappa = np.linalg.cond(M)
        g = rng.standard_normal((S, N))
        Bhat = (g @ M.T)
        Bhat /= np.linalg.norm(Bhat, axis=1, keepdims=True)
        vals = np.array([ET(M, Bhat[i]) for i in range(S)])
        rows.append((mu, kappa, np.sqrt(vals.mean()), np.sqrt(vals).mean()))
        print(f"{mu:>8.0e} {kappa:>11.3g} {np.sqrt(vals.mean()):>12.1f} "
              f"{np.sqrt(vals).mean():>12.1f}")
    return rows


def exp_dense(N=10, S=40_000, seed=2):
    print("\nEXP 3  dense  A = U Sigma V^T, geometric spectrum,  N=%d" % N)
    print(f"{'kappa':>8} {'E[ET] iso':>11} {'E[ET]/kappa^2':>14} "
          f"{'E[ET] unif':>12} {'unif/kappa^4':>13}")
    rows = []
    for kappa in [64, 256, 1024, 4096]:
        rng = np.random.default_rng(seed)
        U, _ = np.linalg.qr(rng.standard_normal((N, N)))
        V, _ = np.linalg.qr(rng.standard_normal((N, N)))
        A = U @ np.diag(kappa ** (-np.linspace(0, 1, N))) @ V.T
        g = rng.standard_normal((S, N))
        Bi = g @ A.T
        Bi /= np.linalg.norm(Bi, axis=1, keepdims=True)
        iso = np.array([ET(A, Bi[i]) for i in range(S)])
        Bu = unit_sphere(S, N, rng)
        unif = np.array([ET(A, Bu[i]) for i in range(S)])
        rows.append((kappa, iso.mean(), unif.mean()))
        print(f"{kappa:>8} {iso.mean():>11.3g} {iso.mean()/kappa**2:>14.3f} "
              f"{unif.mean():>12.3g} {unif.mean()/kappa**4:>13.3g}")
    return rows


def exp_rotation(N=10, rho=0.5, S=20_000, seed=0):
    from scipy.linalg import expm
    print("\nEXP 4  rotate graded family off diagonal  (N=%d, kappa=%d)"
          % (N, round(rho ** (-(N - 1)))))
    print(f"{'theta':>8} {'E[ET] iso':>11} {'/kappa^2':>10}")
    kap = rho ** (-(N - 1))
    for theta in [0.0, 0.01, 0.05, 0.2, 0.5, 1.0, 1.5]:
        rng = np.random.default_rng(seed)
        G = rng.standard_normal((N, N)); G = G - G.T
        R = expm(theta * G / np.linalg.norm(G, 2))
        A = R @ np.diag(rho ** np.arange(N))
        g = rng.standard_normal((S, N))
        Bi = g @ A.T; Bi /= np.linalg.norm(Bi, axis=1, keepdims=True)
        m = np.mean([ET(A, Bi[i]) for i in range(S)])
        print(f"{theta:>8.2f} {m:>11.4g} {m/kap**2:>10.4f}")


if __name__ == "__main__":
    exp_graded()
    exp_newton()
    exp_dense()
    try:
        exp_rotation()
    except ImportError:
        print("\n(EXP 4 skipped: scipy not installed)")
