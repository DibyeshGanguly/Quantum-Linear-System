import os
import numpy as np

def ET_stats(M, S=50_000, seed=1, rtol=1e-12):
    N = M.shape[0]
    M = 0.5 * (M + M.T)
    M = M / np.linalg.norm(M, 2)               
    lam, Q = np.linalg.eigh(M)
    a = np.abs(lam)
    keep = a > rtol * a.max()                   
    kappa = a[keep].max() / a[keep].min()
    inv2 = np.zeros_like(a)
    inv2[keep] = 1.0 / lam[keep] ** 2
    B = (Q * inv2) @ Q.T                        
    rn = np.sum(M * M, axis=1)                   

    rng = np.random.default_rng(seed)
    g = rng.standard_normal((S, N))
    Bhat = g @ M.T
    Bhat /= np.linalg.norm(Bhat, axis=1, keepdims=True)
    P = Bhat @ B                                
    D = rn[None, :] + Bhat ** 2
    ET = np.sum(P ** 2 * D, axis=1)

    if ET.max() > 2 * kappa ** 4 * 1.01:
        print("    [warn] ET exceeded 2 kappa^4 -- numerical trouble at this kappa")
    return ET.mean(), kappa, N


def cluster_D(E, mu, rng):
    d = np.concatenate([np.full(E // 2, mu), np.full(E - E // 2, 1.0 / mu)])
    rng.shuffle(d)
    return d


def M_diagonal(N, mu, rng):                     
    return np.diag(cluster_D(N, mu, rng))


def M_dense_random(N, mu, rng):                
    A = rng.standard_normal((N, N))
    return A @ np.diag(cluster_D(N, mu, rng)) @ A.T


def incidence(edges, N):
    B = np.zeros((N, len(edges)))
    for k, (i, j) in enumerate(edges):
        B[i, k], B[j, k] = 1.0, -1.0
    return B


def path_edges(N):
    return [(i, i + 1) for i in range(N - 1)]


def grid_edges(n):
    ix = lambda r, c: r * n + c
    E = []
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                E.append((ix(r, c), ix(r, c + 1)))
            if r + 1 < n:
                E.append((ix(r, c), ix(r + 1, c)))
    return E, n * n


def random_graph_edges(N, extra, rng):
    perm = rng.permutation(N)                    
    edges = [(int(perm[i]), int(perm[i + 1])) for i in range(N - 1)]
    for _ in range(extra):                       
        i, j = rng.choice(N, 2, replace=False)
        edges.append((int(i), int(j)))
    return edges


def M_laplacian(edges, N, mu, rng):
    B = incidence(edges, N)
    return B @ np.diag(cluster_D(B.shape[1], mu, rng)) @ B.T

def sweep(name, build, mus, S, seed=0):
    print(f"\n{name}")
    print(f"  {'mu':>7} {'kappa':>11} {'N':>4} {'E[ET]':>12} "
          f"{'E[ET]/kappa^2':>14} {'E[ET]/N':>10}")
    for mu in mus:
        rng = np.random.default_rng(seed)      
        M = build(mu, rng)
        eET, kappa, N = ET_stats(M, S=S, seed=seed + 1)
        print(f"  {mu:>7.0e} {kappa:>11.3g} {N:>4} {eET:>12.4g} "
              f"{eET / kappa ** 2:>14.4g} {eET / N:>10.4g}")


if __name__ == "__main__":
    S = int(os.environ.get("SAMPLES", "50000"))
    mus = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
    N = 20

    print("=" * 76)
    print(f"STRUCTURED NORMAL-EQUATIONS TEST   ({S} samples/point)")
    print("=" * 76)
    print("\n--- ANCHORS (calibrate the reading) ---")
    sweep("diagonal  (FAVORABLE anchor: expect E[ET]/N ~ const, /kappa^2 -> 0)",
          lambda mu, rng: M_diagonal(N, mu, rng), mus, S)
    sweep("dense random  (kappa^2 anchor: expect E[ET]/kappa^2 ~ const)",
          lambda mu, rng: M_dense_random(N, mu, rng), mus, S)

    print("\n--- STRUCTURED  M = B diag(d) B^T  (graph Laplacians) ---")
    sweep("path graph (tridiagonal; 1D discretization), N=20",
          lambda mu, rng: M_laplacian(path_edges(20), 20, mu, rng), mus, S)

    ge, gN = grid_edges(5)
    sweep(f"2D grid 5x5 (min-cost-flow structure), N={gN}",
          lambda mu, rng: M_laplacian(ge, gN, mu, rng), mus, S)

    sweep("random sparse connected graph, N=20 (~2N edges)",
          lambda mu, rng: M_laplacian(random_graph_edges(20, 20, rng), 20, mu, rng),
          mus, S)

    print("\nREAD:")
    print("  E[ET]/kappa^2 ~ const            -> generic kappa^2  (UNFAVORABLE)")
    print("  E[ET]/N ~ const & /kappa^2 -> 0  -> flat in kappa    (FAVORABLE)")
