import numpy as np


# DIMENSIONS
#-------------------------------------------------------------

def n_states(params):
    # Number of states |X|.

    return len(params["states"]) 


def n_actions(params):
    # Number of actions |A|.
    return len(params["actions"])


# GENERIC HELPERS
#-------------------------------------------------------------

def normalize_probability(mu):
    # Project a numerical vector to a probability vector.
    # mu(x) >= 0 and sum_x mu(x) = 1.

    mu = np.asarray(mu, dtype=float).copy()
    mu = np.maximum(mu, 0.0)
    s = np.sum(mu)
    if s <= 0:
        raise ValueError("Probability vector must have positive sum.")
    return mu / s


def stationary_distribution(P, params):
    # Compute invariant distribution mu of P.
    # mu P = mu,  sum_x mu(x) = 1.

    n = n_states(params)
    A = P.T - np.eye(n)
    A[-1, :] = np.ones(n)
    b = np.zeros(n)
    b[-1] = 1.0
    mu = np.linalg.solve(A, b)
    return normalize_probability(mu)


def occupation_measure_from_mu_pi(mu, pi):
    # Stationary occupation measure.
    # nu(x,a) = mu(x) π(a|x).

    return mu[:, None] * pi


def logsumexp(v):
    # Stable computation of log(sum_i exp(v_i)).

    m = np.max(v)
    return m + np.log(np.sum(np.exp(v - m)))


def softmax(v):
    # Softmax: softmax(v)_a = exp(v_a) / sum_b exp(v_b).
    
    m = np.max(v)
    w = np.exp(v - m)
    return w / np.sum(w)


# MODEL HELPERS
#-------------------------------------------------------------

def deterministic_next_state_index(x, a, params):
    # Deterministic post-action state before adding transition noise.
    # Action 0 keeps the provider; action 1 switches the provider.

    i, j = params["states"][x]

    if a == 0:   # stay
        nxt = (i, j)
    elif a == 1: # change
        nxt = (2 if i == 1 else 1, j)
    else:
        raise ValueError("Invalid action index.")

    return params["states"].index(nxt)


def tau_provider(x, a, params):
    # Provider component after applying action a in state x.

    y = deterministic_next_state_index(x, a, params)
    provider_after_action, _ = params["states"][y]
    return provider_after_action


def provider_shares(mu):
    # Provider shares induced by mu.

    m1 = mu[0] + mu[1]
    m2 = mu[2] + mu[3]
    return m1, m2


def switch_indicator(a):
    # Indicator of changing provider.

    return 1.0 if a == 1 else 0.0


def mismatch_indicator(x, a, params):
    # Indicator that the post-action provider differs from the preferred provider.

    _, preferred_provider = params["states"][x]
    provider_after_action = tau_provider(x, a, params)
    return 1.0 if provider_after_action != preferred_provider else 0.0


# TRUE TRANSITION AND TRUE REWARD
#-------------------------------------------------------------

def transition_fn(x, a, mu, params):
    # Noisy transition.
    # p(y | x,a,mu) = (1-epsilon) 1{y = T(x,a)} + epsilon * nu_noise[y]

    p = params["epsilon"] * params["nu_noise"].copy()
    y_det = deterministic_next_state_index(x, a, params)
    p[y_det] += (1.0 - params["epsilon"])
    return p


def true_reward_fn(x, a, mu, params, clip=1e-12):
    # True reward used to generate the expert MFE.

    m1, m2 = provider_shares(mu)
    provider_after_action = tau_provider(x, a, params)
    m_tau = m1 if provider_after_action == 1 else m2
    m_tau = max(m_tau, clip)

    provider_bias = params.get("provider1_bias", 0.0)
    provider1_bonus = provider_bias * (1.0 if provider_after_action == 1 else 0.0)

    return (
        params["alpha_share"] * np.log(m_tau + params["delta_share"])
        - params["congestion_coeff"] * (m_tau ** 2)
        + provider1_bonus
        - params["lambda_switch"] * switch_indicator(a)
        - params["rho_mismatch"] * mismatch_indicator(x, a, params)
    )


def true_reward_matrix(mu, params):
    # Matrix form of the true reward R(x,a) for fixed mu.

    R = np.zeros((n_states(params), n_actions(params)))
    for x in range(n_states(params)):
        for a in range(n_actions(params)):
            R[x, a] = true_reward_fn(x, a, mu, params)
    return R


# MINORIZATION 
#-------------------------------------------------------------

def common_lower_bound(mu, params):
    # Compute the common minorization measure xi.
    # xi(y) <= p(y|x,a,mu) for every (x,a).

    xi = np.ones(n_states(params))
    for x in range(n_states(params)):
        for a in range(n_actions(params)):
            xi = np.minimum(xi, transition_fn(x, a, mu, params))
    return xi


def reduced_transition(x, a, mu, xi, params):
    # Reduced kernel.
    # p_tilde(.|x,a,mu) = p(.|x,a,mu) - xi(.).

    return transition_fn(x, a, mu, params) - xi


# FORWARD CLASSICAL MFE SOLVER
#-------------------------------------------------------------

def induced_transition_matrix(pi, mu, params):
    # Transition matrix induced by π.
    # P_π(x,y) = sum_a π(a|x) p(y|x,a,mu).

    P_pi = np.zeros((n_states(params), n_states(params)))
    for x in range(n_states(params)):
        for a in range(n_actions(params)):
            P_pi[x, :] += pi[x, a] * transition_fn(x, a, mu, params)
    return P_pi


def greedy_policy_from_Q(Q, tol=1e-12):
    # Classical greedy policy.
    # π(a|x)>0 only for actions maximizing Q(x,a).
    # Ties are handled uniformly.

    pi = np.zeros_like(Q, dtype=float)

    for x in range(Q.shape[0]):
        q_max = np.max(Q[x, :])
        greedy_actions = np.flatnonzero(np.abs(Q[x, :] - q_max) <= tol)

        if len(greedy_actions) == 0:
            greedy_actions = np.array([int(np.argmax(Q[x, :]))])

        pi[x, greedy_actions] = 1.0 / len(greedy_actions)

    return pi


def expert_greedy_action_mass(pi_E, pi_candidate, mu=None, tol=1e-12):
    # Weighted mass that π_candidate assigns to expert greedy actions.
    # Useful when π_E is deterministic but π_candidate is soft.

    n_x = pi_E.shape[0]

    if mu is None:
        weights = np.ones(n_x) / n_x
    else:
        weights = normalize_probability(mu)

    mass = 0.0

    for x in range(n_x):
        expert_actions = np.flatnonzero(pi_E[x, :] > tol)

        if len(expert_actions) == 0:
            expert_actions = np.array([int(np.argmax(pi_E[x, :]))])

        mass += weights[x] * np.sum(pi_candidate[x, expert_actions])

    return mass


def solve_frozen_true_mdp(mu, params):
    # Solve the classical average-reward control problem for fixed mu.
    # Gives the expert best response for the current mean-field term.

    xi = common_lower_bound(mu, params)
    V = np.zeros(n_states(params))

    for it in range(params["max_inner_iter_true"]):
        Q = np.zeros((n_states(params), n_actions(params)))

        for x in range(n_states(params)):
            for a in range(n_actions(params)):
                ptil = reduced_transition(x, a, mu, xi, params)
                Q[x, a] = true_reward_fn(x, a, mu, params) + np.dot(ptil, V)

        # Hard Bellman operator: max_a Q(x,a).
        V_raw = np.max(Q, axis=1)

        # Normalization removes the additive constant ambiguity.
        V_new = V_raw - V_raw[params["x_ref_true"]]

        diff = np.max(np.abs(V_new - V))
        V = V_new

        if diff < params["tol_true_mdp"]:
            break

    Q = np.zeros((n_states(params), n_actions(params)))
    for x in range(n_states(params)):
        for a in range(n_actions(params)):
            ptil = reduced_transition(x, a, mu, xi, params)
            Q[x, a] = true_reward_fn(x, a, mu, params) + np.dot(ptil, V)

    V_raw = np.max(Q, axis=1)

    pi = greedy_policy_from_Q(
        Q,
        tol=params.get("greedy_tol", 1e-12)
    )

    # Average reward rho and Bellman residual.
    # rho + V(x) = max_a Q(x,a).
    rho = V_raw[params["x_ref_true"]]
    hard_bellman_residual = np.max(np.abs((rho + V) - V_raw))

    return pi, rho, V, hard_bellman_residual, Q


def solve_true_mfe(mu_start, params, verbose=False):
    # Fixed-point iteration for the classical expert MFE.
    # Solve best response for mu, then update mu to the invariant distribution.

    mu = normalize_probability(mu_start)

    history = []
    mu_history = [mu.copy()]
    pi_history = []

    for outer_it in range(params["max_outer_iter_true"]):
        pi, rho, h, bellman_residual, Q = solve_frozen_true_mdp(mu, params)
        pi_history.append(pi.copy())

        P_pi = induced_transition_matrix(pi, mu, params)
        mu_hat = stationary_distribution(P_pi, params)

        mf_residual = np.linalg.norm(mu_hat - mu, ord=1)
        history.append([outer_it, mf_residual, bellman_residual])

        if verbose:
            print(
                "forward outer iter =", outer_it,
                "| mf residual =", mf_residual,
                "| hard Bellman residual =", bellman_residual
            )

        if mf_residual < params["tol_true_mf"] and bellman_residual < 1e-10:
            if verbose:
                print("Forward classical MFE solve converged.")
            return mu_hat, pi, rho, h, Q, history, mu_history, pi_history

        mu_new = (1.0 - params["damping_true_mf"]) * mu + params["damping_true_mf"] * mu_hat
        mu_new = normalize_probability(mu_new)

        mu = mu_new
        mu_history.append(mu.copy())

    if verbose:
        print("Forward classical MFE solve hit max_outer_iter_true.")

    return mu, pi, rho, h, Q, history, mu_history, pi_history


# GAUSSIAN RKHS FEATURE MAP
#-------------------------------------------------------------

def encode_z(x, a, mu, params):
    # Encode z=(x,a,mu) as a finite vector for the Gaussian kernel.
    
    i, j = params["states"][x]
    m1, m2 = provider_shares(mu)

    return np.array([
        1.0 if i == 1 else 0.0,
        1.0 if i == 2 else 0.0,
        1.0 if j == 1 else 0.0,
        1.0 if j == 2 else 0.0,
        1.0 if a == 0 else 0.0,
        1.0 if a == 1 else 0.0,
        m1,
        m2,
    ], dtype=float)


def kernel(z1, z2, params):
    # Gaussian kernel.
    # k(z,z') = exp(-||eta(z)-eta(z')||^2 / (2 sigma^2)).

    x1, a1, mu1 = z1
    x2, a2, mu2 = z2

    v1 = encode_z(x1, a1, mu1, params)
    v2 = encode_z(x2, a2, mu2, params)

    diff = v1 - v2
    sigma = params["sigma_kernel"]

    return np.exp(- np.dot(diff, diff) / (2.0 * sigma ** 2))


def build_basis_points(mu_E, params):
    # RKHS anchors z_i = (x,a,mu_E) for all state-action pairs.

    basis_points = []
    for x in range(n_states(params)):
        for a in range(n_actions(params)):
            basis_points.append((x, a, mu_E.copy()))
    return basis_points


def gram_matrix(basis_points, params, ridge=1e-8):
    # Gram matrix G_ij = k(z_i,z_j).
    # A small ridge is added for numerical stability.

    m = len(basis_points)
    G = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            G[i, j] = kernel(basis_points[i], basis_points[j], params)
    if ridge > 0.0:
        G = G + ridge * np.eye(m)
    return G


def rkhs_features(x, a, mu, basis_points, params):
    # Kernel feature vector.
    # [k((x,a,mu),z_1), ..., k((x,a,mu),z_m)].

    z = (x, a, mu.copy())
    return np.array([kernel(z, z_i, params) for z_i in basis_points], dtype=float)


def full_features(x, a, mu, basis_points, params):
    # Full feature vector used in the RKHS gradient.
    # [e_x, kernel features].

    e_x = np.zeros(n_states(params))
    e_x[x] = 1.0
    return np.concatenate([e_x, rkhs_features(x, a, mu, basis_points, params)])


# EXPERT MOMENTS
#-------------------------------------------------------------

def rkhs_moment_from_nu(nu, mu, basis_points, params):
    # RKHS feature moment.
    # sum_{x,a} nu(x,a) phi_k(x,a,mu).

    val = np.zeros(len(basis_points))
    for x in range(n_states(params)):
        for a in range(n_actions(params)):
            val += nu[x, a] * rkhs_features(x, a, mu, basis_points, params)
    return val


# RKHS PARAMETERIZATION
#-------------------------------------------------------------

def unpack_theta(theta, params, basis_points):
    # Split theta into zeta and kernel coefficients c.

    n_x = n_states(params)
    zeta = theta[:n_x]
    c = theta[n_x:]
    return zeta, c


def pack_theta(zeta, c):
    # Combine zeta and c into one parameter vector theta.

    return np.concatenate([zeta, c])


def gram_corrected_update(theta, grad, gamma, params, basis_points, G_inv):
    # Gradient ascent update.
    # Kernel coefficients are updated using G^{-1} to respect the RKHS geometry.

    n_x = n_states(params)

    zeta = theta[:n_x]
    c = theta[n_x:]

    g_zeta = grad[:n_x]
    g_c = grad[n_x:]

    zeta_new = zeta + gamma * g_zeta
    c_new = c + gamma * (G_inv @ g_c)

    return pack_theta(zeta_new, c_new)


def reward_from_theta(x, a, mu, theta, basis_points, params):
    # Learned reward.
    # r_theta(x,a,mu) = zeta_x + sum_i c_i k((x,a,mu), z_i).

    zeta, c = unpack_theta(theta, params, basis_points)
    return zeta[x] + c @ rkhs_features(x, a, mu, basis_points, params)


def reward_matrix_from_theta(theta, mu, basis_points, params):
    # Matrix form of learned reward R_theta(x,a).

    R = np.zeros((n_states(params), n_actions(params)))
    for x in range(n_states(params)):
        for a in range(n_actions(params)):
            R[x, a] = reward_from_theta(x, a, mu, theta, basis_points, params)
    return R


# THEORY BOUNDS
#-------------------------------------------------------------

def compute_theory_bounds(mu_E, basis_points, xi, params):
    # Compute constants in the RKHS smoothness bound.
    
    kappa = 1.0 - np.sum(xi)

    K_bound = 0.0
    for x in range(n_states(params)):
        for a in range(n_actions(params)):
            K_bound = max(
                K_bound,
                np.linalg.norm(full_features(x, a, mu_E, basis_points, params))
            )

    if kappa < 1.0:
        L_bound = n_actions(params) * (K_bound ** 2) * (kappa + 1.0) / ((1.0 - kappa) ** 3)
    else:
        L_bound = np.inf

    return K_bound, L_bound, kappa


# SOFT BELLMAN SOLVER UNDER FROZEN mu_E
#-------------------------------------------------------------
def solve_policy_from_theta(theta, mu_E, xi, basis_points, params):
    # Compute the softmax policy π_theta for fixed mu_E.
    # Uses the reduced kernel p_tilde and the learned reward r_theta.

    V = np.zeros(n_states(params))

    for it in range(params["rkhs_max_value_iter"]):
        Q = np.zeros((n_states(params), n_actions(params)))

        for x in range(n_states(params)):
            for a in range(n_actions(params)):
                ptil = reduced_transition(x, a, mu_E, xi, params)
                Q[x, a] = reward_from_theta(x, a, mu_E, theta, basis_points, params) + np.dot(ptil, V)

        # Soft Bellman value: V(x) = log sum_a exp(Q(x,a)).
        V_raw = np.zeros(n_states(params))
        for x in range(n_states(params)):
            V_raw[x] = logsumexp(Q[x, :])

        # Normalize value function to remove additive constant.
        V_new = V_raw - V_raw[params["x_ref_rkhs"]]

        diff = np.max(np.abs(V_new - V))
        V = V_new

        if diff < params["rkhs_tol_value"]:
            break

    Q = np.zeros((n_states(params), n_actions(params)))
    for x in range(n_states(params)):
        for a in range(n_actions(params)):
            ptil = reduced_transition(x, a, mu_E, xi, params)
            Q[x, a] = reward_from_theta(x, a, mu_E, theta, basis_points, params) + np.dot(ptil, V)

    V_raw = np.zeros(n_states(params))
    pi = np.zeros((n_states(params), n_actions(params)))

    for x in range(n_states(params)):
        V_raw[x] = logsumexp(Q[x, :])
        pi[x, :] = softmax(Q[x, :])

    # Compute stationary distribution and occupation measure induced by π_theta.
    P_pi = np.zeros((n_states(params), n_states(params)))
    for x in range(n_states(params)):
        for a in range(n_actions(params)):
            P_pi[x, :] += pi[x, a] * transition_fn(x, a, mu_E, params)

    mu_pi = stationary_distribution(P_pi, params)
    nu_pi = occupation_measure_from_mu_pi(mu_pi, pi)

    return Q, V_raw, pi, P_pi, mu_pi, nu_pi


# SCORE AND GRADIENT
#-------------------------------------------------------------

def score_and_gradient(theta, mu_E, nu_E, xi, basis_points, f_exp_E, params):
    # Score: V(theta) = sum_{x,a} nu_E(x,a) log π_theta(a|x).
    # Gradient is expert moments minus recovered moments.

    Q, V_raw, pi, P_pi, mu_pi, nu_pi = solve_policy_from_theta(
        theta, mu_E, xi, basis_points, params
    )

    eps = 1e-15
    score = np.sum(nu_E * np.log(pi + eps))

    phi_exp_pi = rkhs_moment_from_nu(nu_pi, mu_E, basis_points, params)
    f_exp_pi = np.concatenate([mu_pi.copy(), phi_exp_pi])

    grad = f_exp_E - f_exp_pi

    return score, grad, pi, mu_pi, nu_pi, Q, V_raw, f_exp_pi


# FULL PIPELINE
#-------------------------------------------------------------

def run_pipeline(params, mu_start=None, verbose=False):
    # Full experiment:
    # 1. compute expert MFE,
    # 2. build RKHS features,
    # 3. run gradient ascent on the score,
    # 4. return recovered policy and diagnostics.

    params = params.copy()

    params["states"] = [tuple(s) for s in params["states"]]
    params["state_names"] = list(params["state_names"])
    params["actions"] = list(params["actions"])

    if "nu_noise" not in params or params["nu_noise"] is None:
        params["nu_noise"] = np.ones(n_states(params), dtype=float) / n_states(params)
    else:
        params["nu_noise"] = normalize_probability(np.asarray(params["nu_noise"], dtype=float))

    if mu_start is None:
        mu_start = np.ones(n_states(params), dtype=float) / n_states(params)
    else:
        mu_start = normalize_probability(np.asarray(mu_start, dtype=float))

    # Forward MFE solve
    mu_E, pi_E, rho_E, h_E, Q_E, history_true_mfe, mu_history_true, pi_history_true = solve_true_mfe(
        mu_start, params, verbose=verbose
    )

    P_pi_E = induced_transition_matrix(pi_E, mu_E, params)
    nu_E = occupation_measure_from_mu_pi(mu_E, pi_E)
    R_true_E = true_reward_matrix(mu_E, params)

    # RKHS setup
    basis_points = build_basis_points(mu_E, params)
    G = gram_matrix(basis_points, params, ridge=1e-8)
    G_inv = np.linalg.inv(G)

    n_basis = len(basis_points)
    theta_dim = n_states(params) + n_basis

    # Expert moments used in the RKHS score gradient.
    phi_exp_E = rkhs_moment_from_nu(nu_E, mu_E, basis_points, params)
    f_exp_E = np.concatenate([mu_E.copy(), phi_exp_E])

    # Minorization and theoretical constants.
    xi = common_lower_bound(mu_E, params)
    K_bound, L_bound, kappa = compute_theory_bounds(mu_E, basis_points, xi, params)

    # Gradient ascent
    theta = np.zeros(theta_dim)

    history_score_rkhs = []
    history_grad_norm_rkhs = []
    history_mu_error_rkhs = []
    history_phi_error_rkhs = []
    history_monotonicity_rkhs = []

    history_pi_entries_rkhs = {}
    for x in range(n_states(params)):
        for a in range(n_actions(params)):
            history_pi_entries_rkhs[(x, a)] = []

    for k in range(params["rkhs_n_iter"]):
        score, grad, pi_theta, mu_theta, nu_theta, Q_theta, V_theta, f_exp_theta = score_and_gradient(
            theta, mu_E, nu_E, xi, basis_points, f_exp_E, params
        )

        grad_norm = np.linalg.norm(grad)
        phi_exp_theta = f_exp_theta[n_states(params):]

        mu_error = np.linalg.norm(mu_theta - mu_E, ord=1)
        phi_error = np.linalg.norm(phi_exp_theta - phi_exp_E)

        history_score_rkhs.append(score)
        history_grad_norm_rkhs.append(grad_norm)
        history_mu_error_rkhs.append(mu_error)
        history_phi_error_rkhs.append(phi_error)

        # Store all policy entries for convergence plots.
        for x in range(n_states(params)):
            for a in range(n_actions(params)):
                history_pi_entries_rkhs[(x, a)].append(pi_theta[x, a])

        if k > 0:
            history_monotonicity_rkhs.append(history_score_rkhs[-1] - history_score_rkhs[-2])

        # RKHS-gradient corrected ascent step.
        theta = gram_corrected_update(
            theta=theta,
            grad=grad,
            gamma=params["gamma"],
            params=params,
            basis_points=basis_points,
            G_inv=G_inv,
        )

        if verbose and k % 500 == 0:
            print(
                "iter =", k,
                "| score =", score,
                "| grad norm =", grad_norm,
                "| mu error =", mu_error,
                "| phi error =", phi_error
            )

    # Final recovered objects
    score, grad, pi_theta, mu_theta, nu_theta, Q_theta, V_theta, f_exp_theta = score_and_gradient(
        theta, mu_E, nu_E, xi, basis_points, f_exp_E, params
    )

    phi_exp_theta = f_exp_theta[n_states(params):]
    mu_from_nu = np.sum(nu_theta, axis=1)

    R_theta = reward_matrix_from_theta(theta, mu_E, basis_points, params)
    zeta_final, c_final = unpack_theta(theta, params, basis_points)

    pi_error_fro = np.linalg.norm(pi_theta - pi_E)

    expert_action_mass_unweighted = expert_greedy_action_mass(
        pi_E,
        pi_theta,
        mu=None,
        tol=params.get("greedy_tol", 1e-12)
    )

    expert_action_mass_weighted = expert_greedy_action_mass(
        pi_E,
        pi_theta,
        mu=mu_E,
        tol=params.get("greedy_tol", 1e-12)
    )

    results = {
        "params": params,
        "basis_points": basis_points,
        "G": G,
        "G_inv": G_inv,
        "n_basis": n_basis,
        "theta_dim": theta_dim,
        "xi": xi,
        "kappa": kappa,
        "K_bound": K_bound,
        "L_bound": L_bound,
        "mu_E": mu_E,
        "pi_E": pi_E,
        "rho_E": rho_E,
        "h_E": h_E,
        "Q_E": Q_E,
        "P_pi_E": P_pi_E,
        "nu_E": nu_E,
        "R_true_E": R_true_E,
        "phi_exp_E": phi_exp_E,
        "f_exp_E": f_exp_E,
        "theta": theta,
        "score": score,
        "grad": grad,
        "grad_norm": np.linalg.norm(grad),
        "pi_theta": pi_theta,
        "mu_theta": mu_theta,
        "nu_theta": nu_theta,
        "Q_theta": Q_theta,
        "V_theta": V_theta,
        "f_exp_theta": f_exp_theta,
        "phi_exp_theta": phi_exp_theta,
        "mu_from_nu": mu_from_nu,
        "R_theta": R_theta,
        "zeta_final": zeta_final,
        "c_final": c_final,
        "mu_error_l1": np.linalg.norm(mu_theta - mu_E, ord=1),
        "pi_error_max": np.max(np.abs(pi_theta - pi_E)),
        "pi_error_fro": pi_error_fro,
        "expert_action_mass_unweighted": expert_action_mass_unweighted,
        "expert_action_mass_weighted": expert_action_mass_weighted,
        "phi_error": np.linalg.norm(phi_exp_theta - phi_exp_E),
        "history_true_mfe": history_true_mfe,
        "mu_history_true": mu_history_true,
        "pi_history_true": pi_history_true,
        "history_score_rkhs": history_score_rkhs,
        "history_grad_norm_rkhs": history_grad_norm_rkhs,
        "history_mu_error_rkhs": history_mu_error_rkhs,
        "history_phi_error_rkhs": history_phi_error_rkhs,
        "history_monotonicity_rkhs": history_monotonicity_rkhs,
        "history_pi_entries_rkhs": history_pi_entries_rkhs,
    }

    return results


# SUMMARY
#-------------------------------------------------------------

def summarize_results(results):
    # Collect the main forward-MFE and RKHS-IRL diagnostics as text.
    
    lines = []

    lines.append("================ CLASSICAL EXPERT MFE ================")
    lines.append(f"mu_E =\n{results['mu_E']}")
    lines.append(f"\npi_E =\n{results['pi_E']}")
    lines.append(f"\nrho_E = {results['rho_E']}")
    lines.append(f"\nh_E =\n{results['h_E']}")

    lines.append("\n================ RKHS IRL RESULTS ================")
    lines.append(f"Final score = {results['score']}")
    lines.append(f"Final ||grad V(theta)|| = {results['grad_norm']}")
    lines.append(f"\nRecovered pi_theta =\n{results['pi_theta']}")
    lines.append(f"\nRecovered mu_theta =\n{results['mu_theta']}")
    lines.append(f"\nmu_theta - mu_E =\n{results['mu_theta'] - results['mu_E']}")
    lines.append(f"\npi_theta - pi_E =\n{results['pi_theta'] - results['pi_E']}")
    lines.append(f"\nL1 error in mu = {results['mu_error_l1']}")
    lines.append(f"Max entrywise error in pi = {results['pi_error_max']}")
    lines.append(f"Frobenius policy error = {results['pi_error_fro']}")
    lines.append(
        f"Recovered probability mass on expert greedy actions "
        f"(unweighted) = {results['expert_action_mass_unweighted']}"
    )
    lines.append(
        f"Recovered probability mass on expert greedy actions "
        f"(mu_E-weighted) = {results['expert_action_mass_weighted']}"
    )
    lines.append(f"Feature-moment error = {results['phi_error']}")
    lines.append(f"\nK bound = {results['K_bound']}")
    lines.append(f"L bound = {results['L_bound']}")
    lines.append(f"kappa = {results['kappa']}")

    return "\n".join(lines)