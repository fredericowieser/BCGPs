import streamlit as st
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# -------------------------------
# 1) PDE logic & GP helper functions
# -------------------------------

def fit_quadratic_and_extract_factors(xs_plot, u_mean_arr):
    A, B, C = np.polyfit(xs_plot, u_mean_arr, 2)  # Fit quadratic

    # x = -b +/- sqrt(b^2 - 4ac) / 2a
    v1 = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
    v2 = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)

    return min(v1, v2), max(v1, v2)  # Ensure v1 < v2

def second_derivative(fn, x):
    dfn_dx = jax.grad(fn) # first derivative
    d2fn_dx2 = jax.grad(dfn_dx) # second derivative
    return d2fn_dx2(x)

def phi(params, x):
    """
    Approximate distance function for [boundary_1, boundary_2].
    Enforces u(boundary_1) = u(boundary_2) = 0 by construction.
    """
    b1 = params["boundary_1"]
    b2 = params["boundary_2"]
    return (x - b1) * (b2 - x)

def rbf_kernel(params, xa, xb):
    """
    Squared-exponential kernel for the latent GP.
    """
    amplitude = params["amplitude"]
    lengthscale = params["lengthscale"]
    sqdist = (xa - xb)**2
    return amplitude**2 * jnp.exp(-0.5 * sqdist / (lengthscale**2))

def bcgp_kernel(params, xa, xb):
    """
    Boundary-constrained kernel that uses the ADF phi(params, x)
    to ensure the solution is zero at x=boundary_1 and x=boundary_2.
    """
    amplitude   = params["amplitude"]
    lengthscale = params["lengthscale"]
    # standard RBF part
    sqdist      = (xa - xb)**2
    rbf_val     = amplitude**2 * jnp.exp(-0.5 * sqdist / (lengthscale**2))
    # scaled by phi(xa)*phi(xb)
    return phi(params, xa) * phi(params, xb) * rbf_val

def neg_u_dd(params, x):
    """
    Evaluate -u''(x) for u(x) = sum_j alpha_j k_bcgp(x, Xcol_j).
    PDE collocation method:  alpha_j's are fitted to enforce -u''(x) = 2.0 at x_f.
    """
    alpha = params["alpha"]
    Xcol  = params["Xcol"]

    def u_of_x(xx):
        vals = [alpha[j]*bcgp_kernel(params, xx, Xcol[j]) for j in range(len(Xcol))]
        return jnp.sum(jnp.array(vals))

    return -second_derivative(u_of_x, x)

def f_true(x):
    """
    PDE forcing: -u''(x) = 2 =>  f(x)=2, for x in (0,1).
    """
    return 2.0*jnp.ones_like(x)

def u_true(params, x):
    """
    Analytic solution for PDE + BC:  -u''(x)=2, u(0)=0, u(1)=0 => u(x)= x - x^2.
    """
    b1 = params["boundary_1"]
    b2 = params["boundary_2"]
    return (x - b1) * (b2 - x)

def loss_fn(params):
    """
    Mean squared error between PDE residual -u''(x_f) and y_f=2.0
    at the collocation points x_f.
    """
    preds = jax.vmap(lambda xx: neg_u_dd(params, xx))(params["Xcol"])
    return jnp.mean((preds - params["ycol"])**2)

@jax.jit
def update(params: Dict[str, jnp.ndarray], lr=1e-2):
    """
    One step of gradient descent on the PDE MSE.
    """
    g = jax.grad(loss_fn)(params)
    new_params = {}
    for k,v in params.items():
        if k in ["amplitude","lengthscale","alpha"]:
            new_params[k] = v - lr*g[k]
        else:
            new_params[k] = v
    return new_params

# -------------------------------
# 2) Streamlit App
# -------------------------------

def run_bcgp_app():
    st.title("Boundary-Constrained Gaussian Process (BCGP) for 1D Poisson PDE")

    # User controls
    with st.sidebar:
        st.header("Hyperparameters & Settings")
        
        # Boundary conditions
        st.markdown("Boundary conditions:")
        boundary_1 = st.slider("Boundary 1", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
        boundary_2 = st.slider("Boundary 2", min_value=0.5, max_value=1.0, value=1.0, step=0.01)
        st.markdown("---")
        
        # Hyperparameters
        st.markdown("Initial Kernel hyperparameters:")
        amplitude = st.slider("Amplitude", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        lengthscale = st.slider("Lengthscale", min_value=0.1, max_value=1.0, value=0.2, step=0.1)
        st.markdown("---")

        # Plotting settings
        st.markdown("Plotting Settings:")
        error_bars = st.checkbox("Show Uncertainty", value=False)
        plot_true_base_problem = st.checkbox("Plot Analytical Solution", value=True)
        adaptive_lengthscale = st.checkbox("Adaptive Plot Axes", value=False)
        plot_frequency = st.slider(
            "Plot frequency",
            min_value=1, 
            max_value=26,
            value=2,
            step=1,
        )
        st.markdown("---")

        # Training settings
        st.header("Training")
        n_points = st.slider(
            "Collocation points",
            min_value=2,
            max_value=40,
            value=10, 
            step=1,
        )
        n_train_iters = st.slider(
            "Training iterations",
            min_value=10, 
            max_value=1000,
            value=100,
            step=10,
        )
        lr = st.slider("Learning rate", min_value=1e-3, max_value=1e-1, value=1e-2, step=1e-3, format="%.3f")


    # Button to start training

    if st.button("Run Training"):
        st.write(f"Starting training with {n_train_iters} iterations...")
        st.write(f"Solving for -u''(x) = 2 on ({boundary_1},{boundary_2}) with u({boundary_1})=u({boundary_2})=0")
        st.write(f"phi(x) = (x - {boundary_1})({boundary_2} - x)")
        
        # Collocation data
        # skip endpoints to avoid duplication with BC
        Xcol = jnp.linspace(boundary_1, boundary_2, n_points+2)[1:-1]
        Ycol = f_true(Xcol)

        # Initialize parameters
        params = {
            "amplitude": amplitude,
            "lengthscale": lengthscale,
            "alpha": jnp.zeros(shape=(len(Xcol),)), 
            "Xcol": Xcol,
            "ycol": Ycol,
            "boundary_1": boundary_1,
            "boundary_2": boundary_2,
        }

        # Create a placeholder for the plot
        plot_placeholder = st.empty()

        # Training loop
        for step in range(n_train_iters+1):
            params = update(params, lr=lr)

            # Plot every few steps
            if step % plot_frequency == 0 or step == n_train_iters:
                # Evaluate PDE MSE
                mse_val = loss_fn(params)

                # Build final solution predictor
                alpha = params["alpha"]
                def u_pred(xx):
                    vals = [alpha[j]*bcgp_kernel(params, xx, Xcol[j]) 
                            for j in range(len(Xcol))]
                    return jnp.sum(jnp.array(vals))

                # For approximate stdev, use bcgp_kernel(x,x)
                def u_std(xx):
                    return jnp.sqrt(bcgp_kernel(params, xx, xx) + 1e-6)

                xs_plot = jnp.linspace(boundary_1, boundary_2, 200)
                u_mean_arr = jax.vmap(u_pred)(xs_plot)
                u_std_arr  = jax.vmap(u_std)(xs_plot)

                # Matplotlib figure
                fig, ax = plt.subplots(figsize=(6,4))
                if not adaptive_lengthscale:
                    ax.set_ylim([-0.2, 0.5])
                    ax.set_xlim([0, 1])
                if plot_true_base_problem:
                    u_exact_arr= u_true(params, xs_plot)
                    ax.plot(xs_plot, u_exact_arr, 'k-',  label="Analytical Soln.")
                ax.plot(xs_plot, u_mean_arr,  'b--', label="BCGP Mean")
                if error_bars:
                    ax.fill_between(np.array(xs_plot),
                                    np.array(u_mean_arr - u_std_arr),
                                    np.array(u_mean_arr + u_std_arr),
                                    color='blue', alpha=0.2, 
                                    label="±σ region")
                ax.scatter(np.array(Xcol), 
                           np.zeros_like(Xcol), 
                           marker='x', color='red', 
                           label="Colloc. pts")
                ax.set_title(f"Step={step}, PDE MSE={mse_val:.2e}")
                ax.set_xlabel("x")
                ax.set_ylabel("u(x)")
                ax.grid(True)
                ax.legend()
                plot_placeholder.pyplot(fig)
        
        st.success("Training complete!")

        # Compare Analytical Solution with BCGP
        v1, v2 = fit_quadratic_and_extract_factors(xs_plot, u_mean_arr)
        st.write(f"Analytical solution: u(x) = (x - {boundary_1})({boundary_2} - x)")
        st.write(f"BCGP solution: u(x) = (x - {v1:.4f})({v2:.4f} - x) (approx. from quadratic fit)")


    st.markdown("---")
    st.markdown("### PDE & Boundary Conditions")

    st.latex(r"-u''(x) = 2, \quad x \in (b_1,b_2)")

    st.latex(r"u(b_1) = 0, \quad u(b_2) = 0")

    st.markdown("---")
    st.markdown("### Analytical Solution")

    st.markdown("#### Step 1: Solve the Differential Equation")

    st.latex(r"u''(x) = -2")

    st.latex(r"u'(x) = \int -2\,dx = -2x + C_1")

    st.latex(r"u(x) = \int (-2x + C_1)\,dx = -x^2 + C_1 x + C_2")

    st.markdown("#### Step 2: Apply the Boundary Conditions")

    st.latex(r"u(b_1) = -b_1^2 + C_1 b_1 + C_2 = 0")

    st.latex(r"u(b_2) = -b_2^2 + C_1 b_2 + C_2 = 0")

    st.markdown("#### Step 3: Solve for Constants \\( C_1 \\) and \\( C_2 \\)")

    st.latex(r"\begin{cases} -b_1^2 + C_1 b_1 + C_2 = 0, \\ -b_2^2 + C_1 b_2 + C_2 = 0. \end{cases}")

    st.markdown("Subtracting the two equations to eliminate \\( C_2 \\):")

    st.latex(r"\left(-b_2^2 + C_1 b_2 + C_2\right) - \left(-b_1^2 + C_1 b_1 + C_2\right) = 0")

    st.latex(r"-b_2^2 + C_1 b_2 - (-b_1^2 + C_1 b_1) = 0")

    st.latex(r"-b_2^2 + C_1 b_2 + b_1^2 - C_1 b_1 = 0")

    st.latex(r"C_1 (b_2 - b_1) = b_2^2 - b_1^2")

    st.latex(r"C_1 = \frac{b_2^2 - b_1^2}{b_2 - b_1} = b_2 + b_1")

    st.markdown("Now substitute \\( C_1 \\) into one of the boundary equations:")

    st.latex(r"-b_1^2 + (b_1 + b_2) b_1 + C_2 = 0")

    st.latex(r"-b_1^2 + b_1^2 + b_1 b_2 + C_2 = 0")

    st.latex(r"C_2 = -b_1 b_2")

    st.markdown("#### Step 4: Final Solution")

    st.latex(r"u(x) = -x^2 + (b_1 + b_2) x - b_1 b_2")

    st.markdown("Alternatively, this can be rewritten as:")

    st.latex(r"u(x) = (x - b_1)(b_2 - x)")


# Call the main function
if __name__ == "__main__":
    run_bcgp_app()
