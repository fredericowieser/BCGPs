# app.py
import streamlit as st
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# -------------------------------
# 1) PDE logic & GP helper functions
# -------------------------------

def second_derivative(fn, x):
    """
    Compute second derivative (fn''(x)) in 1D using JAX auto-diff.
    """
    dfn_dx = jax.grad(fn)       # first derivative
    d2fn_dx2 = jax.grad(dfn_dx) # second derivative
    return d2fn_dx2(x)

def phi(x):
    """
    Approximate distance function for [0,1] => ensures u(0)=u(1)=0.
    """
    return x * (1.0 - x)

def rbf_kernel(params, xa, xb):
    """
    Standard squared-exponential kernel for the latent GP \hat{u}.
    """
    amplitude = params["amplitude"]
    lengthscale = params["lengthscale"]
    sqdist = (xa - xb)**2
    return amplitude**2 * jnp.exp(-0.5 * sqdist / (lengthscale**2))

def bcgp_kernel(params, xa, xb):
    """
    Boundary-constrained kernel: phi(x)*phi(x') * rbf_kernel(x,x').
    Imposes u(0)=u(1)=0 automatically for any linear combination of kernel basis.
    """
    return phi(xa)*phi(xb)*rbf_kernel(params, xa, xb)

def neg_u_dd(params, x):
    """
    Evaluate -u''(x) for u(x) = sum_j alpha_j k_bcgp(x, Xcol_j).
    PDE collocation method:  alpha_j's are fitted to enforce -u''(x) ~ 2.0 at x_f.
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

def u_true(x):
    """
    Analytic solution for PDE + BC:  -u''(x)=2, u(0)=0, u(1)=0 => u(x)= x - x^2.
    """
    return x - x**2

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
        st.markdown("Partial Differential Equation (PDE)")

        st.latex(r"-u''(x) = 2, \quad x \in (0,1)")

        st.markdown("We exactly impose the boundary conditions:")

        st.latex(r"u(0) = u(1) = 0")

        st.markdown("by using the function:")

        st.latex(r"\phi(x) = x(1-x)")
        error_bars = st.checkbox("Show error bars", value=True)
        n_points = st.slider("Number of PDE collocation points", 
                            min_value=2, max_value=40, value=10, step=1)
        n_train_iters = st.slider("Number of training iterations", 
                                min_value=10, max_value=1000, value=200, step=10)
        lr = st.slider("Learning rate", min_value=1e-3, max_value=1e-1, 
                    value=1e-2, step=1e-3, format="%.3f")

    # Button to start training
    if st.button("Run Training"):
        st.write(f"Training with n_points={n_points}, n_train_iters={n_train_iters}, lr={lr:.3f}...")
        
        # Collocation data
        # skip endpoints to avoid duplication with BC
        Xcol = jnp.linspace(0, 1, n_points+2)[1:-1]
        Ycol = f_true(Xcol)

        # Initialize parameters
        params = {
            "amplitude": 1.0,
            "lengthscale": 0.2,
            "alpha": jnp.zeros(shape=(len(Xcol),)), 
            "Xcol": Xcol,
            "ycol": Ycol
        }

        # We'll plot every "plot_frequency" steps
        # plot_frequency = max(1, n_train_iters // 5)
        plot_frequency = 1

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

                xs_plot = jnp.linspace(0.0, 1.0, 200)
                u_mean_arr = jax.vmap(u_pred)(xs_plot)
                u_std_arr  = jax.vmap(u_std)(xs_plot)
                u_exact_arr= u_true(xs_plot)

                # Matplotlib figure
                fig, ax = plt.subplots(figsize=(6,4))
                ax.plot(xs_plot, u_exact_arr, 'k-',  label="True: x - x^2")
                ax.plot(xs_plot, u_mean_arr,  'b--', label="BCGP Mean")
                if error_bars:
                    ax.fill_between(np.array(xs_plot),
                                    np.array(u_mean_arr - 2*u_std_arr),
                                    np.array(u_mean_arr + 2*u_std_arr),
                                    color='blue', alpha=0.2, 
                                    label="±2σ region")
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


# Call the main function
if __name__ == "__main__":
    run_bcgp_app()
