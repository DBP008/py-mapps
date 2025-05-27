import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import altair as alt
    import polars as pl
    import math


@app.cell
def _(combined_charts, form_elements, output_md):
    mo.vstack([form_elements, combined_charts, output_md])
    return


@app.function
# --- P-SHG Simulation ---


def simulate_shg(theta_L_rad, theta_F_rad, gamma, k=1, y0=0):
    delta = theta_L_rad - theta_F_rad
    term1 = np.sin(2 * delta) ** 2
    term2_base = np.sin(delta) ** 2 + gamma * np.cos(delta) ** 2
    term2 = term2_base**2
    intensity = k * (term1 + term2) + y0
    return intensity


@app.function
def get_theta_phasor(shg_intensity_full, theta_L_full_rad):
    """Calculates (g,s) for theta-plot. Slices [0, pi) internally."""
    if len(shg_intensity_full) != len(theta_L_full_rad):
        print(f"shg len: {len(shg_intensity_full)}")
        print(f"theta_L len: {len(theta_L_full_rad)}")
        raise ValueError(
            "Intensity and angle arrays must match for theta-phasor."
        )
    if not shg_intensity_full.size:
        return 0.0, 0.0

    indices = np.where((theta_L_full_rad >= 0) & (theta_L_full_rad < np.pi))[0]
    if not indices.size:
        print("Warning: No data in [0, pi) for theta-phasor.")
        return 0.0, 0.0

    shg_segment = shg_intensity_full[indices]
    N = len(shg_segment)
    if N <= 1:
        return 0.0, 0.0  # DFT k=1 needs >1 point

    dft = np.fft.fft(shg_segment)
    sum_I = np.real(dft[0])
    if sum_I == 0:
        return 0.0, 0.0

    g = np.real(dft[1]) / sum_I
    s = -np.imag(dft[1]) / sum_I
    return g, s


@app.function
def get_gamma_phasor(shg_intensity_full, theta_L_full_rad, theta_F_pred_rad):
    """Calculates (g,s) for gamma-plot. Infers delta_theta."""
    if len(shg_intensity_full) != len(theta_L_full_rad):
        print(
            f"shg len: {len(shg_intensity_full)} != theta len {len(theta_L_full_rad)}"
        )
        raise ValueError(
            "Intensity and angle arrays must match for gamma-phasor."
        )
    if len(shg_intensity_full) < 2:
        return 0.0, 0.0

    # delta_theta_rad = theta_L_full_rad[1] - theta_L_full_rad[0]

    unique_thetas = np.unique(np.sort(theta_L_full_rad))
    if len(unique_thetas) < 2:
        return 0.0, 0.0
    delta_theta_rad = np.median(np.diff(unique_thetas))
    if (
        delta_theta_rad <= 1e-9
    ):  # Avoid division by zero if sampling is too dense or problematic
        print(
            f"Warning: Inferred delta_theta_rad is too small or zero ({delta_theta_rad:.2e})."
        )
        return 0.0, 0.0

    num_pts_segment = int(round((np.pi / 2) / delta_theta_rad))
    if num_pts_segment <= 1:
        return 0.0, 0.0

    idx_start = np.argmin(np.abs(theta_L_full_rad - theta_F_pred_rad))
    indices = (idx_start + np.arange(num_pts_segment)) % len(theta_L_full_rad)
    shg_segment = shg_intensity_full[indices]

    N_seg = len(shg_segment)
    if N_seg <= 1:
        return 0.0, 0.0

    dft = np.fft.fft(shg_segment)
    sum_I_seg = np.real(dft[0])
    if sum_I_seg == 0:
        return 0.0, 0.0

    g = np.real(dft[1]) / sum_I_seg
    s = -np.imag(dft[1]) / sum_I_seg
    return g, s


@app.function
def extract_theta_F(g, s):  # theta_F in radians
    angle_2thF = np.arctan2(s, g)
    theta_F = angle_2thF / 2.0
    theta_F = np.where(theta_F < 0, theta_F + np.pi, theta_F)
    return theta_F


@app.function
def extract_gamma(g_exp, s_exp, gamma_ref_tuples):
    min_dist_sq, est_gamma = float("inf"), -1
    if not gamma_ref_tuples:
        return est_gamma
    for gamma_val, g_ref, s_ref in gamma_ref_tuples:
        dist_sq = (g_exp - g_ref) ** 2 + (s_exp - s_ref) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq, est_gamma = dist_sq, gamma_val
    return est_gamma


@app.function
def generate_theta_ref_curve(
    gamma_for_ref, num_theta_F_pts, theta_L_full_rad_for_sim
):
    """
    Generates theta-phasor reference curve tuples (theta_F_val, g, s).
    theta_F_val will be in radians.
    """
    theta_F_values_rad = np.linspace(0, np.pi, num_theta_F_pts, endpoint=False)
    ref_tuples = []
    for th_F_iter_rad in theta_F_values_rad:
        shg_ref_full = simulate_shg(
            theta_L_full_rad_for_sim, th_F_iter_rad, gamma_for_ref
        )

        g, s = get_theta_phasor(shg_ref_full, theta_L_full_rad_for_sim)
        ref_tuples.append((th_F_iter_rad, g, s))
    return ref_tuples


@app.function
def generate_gamma_ref_curve(
    theta_F_rad_for_ref,
    num_gamma_pts,
    gamma_start,
    gamma_end,
    theta_L_full_rad_for_sim,
):
    """Generates gamma-phasor reference curve tuples (gamma_val, g, s)."""
    gamma_values = np.linspace(gamma_start, gamma_end, num_gamma_pts)
    ref_tuples = []
    for gamma_iter in gamma_values:
        shg_ref_full = simulate_shg(
            theta_L_full_rad_for_sim, theta_F_rad_for_ref, gamma_iter
        )
        g, s = get_gamma_phasor(
            shg_ref_full, theta_L_full_rad_for_sim, theta_F_rad_for_ref
        )
        ref_tuples.append((gamma_iter, g, s))
    return ref_tuples


@app.cell
def _():
    form_elements = mo.md(
        """
        **Select θ and γ to simulate P-SHG curve**

        Fiber Orientation θ_F = {theta_F_true_deg} deg

        Disorder Parameter γ = {gamma_true}

        Sampling step Δθ_L = {delta_theta_deg} deg
        """
    ).batch(
        theta_F_true_deg=mo.ui.slider(
            start=0,
            stop=180,
            step=1,
            value=30,
            show_value=True,
        ),
        gamma_true=mo.ui.slider(
            start=0.1,
            stop=5,
            step=0.01,
            value=1.5,
            show_value=True,
        ),
        delta_theta_deg=mo.ui.slider(
            start=1,
            stop=20,
            step=1,
            value=10,
            show_value=True,
        ),
    )
    return (form_elements,)


@app.cell
def _(form_elements):
    # Get values from form
    theta_F_true_deg = form_elements.value["theta_F_true_deg"]
    gamma_true = form_elements.value["gamma_true"]
    delta_theta_deg = form_elements.value["delta_theta_deg"]

    theta_F_true_rad = np.deg2rad(theta_F_true_deg)
    delta_theta_rad = np.deg2rad(delta_theta_deg)

    num_steps = int(round((2 * np.pi) / delta_theta_rad))
    if num_steps == 0:
        num_steps = 1
    theta_L_full_rad = np.linspace(0, 2 * np.pi, num_steps, endpoint=False)

    if len(theta_L_full_rad) < 4:
        mo.md(
            f"""## Error
    		Δθ_L ({delta_theta_deg}°) is too large for a reliable P-SHG scan.
    		Please choose a smaller sampling step.
    		<br>
    		Number of points: {len(theta_L_full_rad)} (need at least 4).
    		"""
        )

    shg_exp_full = simulate_shg(theta_L_full_rad, theta_F_true_rad, gamma_true)

    # --- θ-Phasor ---
    g_th_exp, s_th_exp = get_theta_phasor(shg_exp_full, theta_L_full_rad)
    theta_F_pred_rad = extract_theta_F(g_th_exp, s_th_exp)

    theta_ref_tuples = generate_theta_ref_curve(
        gamma_for_ref=1e10,
        num_theta_F_pts=201,
        theta_L_full_rad_for_sim=theta_L_full_rad,
    )
    df_theta_ref = pl.DataFrame(
        theta_ref_tuples, schema=["theta_F_val_rad", "g", "s"], orient="row"
    )
    df_theta_ref = df_theta_ref.sort(by="theta_F_val_rad")

    # --- γ-Phasor ---

    gamma_ref_tuples = generate_gamma_ref_curve(
        theta_F_rad_for_ref=theta_F_pred_rad,
        num_gamma_pts=100,
        gamma_start=0.1,
        gamma_end=5.0,
        theta_L_full_rad_for_sim=theta_L_full_rad,
    )
    df_gamma_ref = pl.DataFrame(
        gamma_ref_tuples, schema=["gamma", "g", "s"], orient="row"
    )

    g_ga_exp, s_ga_exp = get_gamma_phasor(
        shg_exp_full, theta_L_full_rad, theta_F_pred_rad
    )

    gamma_pred = extract_gamma(g_ga_exp, s_ga_exp, gamma_ref_tuples)

    # --- Altair Plotting ---
    plot_width = 200
    plot_height = 200

    df_shg_exp = pl.DataFrame(
        {"theta_L_deg": np.rad2deg(theta_L_full_rad), "intensity": shg_exp_full}
    )
    chart_shg_curve = (
        alt.Chart(df_shg_exp)
        .mark_line()
        .encode(
            x=alt.X("theta_L_deg:Q", title="θ_L (deg)").scale(zero=False),
            y=alt.Y("intensity:Q", title="SHG Intensity").scale(zero=False),
            tooltip=["theta_L_deg", "intensity"],
        )
        .properties(
            title=f"P-SHG Signal",
            width=plot_width,
            height=plot_height,
        )
    )

    phasor_domain = [-0.8, 0.8]

    # THETA Plot

    line_theta_ref = (
        alt.Chart(df_theta_ref)
        .mark_point(size=20, opacity=0.7, filled=True)
        .encode(
            x=alt.X("g:Q", scale=alt.Scale(domain=phasor_domain), title="g"),
            y=alt.Y("s:Q", scale=alt.Scale(domain=phasor_domain), title="s"),
            color=alt.Color("theta_F_val_rad", title="Ref θ [rad]").scale(
                scheme="rainbow"
            ),
            order="theta_F_val_rad",
            tooltip=[
                alt.Tooltip("g:Q", format=".3f"),
                alt.Tooltip("s:Q", format=".3f"),
                alt.Tooltip("theta_F_val_rad:Q", format=".3f"),
            ],
        )
    )

    df_theta_exp = pl.DataFrame(
        {
            "g": [g_th_exp],
            "s": [s_th_exp],
            "θ_F_pred": [np.rad2deg(theta_F_pred_rad)],
        }
    )
    point_theta_exp = (
        alt.Chart(df_theta_exp)
        .mark_point(size=100, color="red", filled=True, opacity=0.8)
        .encode(
            x="g:Q",
            y="s:Q",
            tooltip=[
                alt.Tooltip("g:Q", format=".3f"),
                alt.Tooltip("s:Q", format=".3f"),
                alt.Tooltip("θ_F_pred:Q", format=".3f"),
            ],
        )
    )
    chart_theta_phasor = (
        alt.layer(line_theta_ref, point_theta_exp)
        .properties(
            title=f"θ-Phasor Plot",
            width=plot_width,
            height=plot_height,
        )
        .interactive()
    )

    # GAMMA PLOT

    scatter_gamma_ref = (
        alt.Chart(df_gamma_ref)
        .mark_point(size=20, opacity=0.7, filled=True)
        .encode(
            x=alt.X("g:Q", scale=alt.Scale(domain=phasor_domain), title="g"),
            y=alt.Y("s:Q", scale=alt.Scale(domain=phasor_domain), title="s"),
            color=alt.Color(
                "gamma:Q",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title="Ref γ"),
            ),
            tooltip=[
                alt.Tooltip("g:Q", format=".3f"),
                alt.Tooltip("s:Q", format=".3f"),
                alt.Tooltip("gamma:Q", format=".2f"),
            ],
        )
    )
    df_gamma_exp = pl.DataFrame(
        {
            "g": [g_ga_exp],
            "s": [s_ga_exp],
            "γ_pred": [gamma_pred],
        }
    )
    point_gamma_exp = (
        alt.Chart(df_gamma_exp)
        .mark_point(size=100, color="red", filled=True, opacity=0.8)
        .encode(
            x="g:Q",
            y="s:Q",
            tooltip=[
                alt.Tooltip("g:Q", format=".3f"),
                alt.Tooltip("s:Q", format=".3f"),
                alt.Tooltip("γ_pred:Q", format=".2f"),
            ],
        )
    )
    chart_gamma_phasor = (
        alt.layer(scatter_gamma_ref, point_gamma_exp)
        .properties(
            title=f"γ-Phasor Plot",
            width=plot_width,
            height=plot_height,
        )
        .interactive()
    )

    combined_charts = alt.hconcat(
        chart_shg_curve, chart_theta_phasor, chart_gamma_phasor
    ).resolve_scale(color="independent")

    output_md = mo.md(f"""
    ### Simulation & Prediction Summary:
    - **Input θ_F**: {theta_F_true_deg:.1f}°
    - **Input γ**: {gamma_true:.2f}
    - **Input Δθ_L (sampling step)**: {delta_theta_deg}°
    - **Predicted θ_F**: {float(np.rad2deg(theta_F_pred_rad)):.1f}° 
      (from g={float(g_th_exp):.3f}, s={s_th_exp:.3f})
    - **Predicted γ **: {float(gamma_pred):.2f} 
      (from g={float(g_ga_exp):.3f}, s={float(s_ga_exp):.3f})
    """)
    return combined_charts, output_md


if __name__ == "__main__":
    app.run()
