import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from py_mapps import (
        simulate_shg,
        get_theta_phasor,
        get_gamma_phasor,
        extract_theta_F,
        extract_gamma,
        generate_theta_ref_curve,
        generate_gamma_ref_curve,
    )
    import polars as pl
    import numpy as np
    import base64


@app.cell
def _():
    shg = pl.read_csv(base64.b64decode("W7VtXSAgIE1lYW4NCjAgICAzMi43MTc5DQoxICAgMzUuNTc0NA0KMiAgIDM2LjQ5MjMNCjMgICAzNy44MjU2DQo0ICAgMzkuMjY2Nw0KNSAgIDQxLjYyMDUNCjYgICA0Mi4yMjU2DQo3ICAgNDIuODYxNQ0KOCAgIDQzLjc2OTINCjkgICA0Mi4xNTM4DQoxMCAgIDQyLjMwNzcNCjExICAgNDAuOTc0NA0KMTIgICAzOC44NDEwDQoxMyAgIDM2LjE2OTINCjE0ICAgMzMuNjMwOA0KMTUgICAzMS45MjgyDQoxNiAgIDMwLjY2NjcNCjE3ICAgMzAuNjcxOA0KMTggICAzMi41MDI2DQoxOSAgIDMyLjk0ODcNCjIwICAgMzQuNDI1Ng0KMjEgICAzNy4yNTY0DQoyMiAgIDM4LjgzMDgNCjIzICAgNDAuMzc5NQ0KMjQgICA0MS41NDg3DQoyNSAgIDQxLjg5NzQNCjI2ICAgNDIuNjg3Mg0KMjcgICA0Mi41OTQ5DQoyOCAgIDQyLjE1MzgNCjI5ICAgNDEuNzM4NQ0KMzAgICAzOS4xNTM4DQozMSAgIDM3LjA4NzINCjMyICAgMzUuMTUzOA0KMzMgICAzMS45ODk3DQozNCAgIDMxLjYwNTENCjM1ICAgMzEuMjkyMw0KMzYgICAzMi4xNDg3DQo="), has_header=True, separator=" ")["Mean"]
    return (shg,)


@app.cell
def _(shg):
    delta_theta_deg = 10

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

    shg_exp_full = shg.to_numpy()
    shg_exp_full = shg_exp_full[: len(theta_L_full_rad)]

    # --- θ-Phasor ---
    g_th_exp, s_th_exp = get_theta_phasor(shg_exp_full, theta_L_full_rad)
    theta_F_pred_rad = extract_theta_F(g_th_exp, s_th_exp)


    # --- γ-Phasor ---

    gamma_ref_tuples = generate_gamma_ref_curve(
        theta_F_rad_for_ref=theta_F_pred_rad,
        num_gamma_pts=100,
        gamma_start=0.1,
        gamma_end=5.0,
        theta_L_full_rad_for_sim=theta_L_full_rad,
    )

    g_ga_exp, s_ga_exp = get_gamma_phasor(
        shg_exp_full, theta_L_full_rad, theta_F_pred_rad
    )

    gamma_pred = extract_gamma(g_ga_exp, s_ga_exp, gamma_ref_tuples)

    # --- Output ---

    output_md = mo.md(f"""
    ### Simulation & Prediction Summary:
    - **Input Δθ_L (sampling step)**: {delta_theta_deg}°
    - **Predicted θ_F**: {float(np.rad2deg(theta_F_pred_rad)):.1f}° 
      (from g={float(g_th_exp):.3f}, s={s_th_exp:.3f})
    - **Predicted γ **: {float(gamma_pred):.2f} 
      (from g={float(g_ga_exp):.3f}, s={float(s_ga_exp):.3f})
    """)

    output_md
    return


if __name__ == "__main__":
    app.run()
