import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
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
    import altair as alt
    return alt, get_gamma_phasor, mo, np, pl, simulate_shg


@app.cell
def _(pl):
    pl.Float64
    return


@app.cell
def _(alt, get_gamma_phasor, mo, np, pl, simulate_shg):
    theta_F_rad = np.deg2rad(30)  # Example fixed theta_F
    gamma_values = np.linspace(0.1, 5, 5)  # Example gamma values
    theta_L_full_rad = np.linspace(0, 2 * np.pi, 100)
    harmonic = 2

    plot_width = 200
    plot_height = 200
    df_shg_exp = []

    for gamma in gamma_values:
        shg_exp_full = simulate_shg(theta_L_full_rad, theta_F_rad, gamma)

        g_ga_exp, s_ga_exp = get_gamma_phasor(
            shg_exp_full, theta_L_full_rad, theta_F_rad, harmonic=harmonic
        )
        gamma_ref_tuples = [(gamma, g_ga_exp, s_ga_exp)]

        row = {
            "theta_L_deg": list(theta_L_full_rad),
            "intensity": list(shg_exp_full),
            "gamma": gamma,
            "g_ga_exp": g_ga_exp,
            "s_ga_exp": s_ga_exp,
        }
        df_shg_exp.append(row)

    # Prepare data for plotting
    df_shg_exp = pl.DataFrame(df_shg_exp)

    melted_df = df_shg_exp.explode(["theta_L_deg", "intensity"])
    melted_df = melted_df.with_columns(
        pl.col("theta_L_deg")
        .map_elements(lambda x: x / np.pi, return_dtype=pl.Float64)
        .alias("theta_L_pi_units")
    )

    # Create the Altair chart
    pshg_chart = (
        alt.Chart(melted_df)
        .mark_line()
        .encode(
            x=alt.X(
                "theta_L_pi_units:Q",
                title="θ_L (π units)",
                axis=alt.Axis(
                    values=[0, 0.5, 1, 1.5, 2],  # Key fractions of π
                    labelExpr="datum.value == 0 ? '0' : datum.value == 1 ? 'π' : datum.value == 2 ? '2π' : datum.value == 0.5 ? 'π/2' : datum.value == 1.5 ? '3π/2' : ''",
                ),
            ),
            y=alt.Y("intensity:Q", title="Intensity"),
            color=alt.Color("gamma:Q", title="Gamma").scale(scheme="spectral"),
            tooltip=["theta_L_deg", "intensity", "gamma"],
        )
        .properties(
            title="Intensity vs θ_L for Different Gamma Values",
            # width=400,
            # height=300,
        )
    )

    phasor_domain = [-0.8, 0.8]

    gamma_phasor = (
        alt.Chart(melted_df)
        .mark_point()
        .encode(
            x=alt.X(
                "g_ga_exp:Q", title="G", scale=alt.Scale(domain=phasor_domain)
            ),
            y=alt.Y(
                "s_ga_exp:Q", title="S", scale=alt.Scale(domain=phasor_domain)
            ),
            color=alt.Color("gamma:Q", title="Gamma").scale(scheme="spectral"),
            tooltip=["gamma", "g_ga_exp", "s_ga_exp"],
        )
        .properties(
            title=f"Gamma Phasor (harmonic={harmonic})",  # width=300, height=300
        )
    )

    mo.ui.altair_chart(alt.hconcat(pshg_chart, gamma_phasor))
    return


if __name__ == "__main__":
    app.run()
