from pathlib import Path

import plotly.graph_objects as go


def _common_layout(title: str) -> dict:
    return dict(
        template="simple_white",
        title=f"<b>{title}</b>",
        font=dict(family="Arial Black, Arial, sans-serif", size=28, color="#111111"),
        margin=dict(l=120, r=60, t=120, b=140),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )


def _style_axes(fig: go.Figure, y_title: str, y_range: list[float], x_title: str = "") -> None:
    fig.update_yaxes(
        title=f"<b>{y_title}</b>",
        title_font=dict(size=30),
        tickfont=dict(size=24),
        range=y_range,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.28)",
        zerolinewidth=2,
    )
    fig.update_xaxes(
        title=f"<b>{x_title}</b>" if x_title else "",
        tickfont=dict(size=22),
        tickangle=-20,
    )


def main() -> None:
    out = Path("docs/paper/IEEE-conference-template-062824/figures")
    out.mkdir(parents=True, exist_ok=True)

    variants = ["baseline", "arcface", "contrastive", "triplet", "multi_similarity"]
    acc = [0.9924, 0.7979, 0.9931, 0.9932, 0.9946]
    f1 = [0.9960, 0.7934, 0.9964, 0.9964, 0.9972]
    runtime = [107.1, 93.6, 130.8, 132.6, 128.3]
    lowfpr = [0.9754, 0.0000, 0.9533, 0.9351, 0.9851]

    # Accuracy/F1 grouped bars
    fig = go.Figure()
    fig.add_bar(
        name="<b>Accuracy</b>",
        x=variants,
        y=acc,
        marker_color="#1f77b4",
        marker_line_color="#0b3d66",
        marker_line_width=1.8,
        text=[f"<b>{v:.4f}</b>" for v in acc],
        textposition="outside",
        textfont=dict(size=20, color="#0f2f4f"),
    )
    fig.add_bar(
        name="<b>F1</b>",
        x=variants,
        y=f1,
        marker_color="#17becf",
        marker_line_color="#0f6d78",
        marker_line_width=1.8,
        text=[f"<b>{v:.4f}</b>" for v in f1],
        textposition="outside",
        textfont=dict(size=20, color="#0f4b52"),
    )
    fig.update_layout(
        **_common_layout("DML Quality Comparison (5-seed Mean)"),
        barmode="group",
        bargap=0.22,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            font=dict(size=22),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
        ),
    )
    _style_axes(fig, "Score", [0.75, 1.05])
    fig.write_image(str(out / "dml_acc_f1.png"), width=2400, height=1400, scale=3)

    # Runtime bars
    baseline_runtime = runtime[0]
    fig = go.Figure()
    fig.add_bar(
        x=variants,
        y=runtime,
        marker_color="#9467bd",
        marker_line_color="#593775",
        marker_line_width=1.8,
        text=[f"<b>{v:.1f}s<br>({v / baseline_runtime * 100:.0f}%)</b>" for v in runtime],
        textposition="outside",
        textfont=dict(size=20, color="#4d2d63"),
    )
    fig.update_layout(**_common_layout("Runtime per Variant"))
    _style_axes(fig, "Runtime (s)", [0, 160])
    fig.write_image(str(out / "dml_runtime.png"), width=2400, height=1400, scale=3)

    # Low-FPR bars
    fig = go.Figure()
    fig.add_bar(
        x=variants,
        y=lowfpr,
        marker_color="#2ca9df",
        marker_line_color="#0b5c7d",
        marker_line_width=1.8,
        text=[f"<b>{v:.4f}</b>" for v in lowfpr],
        textposition="outside",
        textfont=dict(size=20, color="#0c4f6b"),
    )
    fig.add_hline(
        y=0.95,
        line_dash="dash",
        line_color="#555555",
        line_width=4,
        annotation_text="<b>Target 0.95</b>",
        annotation_position="top left",
        annotation_font_size=22,
    )
    fig.update_layout(**_common_layout("Low-FPR Detection Sensitivity"))
    _style_axes(fig, "TPR @ FPR = 1e-2", [0, 1.08])
    fig.write_image(str(out / "dml_lowfpr.png"), width=2400, height=1400, scale=3)


if __name__ == "__main__":
    main()
