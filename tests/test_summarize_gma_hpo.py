import runpy


MODULE = runpy.run_path("scripts/misc/summarize_gma_hpo.py", run_name="gma_summary")
ATTACK_PATTERN = MODULE["ATTACK_PATTERN"]
CONFIG_PATTERN = MODULE["CONFIG_PATTERN"]
PLOT_MODULE = runpy.run_path(
    "scripts/misc/plot_gma_hpo_pareto.py", run_name="gma_plot"
)
PLOT_CONFIG_PATTERN = PLOT_MODULE["CONFIG_PATTERN"]
POLISH_MODULE = runpy.run_path(
    "scripts/misc/summarize_seam_gma_polish.py", run_name="seam_gma_summary"
)
POLISH_CONFIG_PATTERN = POLISH_MODULE["CONFIG_PATTERN"]


def test_parses_scientific_notation_learning_rates():
    defence = CONFIG_PATTERN.fullmatch(
        "llama2-7b-beavertail--align-fft-gma"
        "--lr5e-7--coeff-gma1.0--coeff-retain2.0"
    )
    attack = ATTACK_PATTERN.search(
        "llama2-7b-beavertail--align-fft-gma"
        "--lr5e-7--coeff-gma1.0--coeff-retain2.0"
        "--relearn-fft-lr2e-4-ms5"
    )

    assert defence is not None
    assert defence["learning_rate"] == "5e-7"
    assert PLOT_CONFIG_PATTERN.fullmatch(defence.string)["learning_rate"] == "5e-7"
    assert attack is not None
    assert attack["learning_rate"] == "2e-4"


def test_parses_seam_gma_polish_configuration():
    match = POLISH_CONFIG_PATTERN.fullmatch(
        "llama2-7b-beavertail--align-fft-seam-gma-polish"
        "--lr5e-7--coeff-gma5.0--coeff-retain1.0--ms100"
    )

    assert match is not None
    assert match["learning_rate"] == "5e-7"
    assert match["gma"] == "5.0"
    assert match["max_steps"] == "100"
