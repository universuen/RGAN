from pathlib import Path

src: Path = Path(__file__).absolute().parent.parent
project: Path = src.parent
data: Path = project / 'data'
scripts: Path = project / 'scripts'
tests: Path = project / 'tests'
plots: Path = project / 'plots'
datasets: Path = data / 'datasets'
raw_datasets: Path = datasets / 'raw'
processed_datasets: Path = datasets / 'processed'

for i in list(vars().values()):
    if isinstance(i, Path):
        i.mkdir(parents=True, exist_ok=True)
