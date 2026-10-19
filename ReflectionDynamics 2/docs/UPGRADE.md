# v4 → v5

v5 adds `src/cavr/`, `scripts/v5/`, controller and hardware profiles, tests and the new runbook. The historical `refldyn` package remains for compatibility, but its defective string-likelihood API is blocked.

The ZIP contains one top-level `ReflectionDynamics/` code directory. In the existing repository, replace the matching code directory on a new Git branch. Do not replace the repository-root `results/` archive or a server run directory.

```bash
git switch -c cavr-v5
unzip -o /path/to/ReflectionDynamics_CAVR_v5.zip -d .
cd ReflectionDynamics
python -m pip install -e '.[train,test]'
python -m pytest
```

A clean repository root should contain:

```text
LICENSE
README.md
results/                         # existing historical archive
ReflectionDynamics/
    README.md                    # new v5 instructions
    src/cavr/
    src/refldyn/
    scripts/v5/
```

The importer needs the raw server state files and exported adapters. A checkout containing only the numerical GitHub result archive is not sufficient to train a value predictor. `rd5 legacy-audit` can still analyze that archive without loading a model.

The v4 source package remains available through repository history. Use `scripts/v5/` for the current study; the old probability estimator is not restored by rerunning an old script.

## Repository-root workflow

GitHub reads workflow files only from the repository-root `.github/workflows/` directory. After overlaying the nested code directory, install the included workflow there:

```bash
mkdir -p .github/workflows
cp ReflectionDynamics/.github/workflows/tests.yml .github/workflows/reflection-tests.yml
```

Run these commands from the repository root, not from the code subdirectory. The workflow detects both a nested and a flat code layout.
