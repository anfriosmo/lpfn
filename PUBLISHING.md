# Publishing LPFN 0.1.0

This repository is prepared for GitHub + PyPI Trusted Publishing.

## 1. Create the GitHub repository

Create a **public** repository named `lpfn`. Do not initialize it with a README,
license, or `.gitignore`, because those files already exist here.

After the repository exists, add its URLs to `pyproject.toml`:

```toml
[project.urls]
Homepage = "https://github.com/<OWNER>/lpfn"
Repository = "https://github.com/<OWNER>/lpfn"
Issues = "https://github.com/<OWNER>/lpfn/issues"
Documentation = "https://github.com/<OWNER>/lpfn/tree/main/docs"
```

Replace `<OWNER>` with the GitHub username or organization that owns the repo.

## 2. Push the release source

From the project directory:

```bash
git init
git add .
git commit -m "Release LPFN 0.1.0"
git branch -M main
git remote add origin https://github.com/<OWNER>/lpfn.git
git push -u origin main
```

The included `.github/workflows/ci.yml` runs tests on Python 3.10–3.13 and
validates the package build.

## 3. Prepare PyPI

Create/sign in to your PyPI account, verify your email, enable 2FA, and keep
recovery codes securely.

Because `lpfn` is a new project, configure a **pending Trusted Publisher** in
PyPI account settings. Use:

- PyPI project name: `lpfn`
- GitHub owner: `<OWNER>`
- GitHub repository: `lpfn`
- Workflow filename: `release.yml`
- Environment name: `pypi`

A pending publisher does not reserve the package name until the first successful
publish, so publish promptly after configuring it.

## 4. Configure the GitHub `pypi` environment

In GitHub repository settings, create an environment named `pypi`.
Requiring manual approval for this environment is recommended.

The release workflow has only the publish job with `id-token: write`; no PyPI
API token needs to be stored as a GitHub secret.

## 5. Optional: test on TestPyPI first

Create a TestPyPI account and configure its Trusted Publisher separately using:

- project: `lpfn`
- workflow filename: `testpypi.yml`
- environment: `testpypi`

Create the `testpypi` GitHub environment, then run the workflow manually from
GitHub Actions: **Publish to TestPyPI → Run workflow**.

Because PyTorch may not be available from TestPyPI, a practical install check is:

```bash
python -m pip install torch
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps lpfn==0.1.0
python -c "import lpfn; print(lpfn.__version__)"
```

## 6. Publish the real release

Once CI is green:

```bash
git tag -a v0.1.0 -m "LPFN 0.1.0"
git push origin v0.1.0
```

On GitHub, create a release for tag `v0.1.0` and click **Publish release**.
That event triggers `.github/workflows/release.yml`, which builds the wheel and
sdist, runs `twine check --strict`, and publishes through PyPI Trusted
Publishing.

## 7. Verify after release

In a clean environment:

```bash
python -m pip install lpfn
python -c "import lpfn; print(lpfn.__version__)"
```

Then run one of the examples from the repository.

## Important release rules

- PyPI release files are immutable: if `0.1.0` is published with a mistake,
  fix the source and publish a new version such as `0.1.1`; do not try to
  overwrite `0.1.0`.
- Keep large benchmark outputs and checkpoints outside the Python package.
- Update `CHANGELOG.md`, `CITATION.cff`, and `src/lpfn/_version.py` for every
  new release.
