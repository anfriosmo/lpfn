# LPFN 0.1.0 release checklist

## Before publishing
- [ ] Confirm `lpfn` is still available on PyPI/TestPyPI.
- [ ] Create the public GitHub repository.
- [ ] Add repository URLs to `pyproject.toml` once the final GitHub URL exists.
- [ ] Review the MIT license choice with all authors.
- [ ] Run the full CI matrix successfully.
- [ ] Build `sdist` and wheel and run `twine check --strict`.
- [ ] Test installation from the wheel in a clean environment.
- [ ] Create the `pypi` GitHub environment and optionally require approval.
- [ ] Configure the PyPI Trusted Publisher for `.github/workflows/release.yml`.
- [ ] Optionally configure TestPyPI Trusted Publisher for `.github/workflows/testpypi.yml`.

## Release
- [ ] Tag `v0.1.0`.
- [ ] Publish a GitHub Release from that tag.
- [ ] Confirm the `Publish release` workflow succeeds.
- [ ] Install from PyPI in a clean environment: `pip install lpfn`.
- [ ] Run the minimal example.

## After publishing
- [ ] Add the final PyPI and repository URLs to documentation if needed.
- [ ] Archive release artifacts and benchmark provenance.
