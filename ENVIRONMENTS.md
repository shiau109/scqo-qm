# Environments

Which Python environment to use lives in one place for all four repos, because the rule is a
**combo** property — the vendor-version half of it is literally cross-repo, and separate copies
disagree within one cycle:

**https://github.com/shiau109/SCQO/blob/main/ENVIRONMENTS.md**

The one line for this repo, so you cannot get it wrong by not clicking:

```bash
<parent>\.venv-qm\Scripts\python.exe -m pytest tests\ -q
```

**Never `uv run` here.** Its sync would rebuild the environment from `pyproject.toml` +
`uv.lock`, displacing `requirements-qm.lock.txt` — this repo's pin authority for the whole
`qm-qua → quam → qualibrate` stack, and the reason that lockfile exists.

**There is no repo-local venv for this repo.** A `scqo-qm/.venv` on disk is residue of a stray
`uv run`: it resolves from `pyproject.toml` rather than the lockfile, and cannot run the suite.
Build `.venv-qm` per [AGENTS.md](AGENTS.md) *Setup*, or INSTALL.md §1 on a lab machine.
