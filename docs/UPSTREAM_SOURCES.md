# Upstream source references

The HBM formulas and version-specific adapters in this repository were
validated against the following upstream revisions:

| Project | Release | Commit | Repository |
|---|---|---|---|
| vLLM Ascend | `v0.23.0rc1` | `f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db` | <https://github.com/vllm-project/vllm-ascend.git> |
| vLLM | `v0.23.0` | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` | <https://github.com/vllm-project/vllm.git> |

Large source snapshots are intentionally not committed. To inspect the exact
versions locally, clone them outside this repository:

```powershell
git clone --branch v0.23.0rc1 --depth 1 https://github.com/vllm-project/vllm-ascend.git
git clone --branch v0.23.0 --depth 1 https://github.com/vllm-project/vllm.git
```

When an adapter is updated for a new upstream release, add its release and
full commit hash to this table so model results remain reproducible.
