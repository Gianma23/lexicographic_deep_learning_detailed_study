# Pre-projection backup

This folder stores the pre-projection versions of files changed for semantic projection.

## Files
- cast_deit_hier.py
- graph_pool.py
- modules.py
- factory.py

## Restore (PowerShell)
Run from repo root:

```powershell
powershell -ExecutionPolicy Bypass -File models/hcast/internal_pre_projection_backup/restore_pre_projection.ps1
```

This will overwrite current files with the backed-up versions.
