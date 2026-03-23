# FR/CDR Sync File Changes

## File-level checklist

### Inference / deployment

- `IgGM/deploy/ab_design.py`
- `IgGM/deploy/base_designer.py`
- `design.py`

### Model / training files already touched across tasks

- `IgGM/model/arch/core/diffuser.py`
- `IgGM/model/arch/core/module/fr_rigid_head.py`
- `IgGM/model/arch/core/module/cdr_loop_head.py`
- `IgGM/model/arch/core/module/structure_module.py`
- `IgGM/model/arch/design_model/model.py`
- `src/iggm_lightning/losses.py`
- `src/iggm_lightning/lightning_module.py`
- `src/train_iggm_lightning.py`

### Data / metadata files already touched across tasks

- `IgGM/protein/antibody_regions.py`
- `data/prepare_data_fromzip.py`
- `src/iggm_lightning/data_module.py`

### Documentation

- `docs/fr_cdr_metadata_layer.md`
- `docs/fr_cdr_sync_diffuser.md`
- `docs/fr_cdr_structure_module.md`
- `docs/fr_cdr_design_model_integration.md`
- `docs/fr_cdr_sync_losses.md`
- `docs/fr_cdr_sync_inference.md`
- `docs/fr_cdr_sync_config.md`
- `docs/fr_cdr_sync_migration_notes.md`
- `docs/fr_cdr_sync_file_changes.md`
- `docs/fr_cdr_sync_known_assumptions.md`
