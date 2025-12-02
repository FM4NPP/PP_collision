# Publication Checklist

**Before publishing this repository**, update all placeholders and anonymized information.

## Files Requiring Updates

### 1. README.md
- [ ] Update `[Authors]` in citation
- [ ] Update `[Journal]` in citation
- [ ] Add real contact information
- [ ] Update funding acknowledgments
- [ ] Add actual project/paper URL

### 2. SETUP.md
- [ ] Update contact email/info at bottom
- [ ] Verify all paths are anonymized
- [ ] Add actual computing facility name (if appropriate)
- [ ] Update expected training times based on final results

### 3. CITATION.cff
- [ ] Fill in all author names and affiliations
- [ ] Add repository URL: `https://github.com/[organization]/FM4NPP_Public`
- [ ] Add paper DOI when available
- [ ] Add paper title, journal, volume, pages
- [ ] Set version number and release date
- [ ] Update project website URL

### 4. LICENSE
- [ ] Update copyright holder: `[Authors/Institution]`
- [ ] Verify license type (MIT, Apache, BSD, etc.)
- [ ] Add year of first publication

### 5. Config Files

#### `scripts/configs/mamba_pretrain.yaml`
- [ ] Update `data_root` example path (can stay generic)
- [ ] Update `stat_dir` example path
- [ ] Update `checkpoint_dir` example path
- [ ] Verify all parameters match final paper results

#### `scripts/configs/mamba_tracking.yaml`
- [ ] Update all path examples
- [ ] Update `pretrained_ckpt` path example
- [ ] Verify downstream parameters match paper

### 6. SLURM Scripts

#### `scripts/run/submit_mamba_pretrain.sh`
- [ ] Replace `YOUR_ACCOUNT` with example or remove line
- [ ] Update `PYTHON_BIN` path to generic example
- [ ] Add cluster-specific instructions in comments

#### `scripts/run/submit_downstream_mamba.sh`
- [ ] Replace `YOUR_ACCOUNT` with example or remove line
- [ ] Update `PYTHON_BIN` path to generic example

### 7. Source Code Files

Check all `.py` files for:
- [ ] No hardcoded personal paths
- [ ] No real names in comments
- [ ] No cluster-specific hostnames
- [ ] No account numbers or IDs
- [ ] No email addresses (except in designated places)

Files to check:
- [ ] `train/pretrain/nppmamba/train_multi_gpu_mamba1.py`
- [ ] `train/downstream/track_finding_trainer.py`
- [ ] `train/downstream/model.py`
- [ ] `fm4npp/models/mambagpt.py`
- [ ] `fm4npp/models/mamba2.py`
- [ ] `fm4npp/models/embed.py`
- [ ] `fm4npp/datasets/dataset.py`
- [ ] `fm4npp/utils.py`

### 8. Example Scripts
- [ ] `example_usage.py`: Update checkpoint paths to generic examples

### 9. Documentation

#### Add if needed:
- [ ] `CHANGELOG.md`: Version history
- [ ] `EXPERIMENTS.md`: Reproduction instructions for paper results
- [ ] `FAQ.md`: Frequently asked questions
- [ ] `docs/`: Extended documentation folder

### 10. Data

If releasing sample data:
- [ ] Create `data/sample/` with small example dataset
- [ ] Add data format documentation
- [ ] Include preprocessing scripts
- [ ] Add data license information

### 11. Model Checkpoints

If releasing pretrained models:
- [ ] Upload to repository or hosting service (Zenodo, Hugging Face, etc.)
- [ ] Add download instructions to README
- [ ] Document model versioning
- [ ] Include model cards with metadata

## Pre-Publication Testing

### Functionality Tests
- [ ] Test pretraining on small dataset
- [ ] Test downstream training with pretrained model
- [ ] Test downstream training from scratch
- [ ] Verify all scripts run without errors
- [ ] Test on fresh conda environment

### Documentation Tests
- [ ] Follow setup instructions on fresh system
- [ ] Verify all links work
- [ ] Check code examples run correctly
- [ ] Spell-check all documentation
- [ ] Verify citation format

### Code Quality
- [ ] Run linting (pylint, flake8, black)
- [ ] Check for TODO/FIXME comments
- [ ] Verify consistent code style
- [ ] Add docstrings to all public functions
- [ ] Type hints where appropriate

## Git Repository Setup

### Before First Commit
- [ ] Initialize git repository: `git init`
- [ ] Add all files: `git add .`
- [ ] Create initial commit: `git commit -m "Initial release"`
- [ ] Verify .gitignore works correctly

### GitHub/GitLab Setup
- [ ] Create repository on hosting service
- [ ] Add repository description
- [ ] Set repository topics/keywords
- [ ] Enable issues
- [ ] Add repository README preview
- [ ] Set up branch protection for main
- [ ] Add collaborators/maintainers

### Optional Features
- [ ] Set up GitHub Actions for CI/CD
- [ ] Add pre-commit hooks
- [ ] Set up automatic code formatting
- [ ] Add issue templates
- [ ] Add pull request template
- [ ] Set up automated testing

## Final Steps

### Pre-Release
1. [ ] Complete all items above
2. [ ] Run full test suite
3. [ ] Review all files one final time
4. [ ] Get co-author approval
5. [ ] Create release branch

### Release
1. [ ] Tag release: `git tag -a v1.0.0 -m "First public release"`
2. [ ] Push to repository: `git push origin main --tags`
3. [ ] Create GitHub release with notes
4. [ ] Archive on Zenodo (optional, gets DOI)
5. [ ] Update paper/arxiv with repository link

### Post-Release
1. [ ] Monitor issues and pull requests
2. [ ] Update documentation based on feedback
3. [ ] Maintain CHANGELOG for updates
4. [ ] Consider creating tutorial videos/notebooks

## Anonymization Verification

### Paths Anonymized
- ✅ `/global/u1/d/dpark1` → `/path/to`
- ✅ `/pscratch/sd/d/dpark1` → `/path/to/scratch`
- ✅ `/global/cfs/cdirs/m4722` → `/path/to/data`

### Accounts Anonymized
- ✅ `m4722` → `YOUR_ACCOUNT`
- ✅ `dpark1` → `USER` or removed

### Personal Info Removed
- ✅ No real email addresses (except contact placeholder)
- ✅ No real names in code/comments
- ✅ No institution-specific details in code
- ✅ No cluster hostnames (except in comments)

## Notes

- Keep this checklist in the repository until all items are completed
- Delete this file before final publication, or move to internal docs
- Consider keeping an internal version with real paths for team use
- Update version numbers consistently across all files

---

**Last Updated**: [Date]
**Reviewed By**: [Names]
**Status**: [Pre-publication / Ready for review / Published]
