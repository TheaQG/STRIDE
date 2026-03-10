"""
Responsibilities:
- save generated arrays
- save ensemble members
- save PMM outputs
- save conditioning data used for generation
- save generation metadata (e.g. config, generation payload, PMM payload) JSON/YAML
- create output directory structure
- maybe save a few preview plots (e.g. generated vs. obs for a few cases)

Suggested structure:
runs/train_edm_small/
    generation/
        test_best/
            metadata.json
            samples/
                19911009/
                    member_0000.npz
                    member_0001.npz
                    ...
                    pmm.npz
                    ensemble_mean.npz
            figures/
                19911009_preview.png
(Maybe flatter)
Should DEFINITELY contain conditioning data used

Metadata to save (key for reproducibility, debugging, and interpretability):
- checkpoint used (path, maybe also training step/epoch)
- checkpoint mode
- split used
- ensemble size
- generation + model + dataset + training config paths
- run timestamp
- variable names
- whether outputs are model-space or physical-space (i.e. whether transforms were applied)
"""