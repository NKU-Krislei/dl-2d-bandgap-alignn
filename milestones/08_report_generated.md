# Milestone: Report Generated
**Created**: 2026-04-27 23:17
**Status**: COMPLETE

## What Was Just Done
Generated the final Markdown report and slide outline on the remote host, then downloaded the report, figures, results, and split CSVs to the local workspace.
The report reflects the fallback path: RF/Ridge classical baselines are complete, pretrained ALIGNN is unavailable in this run, and self-training is documented as blocked by ALIGNN/DGL/PyTorch compatibility.

## Current Project State
- Phase: Phase 3 report generation complete
- Key files modified: `report/report.md`, `report/slides_outline.md`, `figures/`, `results/`, `data/*_id_prop.csv`
- Remote state: Complete artefacts remain in `/root/dl_2d_bandgap/`
- Results: Local workspace now has updated evaluation/report artefacts

## Next Immediate Steps
1. Review final metrics and generated report locally.
2. Shut down the AutoDL instance to stop billing.
3. If future self-training is required, use a PyTorch/DGL/ALIGNN environment with matching CUDA wheels rather than this PyTorch 2.8 image.

## Key Commands That Work
```bash
python3 run_pipeline.py --skip-to 7
rsync -avz -e 'ssh -p 48992 -o StrictHostKeyChecking=no' root@connect.cqa1.seetacloud.com:/root/dl_2d_bandgap/results/ results/
rsync -avz -e 'ssh -p 48992 -o StrictHostKeyChecking=no' root@connect.cqa1.seetacloud.com:/root/dl_2d_bandgap/figures/ figures/
rsync -avz -e 'ssh -p 48992 -o StrictHostKeyChecking=no' root@connect.cqa1.seetacloud.com:/root/dl_2d_bandgap/report/ report/
```

## Lessons Learned / Gotchas
- The fallback path is complete and reportable.
- Do not leave the AutoDL GPU running after downloading artefacts.
