# Slides Outline

1. Title slide
   Project title, author, course, date.
2. Motivation
   Why 2D materials matter and why band-gap prediction is useful.
3. Problem setting
   DFT accuracy versus high-throughput cost.
4. Dataset
   JARVIS source, filtering, and final split sizes.
5. ALIGNN architecture
   Crystal graph plus line graph.
6. Baseline models
   Random Forest and Ridge with Magpie descriptors.
7. Data exploration
   Distribution of band gaps and family composition.
8. Benchmark results
   MAE / RMSE / R² comparison showing self-trained ALIGNN outperforming Random Forest and Ridge.
9. Error analysis
   Self-trained ALIGNN scatter plot, residuals, and per-family behavior.
10. Training status
   Completed V100 CUDA run: 50 epochs, best validation MAE, and final test MAE.
11. Discussion
   Why graph and bond-angle information improves over composition-only baselines.
12. Conclusion
   Main finding: ALIGNN reaches 0.115 eV test MAE and beats the baseline target.
