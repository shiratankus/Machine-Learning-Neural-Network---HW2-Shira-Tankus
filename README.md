## General instructions

The repository contains:
- This README.md file
- Homework notebook for the second exercise, parts 1-3,5,6: HW_2.ipynb
- Separate file for part 4: part4_augmentation.py
- Plots from part 4's output: augmented_datasets.png

## Part 4 – Computational Optimization

Due to the computational cost of performing 100,000 data augmentation iterations, several implementations were investigated to reduce execution time.

The final implementation uses NumPy, Scikit-learn and Joblib, combining batch processing with parallel execution across 16 CPU workers. 
Each batch is assigned a separate child seed to ensure reproducible and independent random number generation.

The calculation was executed on the university's computing server and completed in approximately 7.1 minutes.

### Execution

Install the required dependencies:

```bash
pip install numpy matplotlib scikit-learn joblib
```

Run the standalone script:

```bash
python part4_augmentation.py
```

The number of parallel workers can be adjusted using the `nprocs` parameter according to the available computational resources.


