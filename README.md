# AquaFieldINR

Version 2.1.3.

## Installation

Python >=3.11, NumPy >=2.0, SciPy >=1.13, PyTorch >=2.2.

```text
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .
```

## Commands

```text
python -m aquafieldinr --help
python preprocessing/preprocess.py --help
python run_validation.py --output validation_output
```

## Tests

```text
python tests/field_streaming.py
python tests/field_sparse_review.py
python tests/sparse_exact_tests.py
python tests/sparse_large_test.py
python tests/training_alignment.py
python tests/audit_cross_module.py
python tests/test_correlated.py
python tests/test_likelihood_batching.py
python tests/gpu_core.py
python tests/gpu_runtime.py
```

## License

BSD 3-Clause. See `LICENSE`.
