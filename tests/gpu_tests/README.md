# GPU Tests

## Environment for alpha-beta-crown

Remove old environment if necessary:
```bash
conda deactivate; conda env remove --name alpha-beta-crown
```

Create a new alpha-beta-crown environment:
```bash
conda env create -f complete_verifier/environment.yaml
conda activate alpha-beta-crown
```

Get a Gurobi license. For academic users, see [Academic License Registration](https://www.gurobi.com/downloads/end-user-license-agreement-academic/).

If you need to run GCP-CROWN tests, CPLEX is needed, see
[VNN-COMP benchmark instructions](../../complete_verifier/docs/vnn_comp.md#installation).

Download VNNCOMP benchmarks at the parent folder of the verifier:
```bash
git clone https://github.com/stanleybak/vnncomp2021
git clone https://github.com/ChristopherBrix/vnncomp2022_benchmarks
git clone https://github.com/ChristopherBrix/vnncomp2023_benchmarks
(cd vnncomp2022_benchmarks && bash setup.sh)
(cd vnncomp2023_benchmarks && bash setup.sh)
```

Then, navigate back to this directory (`tests/gpu_tests`).

## Run Testcases

```bash
python test.py -s TESTSET --run
```

* `TESTSET` can be chosen from: `vnncomp21`, `vnncomp22`, `vnncomp23`,
`beta_crown`, `gcp_crown`.

You may also choose to run a single benchmark in a test set, e.g.:
```bash
python test.py -s vnncomp/acasxu --run
```

## Check Results

```bash
python test.py -s TESTSET
```

Options:
* `--reference`: Specify a different reference (by default the latest will be taken)
* `--ignore-time`: Ignore comparison on time cost
* `--ignore-decision`: Ignore comparison on branching decisions

## Save a Reference

```bash
python test.py -s TESTSET --install-ref REF_NAME
```

* `REF_NAME` may look like `20211105_41a3`.
