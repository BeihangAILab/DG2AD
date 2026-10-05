# Benchmark data

The anonymous review artifact does not redistribute third-party benchmark
files. It contains only a small synthetic FSSP fixture for the offline smoke
test. Obtain each benchmark from its original publisher, check the current
terms, and place it below `dataset.root` (default: `data/`).

`data/catalog.json` describes and hashes the 12 default training groups.
Import or verify one of those groups with:

```text
python scripts/prepare_data.py --problem fssp --group tai20_5 --source <source-directory>
python scripts/prepare_data.py --problem fssp --group tai20_5 --verify-only
python scripts/prepare_data.py --all-defaults --verify-only
```

These commands make no network request. A hash mismatch is a hard failure.
The broader Appendix E collection is documented below for manual preparation;
it is not represented by a complete downloadable or hashed catalog.

## Appendix E scope

| Problem | Included groups below `data/<problem>/` | Unique instances | Size |
| --- | --- | ---: | --- |
| 3D-BPP | `MPV/` groups `n10,n20,n30,n40,n50,n60,n100,n150,n200` | 720 | 10--200 items |
| 3D-CLP | `3dclp_instances/` groups BR1--BR15 | 150 | 95--476 items |
| CVRP | `A/`, `B/`, `CMT/`, `E/`, `F/`, `Golden/`, `M/`, `P/`, `tai/` | 142 | 12--483 customers |
| FJSP | `barnes/`, `behnke/`, `brandimarte/`, `dauzere/`, `fattahi/`, `hurink/`, `kacem/` | 336 | 2--100 jobs, 2--60 machines |
| FSSP | Taillard 20-, 50-, and 100-job groups plus `tai200_10` and `tai200_20` | 110 | 20x5--200x20 |
| JSP | `abz`, `ft`, `la`, `orb`, `swv`, `ta`, `yn` entries in `jsp_instances.json` | 162 | 6x5--100x20 |
| Max-Cut | G-set instances selected by `G-set_le2000` | 41 | 800--2,000 vertices |
| MIS | BHOSLIB and DIMACS instances selected by `BHOSLIB_DIMACS_le2000` | 112 | 28--2,000 vertices |
| OSSP | Taillard `4x4`, `5x5`, `7x7`, `10x10`, `15x15`, `20x20` entries | 60 | 4x4--20x20 |
| RCPSP | PSPLIB `j30.sm/` | 480 | 30 real activities |
| SALBP-1 | `small data set_n=20/` | 525 | 20 tasks |
| TSP | `tsplib/` collection | 35 | 51--299 cities |

Metadata, BKS files, solution files, README files, and alternative encodings of
the same source instance are not counted as instances. Each benchmark group is
split independently after stable-ID sorting, using `split_seed=0` and a
60/20/20 train/validation/test ratio.

## Fixed exclusions and optional filtered groups

The Appendix E scope excludes CVRP `AGS`, FSSP `tai500_20`, and SALBP-1
`very large data set_n=1000`.

For Max-Cut, place the G-set `G*.txt` files and `gset_bks.json` directly in
`data/max_cut/`, then use:

```text
problem=max_cut problem.target_group=G-set_le2000
```

This group excludes `G50`, `G55`, `G60`, `G65`, `G70`, `G72`, and `G81` and
contains 41 instances in the documented collection.

For MIS, place `.clq` files under `data/mis/BHOSLIB/`,
`data/mis/DIMACS_all/`, and `data/mis/DIMACS_subset/`, with
`data/mis/mis_bks.json`, then use:

```text
problem=mis problem.target_group=BHOSLIB_DIMACS_le2000
```

The loader deduplicates case-insensitive instance stems across the three
directories and excludes `C4000.5`, `MANN_a81`, and `keller6`, yielding 112
instances in the documented collection.

The original benchmark names, papers, and source pages for the 12 default
groups are recorded in `data/catalog.json`. This document does not grant
redistribution permission for any third-party data.
