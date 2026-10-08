# Benchmark data

This repository provides the DGA2D code, experiment configurations and a data
catalog. Third-party benchmark instances are not bundled. Obtain data from
the original publishers and place it under `dataset.root` (default: `data/`).
Source citations identify the datasets; a paper DOI is not a download URL.
Consult the publisher's distribution terms before sharing benchmark files.

## Default groups and verified file layouts

[The catalog](./data/catalog.json) records 12 default groups, required file
patterns, byte counts and SHA-256 tree digests. These describe the exact
prepared files expected by this version of the code, including converted
JSON and reference-value files where required. File counts in the catalog
are not necessarily instance counts. The default group for each problem is
selected in `cfg/problem/`; the default problem in `cfg/config.yaml` is 3D-CLP.

| Problem | Default group | Required files below `data/<problem>/` | Source / citation |
| --- | --- | --- | --- |
| 3dbbp | `MPV_n10` | `MPV/instance_n10_*.txt`, `3dbpp_instances_db.json` | [Martello, Pisinger, and Vigo (2000), The Three-Dimensional Bin Packing Problem.](https://doi.org/10.1287/opre.48.2.256.12386) |
| 3dclp | `BR` | `3dclp_instances/*.json`, `3dclp_bks.json` | [Bischoff and Ratcliff (1995), Issues in the development of approaches to container loading.](https://doi.org/10.1016/0377-2217(94)00015-G) |
| cvrp | `CMT` | `CMT/*` | [Christofides, Mingozzi, and Toth CVRP instances, distributed through CVRPLIB.](http://vrp.atd-lab.inf.puc-rio.br/index.php/en/) |
| fjsp | `brandimarte` | `brandimarte/*.txt`, `instances.json` | [Brandimarte (1993), Routing and scheduling in a flexible job shop by tabu search.](https://doi.org/10.1016/0925-5273(93)90048-M) |
| fssp | `tai20_5` | `fssp_instances.json` | [Taillard (1993), Benchmarks for basic scheduling problems.](https://doi.org/10.1016/0377-2217(93)90182-M) |
| jsp | `ta` | `jsp_instances.json` | [Taillard (1993), Benchmarks for basic scheduling problems.](https://doi.org/10.1016/0377-2217(93)90182-M) |
| max_cut | `G-set` | `G*.txt`, `gset_bks.json` | [G-set weighted Max-Cut benchmark instances.](https://web.stanford.edu/~yyye/yyye/Gset/) |
| mis | `DIMACS_subset` | `DIMACS_subset/*.clq`, `mis_bks.json` | [Second DIMACS Challenge maximum clique instances, evaluated through graph complementation.](https://dimacs.rutgers.edu/programs/challenge/) |
| ossp | `tai10_10` | `ossp_instances.json` | [Taillard (1993), Benchmarks for basic scheduling problems.](https://doi.org/10.1016/0377-2217(93)90182-M) |
| rcpsp | `j30` | `j30.sm/*.sm`, `j30_sm_merged.json`, `rcpsp_bks.json` | [PSPLIB single-mode j30 RCPSP instances.](https://www.om-db.wi.tum.de/psplib/) |
| salbp | `small` | `small data set_n=20/*.alb` | [Scholl (1993) SALBP benchmark data sets.](https://assembly-line-balancing.de/salbp/benchmark-data-sets-1993/) |
| tsp | `tsplib` | `tsplib/*.tsp`, `tsp_bks.json` | [Reinelt (1991), TSPLIB: A traveling salesman problem library.](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) |

After installing the environment described in [README.md](./README.md),
import an already prepared source directory and verify it:

```bash
uv run --locked --extra cu126 python scripts/prepare_data.py --problem fssp --group tai20_5 --source "/path/to/benchmarks/fssp"
uv run --locked --extra cu126 python scripts/prepare_data.py --problem fssp --group tai20_5 --verify-only
uv run --locked --extra cu126 python scripts/prepare_data.py --all-defaults --verify-only
uv run --locked --extra cu126 python main.py problem=fssp problem.target_group=tai20_5
```

The source directory must already have the relative layout shown above.
`prepare_data.py` copies and verifies files; it does not download benchmarks
or convert publisher formats. The script itself makes no network requests
(uv may need network access to install dependencies). A hash mismatch is a
hard failure: check the source, file layout and conversion, rather than
changing the catalog digest just to accept different data.

### Converted formats

Some publishers distribute formats different from those consumed here.
There is no general conversion command in this repository. If you prepare
files yourself, follow the corresponding `src/problems/<problem>/domain_evaluator.py`:

- FSSP: `fssp_instances.json` is a list of entries with `name`, `group`,
  `processing_times` (machines by jobs), and reference bounds.
- JSP: `jsp_instances.json` is consumed by the JSP loader; entries contain `n_jobs`,
  `n_machines`, `machines_matrix` and `times_matrix` (jobs by operation order).
  Preserve zero-based machine IDs, reference bounds and group names.
- OSSP: `ossp_instances.json` maps group names to lists of entries containing
  `ossp_times` (jobs by machines) and reference bounds.
- 3D-CLP: per-instance JSON contains `Objects` container dimensions and `Items`
  with `Length`, `Height`, `Depth` and `Demand`; preserve the associated reference file.

The catalog hashes apply to the prepared corpus, not arbitrary equivalent
serializations of the publisher's files. Independently converted datasets
may be loadable but will not pass the existing exact-file verification;
record their provenance and hashes separately. Do not describe them as the
catalog-verified corpus.

## Documented experiment collection

The table below describes the collection associated with the repository's
Appendix E configuration notes. It is an expected layout, not bundled data or
a claim that every listed group has been imported and verified. The catalog
above only hashes default groups; the broader collection requires manual
preparation and separate provenance records. These counts should not be
interpreted as a new audit of the paper's experiments.

| Problem | Expected layout below `data/<problem>/` | Documented instances | Size |
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

Metadata, reference-value files, solution files and alternative encodings
of the same source instance are not counted as instances. Groups are split
after stable-ID sorting with `split_seed=0` and a 60/20/20 training/validation/test
ratio. The split manifest records the actual instances used by a run.

## Additional groups and reference values

The documented Appendix E selection excludes CVRP `AGS`, FSSP `tai500_20`
and SALBP `very large data set_n=1000`. Loader support for an additional group
does not establish that it was used in the paper or is covered by the catalog.

SALBP defaults to `small` (`small data set_n=20/`, 525 instances). The loader
also supports `large` (`large data set_n=100/`, 525 instances) and
`very_large` (`very large data set_n=1000/`). Only `small` is in the catalog.
After manually preparing and recording hashes for the complete large group:

```bash
uv run --locked --extra cu126 python main.py problem=salbp problem.target_group=large
```

For both SALBP groups, the current loader uses the theoretical workload lower
bound `ceil(sum(task_times) / cycle_time)`. The reported GAP is
`(stations - lower_bound) / lower_bound * 100%`; it is not a verified BKS gap.
Results on different instance sizes must be reported separately.

Max-Cut's optional `G-set_le2000` filter selects the documented 41-instance
collection, excluding `G50`, `G55`, `G60`, `G65`, `G70`, `G72` and `G81`.
Place `G*.txt` and `gset_bks.json` directly under `data/max_cut/` and select
`problem=max_cut problem.target_group=G-set_le2000`.

MIS's optional `BHOSLIB_DIMACS_le2000` filter reads `.clq` files from
`data/mis/BHOSLIB/`, `data/mis/DIMACS_all/` and `data/mis/DIMACS_subset/`, with
`data/mis/mis_bks.json`. It deduplicates case-insensitive stems and excludes
`C4000.5`, `MANN_a81` and `keller6`, yielding 112 instances in the documented
collection. Select `problem=mis problem.target_group=BHOSLIB_DIMACS_le2000`.

These optional filtered groups are not separately locked by the current
catalog. Preserve each dataset's original names, citations and reference
values when preparing data. This repository's code distribution does not
confer redistribution rights to third-party benchmarks.
