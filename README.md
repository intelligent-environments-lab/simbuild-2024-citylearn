# SimBuild 2024: Applications in CityLearn v2 Gym environment for multi-objective control benchmarking in grid-interactive buildings and communities

To reproduce simulations, execute the following:

```console
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python -m src.simulate run_work_order workflow/simulation.sh
```

Finally, run the [results.ipynb](notebooks/results.ipynb) notebook to generate the figures in the paper.