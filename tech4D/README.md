# This directory contains the following:

- [`post_damage_post_tech.py`](./post_damage_post_tech.py): computes post damage post technological change 3D HJB with a given input of
$\gamma_3$
(description can be found in [write-ups](../write-ups/write-up.pdf) section 1.2.1 on page 3.)
- [`post_damage_pre_tech.py`](./post_damage_pre_tech.py): computes post damage pre technological change 4D HJB with a given input of 
$\gamma_3$
(description can be found in [write-ups](../write-ups/write-up.pdf) section 2 on page 4.)
- [`Results.ipynb`](./Results.ipynb): contains simulation code for two capital, endogenized R&D investment model
	- `simulate_post`: simulate trajectories for solutions from [`post_damage_post_tech.py`](./post_damage_post_tech.py)
	- `simulate_pre`: simulate trajectories for solutions from [`post_damage_pre_tech.py`](./post_damage_pre_tech.py)

