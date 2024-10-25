# 2D Ising model Monte-Carlo at different temperatures
The temperature domain is split to carry out MC simulations of 2D Ising model in parallel. The MPI for python was used to parallelize computations. On Windows, you will need Microsoft MPI package.

## How to use
In command line type this, put a number of cores in <>, for example, 4
```shell
	mpiexec -n <number of cores> py ising_parallel.py
```
Parallelizing calculations at different temperatures might significantly reduce computational time. In my case, I got speedup ~3 with 4 cores.
### Settings

By default, simulation settings are:
```python
	N = 20 # N*N spins in total
	Nsteps = 250 #usually from 250 to 1000 is optimal
	N_eq = 25 #number of equilibrium steps 10:90 is OK

	temperatures = np.arange(1, 4, 0.02)
```
You can change them if you want. Do not put irrationably big number of simulations or small temperature step.
### Animation of heating
By default, animation is disabled. But if you want to make a .gif, you should change one line in the code from this:
```python
	make_animation = False
```
To this:
```python
	make_animation = True
```
It usually requires adjusting fps of animation, which is hard to define sometimes.
