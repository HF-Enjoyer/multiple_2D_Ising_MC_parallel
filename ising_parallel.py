import time
from mpi4py import MPI
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation, rc
rc('animation', html='jshtml')

# ----------------- FUNCTIONS ------------------

def mc(config, beta): # for configuration at given beta
    for j in range(N):
        for i in range(N):
            #select a random spin from NxN system  
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            s =  config[a, b]
            # calculate energy cost of this new configuration (% is for pbc)
            Delta_H = 2*s*(config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N])
            # flip spin or not depending on the Delta_H and thermal fluctuation probability
            if Delta_H < 0:
                s *= -1
            elif np.random.rand() < np.exp(-Delta_H*beta):
                s *= -1
            config[a, b] = s
    return config

def calc_energy(config):
    #calculate energy H (of 1 configuration)
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S_ij = config[i,j]
            H_ij = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            energy += -H_ij*S_ij
    return energy/4.

def mc_res(N_eq, Nsteps, state, T):
    beta = 1./T
    states_history = [state.copy()]
    mag_history = [np.sum(state.copy())/(N**2)]
    energy_history = [calc_energy(state.copy())]
    en2_history = [calc_energy(state.copy())**2]

    for i in range(Nsteps):
        state = mc(state, beta)
        states_history.append(state.copy())
        mag_history.append(np.sum(state)/(N**2))
        energy_history.append(calc_energy(state.copy()))
        en2_history.append(calc_energy(state.copy())**2)
    # Calculate averages of magnetization and energy after equilibrium is reached
    mag_ave = np.abs(np.sum(mag_history[-N_eq:])/N_eq)
    energy_ave = np.sum(energy_history[-N_eq:])/(N_eq*N**2)
    en2_ave = np.sum(en2_history[-N_eq:])/(N_eq*N**4)

    return states_history, round(mag_ave, 5), round(energy_ave, 6), round(en2_ave, 7)

def create_animation(state, history, temp_range):
    fig = plt.figure()
    im = plt.imshow(state+1, vmin=0, vmax=1, cmap='Greys_r',
                        interpolation='none', animated=True)
    title = plt.title('')

    def animate(i):
        im.set_array(history[i])
        title.set_text(f'Temperature: {round(temp_range[i], 2)}')

    anim = animation.FuncAnimation(fig, animate, len(history))
    writer = animation.PillowWriter(fps=15) # its sometimes difficult to choose what fps is optimal
    anim.save('anim_ising.gif', writer=writer)

# ---------------- FUNCTIONS END ---------------

if __name__ == "__main__":
    
    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 20 # N*N particles
    Nsteps = 250 #usually from 250 to 1000 is optimal
    N_eq = 25 #number of equilibrium steps 10:90 is OK

    Temp = 1.
    beta = 1./Temp # /kBT, kB = 1, T = element from Temp_range massive.
    state = 2*np.random.randint(2, size=(N,N))-1 #initial state
    make_animation = True # Leave False if you dont need an animation

    if rank == 0:
        # temperature range
        temperatures = np.arange(1, 4, 0.02)
        start_time = time.time() #just to compare time performance
        temps_per_proc = np.array_split(temperatures, size)
    else:
        temps_per_proc = None

    local_temperatures = comm.scatter(temps_per_proc, root=0)
    # results of each processor
    local_results = []
    local_states = []
    for i in local_temperatures:
        result = mc_res(N_eq, Nsteps, state, i)
        local_results.append((i, result[1], result[2], result[3])) 
        local_states.append((i, result[0][-1])) # result[0][-1] - last state of a system at given T

    # collect all results on 0 proc
    gathered_results = comm.gather(local_results, root=0)
    gathered_states = comm.gather(local_states, root=0)

    if rank == 0:
        # flatten the list of lists into a single list of tuples (temperature, energy)
        final_results = [item for sublist in gathered_results for item in sublist]
        final_states = [item for sublist in gathered_states for item in sublist]

        final_results.sort(key=lambda x: x[0]) # sorts results by temperature
        final_states.sort(key=lambda x: x[0])

        # write results to a text file
        with open("mc_ising.txt", "w") as f:
            f.write("Temp\tMag\tEn\tEn2\n")
            for temp, mag, en, en2 in final_results:
                f.write(f"{round(temp, 5)}\t{mag}\t{en}\t{en2}\n")
        print(f"MC done after: {round((time.time()-start_time), 4)} seconds")

        # make animation
        if make_animation == True:
            all_states = [] 
            for s in final_states:
                all_states.append(s[1])
            print('Making animation...')
            create_animation(all_states[0], all_states, temperatures)
            print('Animation saved!')
