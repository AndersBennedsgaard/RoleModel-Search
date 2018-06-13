import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animation(i, f, scatt):
    numatoms = 24
    struct = np.load('structures_test.npy')

    struct = struct[i * numatoms:(1 + i) * numatoms, :]
    # struct = struct[struct[:, 2].argsort(), :]

    energy = np.load('energies_test.npy')

    scatt[0].set_offsets(struct[:, 0:2])
    # scatt[0].set_color(c=struct[:, 2]/np.max(struct[:, 2]))
    scatt[1].set_data(list(range(len(energy[:i]))), energy[:i])
    return scatt


structures = np.load('structures_test.npy')
energies = np.load('energies_test.npy')

num_atoms = 24
x = structures[:num_atoms, 0]
y = structures[:num_atoms, 1]
z = structures[:num_atoms, 2]

num_frames = len(energies)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]})
scat = ax1.scatter(x, y, c=z, s=370, cmap='copper', edgecolor='black')
line, = ax2.plot_struct_clustering(0, energies[0])
scat_line = [scat, line]

fig.colorbar(scat, ax=ax1)
ax1.axis('equal')
ax1.set_title('Relaxing 24 carbon-atoms with Lennard-Jones (12,6)-potential')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid()

ax2.set_xlim(0, num_frames-1)
ax2.set_ylim(min(energies), max(energies))
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Energy')
ax2.grid()

anim = FuncAnimation(fig, animation, frames=num_frames, interval=30, fargs=(fig, scat_line), blit=True)

# anim.save('animation.mp4', dpi=100, fps=10, writer='imagemagick')

plt.show()
print("done!")
