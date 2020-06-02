"""
Author:      XuLin Yang
Student id:  904904
Date:        
Description: 
"""

from matplotlib import pyplot as plt


if __name__ == "__main__":
    x_count = {i: 0 for i in range(6)}
    x = [i for i in range(6)]
    y = [0 for _ in range(6)]
    z = [0 for _ in range(6)]

    with open("out.txt", "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if line.startswith("branching factor:  "):
                branching_factor = int(line[19:])
                # print(branching_factor)

                time_elapsed = float(f.readline()[14:])
                n_iter = int(f.readline()[7:])

                x_count[branching_factor-1] += 1
                y[branching_factor-1] += time_elapsed
                z[branching_factor-1] += n_iter
                # print(branching_factor, time_elapsed, n_iter)

        # print(x_count)
        # print(x)
        # print(y)
        # print(z)

        plt.figure()
        plt.xlabel("branching factor")
        plt.ylabel("time elapsed (s)")
        x_plot = [i+1 for i in x]
        y_plot = [i / x_count[b] for b, i in enumerate(y)]
        print(x_plot, y_plot)
        plt.plot(x_plot, y_plot, 'b-')
        plt.ylim(0.5, 1)
        plt.savefig("MCTS time use")
        plt.show()

        plt.figure()
        plt.xlabel("branching factor")
        plt.ylabel("avg iteration")
        x_plot = [i + 1 for i in x]
        y_plot = [i / x_count[b] for b, i in enumerate(z)]
        print(x_plot, y_plot)
        plt.plot(x_plot, y_plot, 'b-')
        # plt.ylim(0.5, 1)
        plt.savefig("MCTS avg iteration use")
        plt.show()
