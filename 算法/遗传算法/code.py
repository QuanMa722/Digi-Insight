import numpy as np
from numpy.ma import cos
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


DNA_SIZE = 24  # 编码长度
POP_SIZE = 100  # 种群大小
CROSS_RATE = 0.5  # 交叉率
MUTA_RATE = 0.15  # 变异率
Iterations = 100  # 迭代次数
X_BOUND = [0, 10]  # X区间
Y_BOUND = [0, 10]  # Y区间


def F(x, y):
    return (6.452 * (x + 0.125 * y) * (cos(x) - cos(2 * y)) ** 2) / (0.8 + (x - 4.2) ** 2 + 2 * (y - 7) ** 2) + 3.226 * y


def getfitness(pop):
    x, y = decodeDNA(pop)
    temp = F(x, y)
    return (temp - np.min(temp)) + 0.0001


def decodeDNA(pop):
    x_pop = pop[:, 1::2]
    y_pop = pop[:, ::2]
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y


def select(pop, fitness):
    temp = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness / (fitness.sum()))
    return pop[temp]


def mutation(temp, MUTA_RATE):
    if np.random.rand() < MUTA_RATE:
        mutate_point = np.random.randint(0, DNA_SIZE)
        temp[mutate_point] = temp[mutate_point] ^ 1


def crossmuta(pop, CROSS_RATE):
    new_pop = []
    for i in pop:
        temp = i
        if np.random.rand() < CROSS_RATE:
            j = pop[np.random.randint(POP_SIZE)]
            cpoints1 = np.random.randint(0, DNA_SIZE * 2 - 1)
            cpoints2 = np.random.randint(cpoints1, DNA_SIZE * 2)
            temp[cpoints1:cpoints2] = j[cpoints1:cpoints2]
            mutation(temp, MUTA_RATE)
        new_pop.append(temp)
    return new_pop


def print_info(pop):
    fitness = getfitness(pop)
    maxfitness = np.argmax(fitness)
    x, y = decodeDNA(pop)

    print("-" * 30)
    print(f"x = {x[maxfitness]}, y = {y[maxfitness]}")
    print(f"F(x,y) = {F(x[maxfitness], y[maxfitness])}")
    print("-" * 30)


def plot_3d(ax):  # 建模
    X = np.linspace(*X_BOUND, 100)
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y)
    Z = F(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_zlim(-20, 160)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(0.5)
    plt.show()


if __name__ == "__main__":
    fig = plt.figure()
    ax = Axes3D(fig)
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    plt.ion()
    plot_3d(ax)
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))

    for i in range(Iterations):
        print("-" * i)
        x, y = decodeDNA(pop)
        if 'sca' in locals():
            sca.remove()
        sca = ax.scatter(x, y, F(x, y), c="black", marker='o')
        plt.show()
        plt.pause(0.1)
        pop = np.array(crossmuta(pop, CROSS_RATE))
        fitness = getfitness(pop)
        pop = select(pop, fitness)

    print_info(pop)
    plt.ioff()
    plot_3d(ax)
