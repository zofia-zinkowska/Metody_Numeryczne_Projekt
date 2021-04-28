import matplotlib.pyplot as plt
import numpy as np

def student_indeks(indeks):
    return indeks % 100, int(indeks/1000)

def teoretyczne(t, T1, T2, D):
    nmax=10000
    u = np.zeros(11)
    x = np.linspace(0, 1, 11)
    for n in range(1, nmax + 1):
        if n % 2 == 0:
            u = u + 1/n * np.exp(- n**2 * np.pi**2 * D * t) * np.sin(n * np.pi * x)
        else:
            u = u - 1/n * np.exp(- n**2 * np.pi**2 * D * t) * np.sin(n * np.pi * x)

    u = u * 2*(T2 - T1)/ np.pi
    u += (T2 - T1) * x + T1
    return u

def main():
    indeks = 180925
    indeks1 = student_indeks(indeks)[0]
    indeks2 = student_indeks(indeks)[1]
    tolerance = 10**-4
    D = 0.00001
    z = 0.1 #skok x
    H = z * z / (2 * D)
    tablica = [[indeks1 for i in range(10)] + [indeks2]]
    error=1
    j=0
    while(error > tolerance):
        lista =[indeks1]
        for i in range(1,10):
            lista.append((tablica[j][i + 1] + tablica[j][i - 1])/2)
        lista.append(indeks2)
        tablica.append(lista)
        j+=1
        error = 0
        for i in range(11):
            error += abs((tablica[j][i] - tablica[j - 1][i]) / tablica[j][i])
        error /= 11
    np.savetxt('dane', tablica, fmt='%10.5f',delimiter='\t')
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(list(x for x in range(11)), list(range(j+1)))
    X = X*z
    Y = Y*H
    #ax.scatter(X, Y, tablica, color='red')
    teor = np.zeros((11, j + 1))
    for i in range(j + 1):
        teor[:, i] = teoretyczne(i * H, indeks1, indeks2, D)
    teor = np.transpose(teor)
    ax.scatter(X, Y, teor)
    plt.show()

    blad_wzgl = []
    for j in range(97):
        blad_wzgl_suma = 0
        for i in range(11):
            blad_wzgl_suma += abs((teor[j, i] - tablica[j][i]) / teor[j][i])
        blad_wzgl_suma /= 11
        blad_wzgl.append(np.log10(blad_wzgl_suma))

    plt.plot(list(j * H for j in range(j + 1)), blad_wzgl)
    plt.show()

    np.savetxt('Metoda Numeryczna', tablica, fmt='%10.5f', delimiter='\t')
    np.savetxt('Metoda Analityczna', teor, fmt='%10.5f', delimiter='\t')
if __name__ == '__main__':
    main()