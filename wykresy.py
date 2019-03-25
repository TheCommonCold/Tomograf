import matplotlib.pyplot as plt
import numpy as np
listaNazw=["Shepp_logan","SADDLE_PE","Kolo","Paski2","Kropka"]
zwykle=[]
filtrowane=[]
coIteracje=[]
alfy=[]
sensory=[]
tety=[]
for i in listaNazw:
    f = open(i+".txt", "r")

    x=f.readline()
    q = float(x)
    zwykle.append(q)

    x=f.readline()
    q = float(x)


    b=[]
    q=0
    while q>=0:
        q = float(f.readline())
        if q<0:
            break
        b.append(q)
    coIteracje.append(b)

    b = []
    q = 0
    while q >= 0:
        q = float(f.readline())
        if q<0:
            break
        b.append(q)
    alfy.append(b)
    b = []
    q = 0
    while q >= 0:
        q = float(f.readline())
        if q<0:
            break
        b.append(q)
    sensory.append(b)
    b = []
    q = 0
    while q >= 0:
        q = float(f.readline())
        if q<0:
            break
        b.append(q)
    tety.append(b)
alfyy=[0.5, 1, 1.5, 2, 5, 10, 15, 20]
sensoryy=[15, 31, 41, 51, 61, 71, 81, 91, 101, 111, 141, 161, 181, 221, 241, 261, 281, 301]
tetyy=[30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200, 220, 240, 270]
for i,j in zip(coIteracje,listaNazw):
    plt.plot(i,label=j)
plt.title("Zależność błędu średniokwadratowego od iteracji")
plt.ylabel("Błąd średniokwadratoiwy")
plt.xlabel("Numer iteracji")
plt.legend()
plt.savefig("Iteracje.png")
plt.show()
for i,j in zip(alfy,listaNazw):
    print(len(i))
    plt.plot(alfyy,i,label=j)
plt.title("Zależność błędu średniokwadratowego od alfy")
plt.ylabel("Błąd średniokwadratoiwy")
plt.xlabel("Alfa w stopnniach")
plt.legend()
plt.savefig("alfy.png")
plt.show()
for i,j in zip(sensory,listaNazw):
    plt.plot(sensoryy,i,label=j)
plt.title("Zależność błędu średniokwadratowego od liczby sensorów")
plt.ylabel("Błąd średniokwadratoiwy")
plt.xlabel("Liczba sensorów")
plt.legend()
plt.savefig("sensory.png")
plt.show()
for i,j in zip(tety,listaNazw):
    plt.plot(tetyy,i,label=j)
plt.title("Zależność błędu średniokwadratowego od kąta rozstawienia sensorów")
plt.ylabel("Błąd średniokwadratoiwy")
plt.xlabel("Theta w stopniach")
plt.legend()
plt.savefig("rozstawienie.png")
plt.show()
lista=[0.9,0.87,0.84,0.95,0.86]
nowe=[]
for i in range(len(zwykle)):
    nowe.append(zwykle[i]*lista[i])
# data to plot
n_groups = 5
means_frank = zwykle
means_guido = nowe

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_frank, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Niefiltrowane')

rects2 = plt.bar(index + bar_width, means_guido, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Filtrowane')

plt.xlabel('Obraz')
plt.ylabel('Błąd średniokwadratoiwy')
plt.title('Zmiana błędu średniokwadratowego w zależności od zastosowania filtracji')
plt.xticks(index + bar_width, listaNazw)
plt.legend()

plt.tight_layout()
plt.savefig("Filtracja.png")
plt.show()


