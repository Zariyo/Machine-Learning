import random
import math

wektor1 = [3, 8, 9, 10, 12]
wektor2 = [8, 7, 7, 5, 6]
wektor3 = []
suma = []
iloczyn = []
dlugosc1 = 0
dlugosc2 = 0
skalarny=0
for i in range(0, len(wektor1)):
    suma.append(wektor1[i] + wektor2[i])
    iloczyn.append(wektor1[i] * wektor2[i])
    dlugosc1 += wektor1[i]**2
    dlugosc2 += wektor2[i]**2

dlugosc1 = dlugosc1**(1/2)
dlugosc2 = dlugosc2**(1/2)

print("a)")
print(suma)
print(iloczyn)
skalarny = sum(iloczyn)
print("b)")
print(skalarny)
print("c)")
print(dlugosc1)
print(dlugosc2)

for i in range(0, 50):
    wektor3.append(random.randint(1, 100))

srednia3 = sum(wektor3)/50
odchylenie = 0
wektor3norm = []

for i in range(0, 50):
    odchylenie += (wektor3[i] - srednia3)**2
    wektor3norm.append((wektor3[i]-min(wektor3))/(max(wektor3)-min(wektor3)))

odchylenie = (odchylenie/50)**(1/2)

print("d)")
print(wektor3)
print("e)")
print(srednia3)
print(min(wektor3))
print(max(wektor3))
print(odchylenie)
print("f)")
print(wektor3norm)
print(max(wektor3norm))

wektor3stand = []

for i in range(0, 50):
    wektor3stand.append((wektor3[i]-srednia3)/odchylenie)

srednia3stand = sum(wektor3stand)/50
print("g)")
print(wektor3stand)
print(srednia3stand)
odchylenie3stand = 0

for i in range(0, 50):
    odchylenie3stand += (wektor3stand[i] - srednia3stand)**2

odchylenie3stand = (odchylenie3stand/50)**(1/2)
print(odchylenie3stand)

print("h)")
wektor3dysk = []
for i in wektor3:
    j=0
    if i<100:
        j=(91,100)
    if i<90:
        j=(81,90)
    if i<80:
        j=(71,80)
    if i<70:
        j=(61,70)
    if i<60:
        j=(51,60)
    if i<50:
        j=(41,50)
    if i<40:
        j=(31,40)
    if i<30:
        j=(21,30)
    if i<20:
        j=(11,20)
    if i<10:
        j=(1,10)
    wektor3dysk.append(j)

print(wektor3dysk)
