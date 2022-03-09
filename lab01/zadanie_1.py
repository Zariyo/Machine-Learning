

def prime(liczba):
    if liczba//2 != 1:
        if liczba % (liczba//2) == 0:
            return False
        else:
            for i in range(liczba//2, 1, -1):
                if liczba % i == 0:
                    return False
    return True

print(prime(5))

def select_primes(lista):
    nowa = []
    for i in lista:
        if prime(i) is True:
            nowa.append(i)
    return nowa

print(select_primes([3,6,11,25,19]))