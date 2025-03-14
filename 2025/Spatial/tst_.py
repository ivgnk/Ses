# for i in range(2):  print(i)
import numpy as np
# https://habr.com/ru/companies/ruvds/articles/482464/
def args_kwargs():
    a,b,c, = 1,11, 111
    print(f'{a=} {b=} {c=}\n')
    a,b,*c, = 1,11, 111, 1111
    print(f'{a=} {b=} {c=}\n')
    a = [1, 2, 3]
    b = [*a, 4, 5, 6]
    print(f'{a=} {b=}\n')

    #---  *args
    def printScores(student, *scores):
        print(f"Student Name: {student}")
        for score in scores:
            print(score)
    printScores("Jonathan", 100, 95, 88, 92, 99)
    print()

    #---  **kwargs
    def printPetNames(owner, **pets):
        print(f"Owner Name: {owner}")
        for pet, name in pets.items():
            print(f"{pet}: {name}")
    printPetNames("Jonathan", dog="Brock", fish=["Larry", "Curly", "Moe"], turtle="Shelldon")

if __name__=="__main__":
    print(np.linspace(0,50,51))
# args_kwargs()