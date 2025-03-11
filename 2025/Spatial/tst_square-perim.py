"""
Эллипс: площадь и длина
Формула площади эллипса
S = πab, где a и b — большая и малая полуоси эллипса.
Каноническое уравнение эллипса на координатной плоскости:
x²/a² + y²/b² = 1.
Приблизительная формула длины периметра эллипса (L) с
максимальной погрешностью около 0,63%:  5
L ≈ 4 πab + (a - b)2 / (a + b)

# https://ru.onlinemschool.com/math/formula/ellipse/
"""
from math import pi
def ellips_sq_per():
    b=1.0 # малая полуось эллипса
    for i in range(1,301,10): # большая полуось эллипса
        a=i*1.0
        s=pi*a*b
        top_=s+(a-b)**2
        bot_=a+b
        p=4*top_/bot_
        print(f'a={a:5.1f}   b={b:5.1f}   s={s:9.2f}    p={p:9.2f}  s/p={s/p:8.4f}   p/s={p/s:8.4f}')

if __name__=='__main__':
    ellips_sq_per()