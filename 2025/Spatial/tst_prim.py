from math import sqrt

def primfacs(n):
   i = 2
   primfac = []
   while i * i <= n:
       while n % i == 0:
           primfac.append(i)
           n = n / i
       i = i + 1
   if n > 1:
       primfac.append(n)
   return primfac

def tst_primfacs():
    for i in range(2,30):
        print(i,primfacs(i))

# разбить число на два множителя
# https://ru.stackoverflow.com/questions/769530/Разбить-число-на-два-множителя
def factoriz2(value):
    res = []
    for x in range(1, int(sqrt(value) + 1)):
        if not (value % x):
            res.append([x, value // x])
    return res  # 24 - [[1, 24], [2, 12], [3, 8], [4, 6]]

def factoriz2q(value2):
    """
    разбить на 2 множителя, выбрать наиболее близкие друг к другу
    """
    if value2 % 2 == 0: value = value2
    else: value = value2 + 1

    res = []
    for x in range(1, int(sqrt(value) + 1)):
        if not (value % x):
            res.append([x, value // x])
    ll=len(res)
    if ll == 1: return res[0][0], res[0][1], res
    mmin=[abs(res[i][0]-res[i][1]) for i in range(ll)]
    nmin=mmin.index(min(mmin))
    return res[nmin][0], res[nmin][1], res

def tst_factoriz2():
    print(factoriz2(24))  # 24 - [[1, 24], [2, 12], [3, 8], [4, 6]]
    for i in range(2,30):
        if i%2==0: n=i
        else: n=i+1
        print(i, n, factoriz2(n))

def tst_factoriz2q():
    for i in range(2,30):
        print(i, factoriz2q(i))



if __name__=="__main__":
    # tst_primfacs()
    # tst_factoriz2()
    tst_factoriz2q()