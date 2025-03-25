"""
Вод данных из расширенного mid-файла в dataclass
"""

# Import in data class
def read1(fn_mid:str):
    from dataclasses import dataclass
    import csv

    @dataclass
    class the_mid:
        Labels: str
        Perimetr: float
        Square: float
        CentrX: float
        CentrY: float
        Area: float

    type T_mid_lst = list[the_mid]
    lst: T_mid_lst=[]

    with open(fn_mid, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for i,r in enumerate(reader):
            tm=the_mid(r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
            lst.append(tm)
    print(lst)

if __name__=="__main__":
    # read1('Str_H_2k_13-11-13_w.MID')
    pass