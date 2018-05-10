import math
import random
import numpy

# turn csv into table skipping title line
#function from hw4
def csvToTableSkipTitle(fileName):
    file = open(fileName, "r")
    rows = filter(None, file.read().split("\r"))
    file.close()
    
    table = []
    for i in range(1, len(rows)):
        splitRow = rows[i].split(",")
        table.append(splitRow)
    return table

def main():
    table = csvToTableSkipTitle("Employee_Salaries_2015.csv")

    d6 = []
    a7 = []
    p8 = []

    for row in table:
        if row[6] not in d6:
            d6.append(row[6])
        if row[8] not in a7:
            a7.append(row[8])
        if row[9] not in p8:
            p8.append(row[9])

    print d6
    print len(d6)
    print a7
    print len(a7)
    print p8
    print len(p8)

if __name__ == '__main__':
    main()
