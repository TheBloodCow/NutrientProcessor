import numpy as np
import csv
def csv_loader(filename):
    try:
        with open(filename, 'r') as f:
            lines = csv.reader(f)
            y = list()
            stringTupleList = list()
            for line in lines:
                stringTupleList.append(line)
            return stringTupleList;
        if lines is None:
            print("no file.")
            return


    except Exception as e:
        print(e)
        print("Error occurred. Check if file exists.")

def main():
    coords = csv_loader("FoodNutrients.csv")
    print(coords[0])
    print(coords[0][0])
    print(coords)

if __name__ == "__main__":
    main()
