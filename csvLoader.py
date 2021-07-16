import numpy as np
import csv
def csv_loader(filename):
    try:
        with open(filename, 'r') as f:
            lines = csv.reader(f)
            # print(reader)
            # for line in reader:
            # lines = f.read().splitlines()
            # print(lines)x = list()
            y = list()
            stringTupleList = list()
            #male and female are the classifiers
            for line in lines:
                # line = line.replace("'", " ")
                # line = line.replace(",.,", ",,")
                # line = line.replace("FEMALE", "1")
                # line = line.replace("MALE", "0")
                # line = line.replace("Yes", "1")
                # line = line.replace("No", "0")
                # line = line.split(',',-1)
                stringTupleList.append(line)
                # print(restaurant.get_id(),',',restaurant.get_name(),',',restaurant.get_address(),',',restaurant.get_city())
            # coords = np.array((x,y)).T
            # print(stringTupleList)
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
