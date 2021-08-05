import os, sys
from file_preprocessing import read_csv
from model_pred import LabelHarassment


def main():
    file = sys.argv[1]
    _, fname = os.path.split(file)
    df = read_csv(file)
    LabelHarassment.init_from_df(df, fname)


if __name__ == '__main__':
    main()
