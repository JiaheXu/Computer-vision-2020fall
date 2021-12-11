import numpy as np
from TKFactorization import TKFactorization
from TKTranslation import TKTranslation
from TKCenterData import TKCenterData
from TKDisplay3D import TKDisplay3D


def main():
    matrix_path = 'data/measurement_matrix.txt'
    with open(matrix_path, 'r') as f:
        lines = f.readlines()
        # m = len(lines)/2
        matrix_data = []
        for line in lines:
            line_data = line.split(' ')
            # n = len(line_data)
            matrix_data.append(line_data)
    W = np.asarray(matrix_data)
    W = W.astype(np.float32)
    print("W shape: ", W.shape)
    print("W type: ", type(W))
    print("W data type: ", type(W[0, 0]))
    t = TKTranslation( W )
    centered_W = TKCenterData( W, t )
    M, X = TKFactorization(centered_W)
    TKDisplay3D(X)



if __name__ == "__main__":
    main()