import numpy as np
import argparse
import os

def generate_houses(out_dir):
    # first, make a generic cube
    X, Y, Z = 10, 10, 7
    b = 5

    house = np.zeros((Z, X, Y, 2))

    # 4 walls
    house[:,0,:,0] = b
    house[:,:,0,0] = b
    house[:,X-1,:,0] = b
    house[:,:,Y-1,0] = b

    # roof
    house[Z-1,:,:,0] = b
    
    # optionally, ablate one object type
    # roof

    fname = os.path.join(out_dir, "house.npy")
    np.save(fname, house)



if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Programmatically generate houses"
    )
    parser.add_argument(
        "--out_dir", type=str, help="location to store npy houses"
    )
    args = parser.parse_args()

    generate_houses(args.out_dir)

