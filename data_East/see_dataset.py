import h5py

pth_ = "./EAST_database.h5"

dataset = h5py.File(pth_, 'r')
x = dataset["x"]
y = dataset["y"]
regi = dataset['regi']
posi = dataset['posi']