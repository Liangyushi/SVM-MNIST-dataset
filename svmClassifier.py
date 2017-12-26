import numpy as np
import cvxopt
import math
import os
import struct

def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise (ValueError, "dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), 1, 784)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()		
		
def rbf(x,y,gamma):
	#radial basis kernel
	return math.exp((-gamma)*(np.dot((x-y).T,(x-y))))
	

def classifiction(so,digit_data,x,w0,gamma):
	f = 0
	for i,entry in enumerate(so):
		f += so[i][0]*y[entry[1]]*rbf(digit_data[entry[1]][1][0],x,gamma)
	f = f + w0
	return f
		

	
num_entries = 50
num_entries_test = 10
num1 = 0
num2 = 1

gamma = 0.1 #gamma value for radial basis kernel

c = 2 #trade off constant for soft margin support vector machine

print('SVM Classifier for {} and {}'.format(num1,num2))

print('Training')

mnist_reader = read('training')


i = 0
j = 0
digit_data = []

while (i<num_entries or j<num_entries):
	#reading the data from training dataset and copying into python list
	image = next(mnist_reader)
	
	
	if(image[0] == num1 or image[0] == num2):
		
		digit_data.append(image)
		
		if(image[0]==num1):
			i+=1
		else:
			j+=1

n = len(digit_data)	
print ('Total no. of Training samples {}'.format(n))	
Q = np.empty(shape=[n,n])	
y = np.empty(shape=[n,1])	

for i,imageX in enumerate(digit_data):
	if imageX[0] == num1:
		y[i] = -1
	else:
		y[i] = 1
	

	

	
for i,imageX in enumerate(digit_data):
	for j,imageY in enumerate(digit_data):
		#creation of Q matrix of SVM
		Q[i][j] = y[i]*y[j]*rbf((imageX[1][0]),(imageY[1][0]),gamma)
		
	


#convex optimization using cvxopt library
G = np.zeros(shape=[2*n,n])
h = np.zeros(shape=[2*n,1])
for i in range(n):
	G[2*i][i] = -1
	G[(2*i)+1][i] = 1
	h[2*i] = 0
	h[(2*i)+1] = c
		


P = cvxopt.matrix(Q,tc='d')
q = cvxopt.matrix((-1)*np.ones(shape=[n,1]),tc='d')
g = cvxopt.matrix(G,tc='d')
H = cvxopt.matrix(h,tc='d')	
A = cvxopt.matrix(y.T,tc='d')
b = cvxopt.matrix(np.zeros(shape=[1,1]),tc='d')	


sol = cvxopt.solvers.qp(P,q,g,H,A,b,kktsolver='ldl', options={'kktreg':1e-9,'maxiters':100})

print(sol['status'])
solution = sol['x']


sv = []
so = []
for i,entry in enumerate(solution):
	ent = (round(entry,3),i)
	if 1e-4 < entry < 1.9999:
		sv.append(ent)
		
	if 1e-4 < entry <= 2.00e0:
		so.append(ent)

#support vectors and outliers copied into a file which can be used during classification
with open('sv_so.txt','w') as fsv:
	fsv.write('support vectors for classification of '+str(num1)+' & '+str(num2)+' *********************')
	fsv.write('\n')
	for e in sv:
		im = digit_data[e[1]][1]
		counter = 0
		for i,a in enumerate(im[0]):
			if counter == 0:
				previous = a
				counter += 1
			elif counter > 0:
				if previous == a:
					
					counter += 1
				else:
					fsv.write(str(previous)+',')
					fsv.write(str(counter)+',')
					previous = a
					counter = 1
			if i == 783:
				fsv.write(str(previous)+',')
				fsv.write(str(counter)+',')
		
		fsv.write('\n')
	fsv.write('\n')	
	fsv.write('support vectors and outliers for classification of '+str(num1)+' & '+str(num2)+' *********************')
	fsv.write('\n')
	for e in so:
		im = digit_data[e[1]][1]
		counter = 0
		for i,a in enumerate(im[0]):
			if counter == 0:
				previous = a
				counter += 1
			elif counter > 0:
				if previous == a:
					
					counter += 1
				else:
					fsv.write(str(previous)+',')
					fsv.write(str(counter)+',')
					previous = a
					counter = 1
			if i == 783:
				fsv.write(str(previous)+',')
				fsv.write(str(counter)+',')
		
		fsv.write('\n')
#print sv
#print so
		
print ('No. of Support Vectors = {}'.format(len(sv)))
print ('No. of Support Vectors + Outliers = {}'.format(len(so)))

sum_double = 0
for i,entry_m in enumerate(so):
	for j,entry_n in enumerate(sv):
		sum_double += entry_m[0]*y[entry_n[1]]*Q[i][entry_n[1]]
		
sum_y = 0
for entry_n in sv:
	sum_y += y[entry_n[1]]
	
w0 = (sum_y-sum_double)/len(sv)
print('wo = {}'.format(w0))
		
		
print('Testing')
mnist_test = read('testing')
digit_data_test = []

i = 0
j = 0
while (i<num_entries_test or j<num_entries_test):
	#reading test dataset and copying into python list
	image = next(mnist_test)
	
	
	if(image[0] == 1 or image[0] == 2):
		
		digit_data_test.append(image)
		
		if(image[0]==1):
			i+=1
		else:
			j+=1

n = len(digit_data_test)
print ('Total no. of Test samples {}'.format(n))

result = []
error_counter = 0
for i,entry in enumerate(digit_data_test):
	
	#classifying test data
	f = classifiction(so,digit_data,entry[1][0],w0,gamma)
	
	if f < 0:
		r = (1,entry[0])
	if f > 0:
		r = (2,entry[0])
		
	result.append(r)
	#comparing the predicted value to actual value
	if r[0] != r[1]:
		error_counter += 1
	

#print result
print ('Total no. of wrongly classified digits are {} out of a total of {}'.format(error_counter,n))
error_percent = (float(error_counter)/float(n))*100
print ('Error Percentage = {}'.format(error_percent))

