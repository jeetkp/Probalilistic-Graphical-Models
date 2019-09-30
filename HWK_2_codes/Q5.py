import numpy as np
import scipy.io

file= scipy.io.loadmat('C:/Users/jeetp/OneDrive/Desktop/GAtech study material/ece8803 PGM/Assignment2/banana.mat')

pxgh= file['pxgh'].tolist()
pygh= file['pygh'].tolist();
phtghtm = file['phtghtm'].tolist();
ph1 = file['ph1'].tolist();
x= file['x'].tolist()
# x=['AA']
characters=[]
for c in x[0]:
    characters.append(c)
#ACGT
characters_ints=[]
for c in characters:
    if(c=='A'):
        characters_ints.append(0);
    if(c=='C'):
        characters_ints.append(1);
    if(c=='G'):
        characters_ints.append(2);
    if(c == 'T'):
        characters_ints.append(3);

ph1_flat=[]
for sublist in ph1:
    for item in sublist:
        ph1_flat.append(item)
# print(ph1_flat)
# print(pxgh[0])
h=np.zeros([len(characters_ints),len(ph1_flat)])
index=np.zeros([len(characters_ints),len(ph1_flat)])
h[0]=[a*b for a,b in zip(ph1_flat,pxgh[characters_ints[0]])]
index[0]=np.ones([1,5])
for i in range(1,len(characters_ints)):
    b=np.amax(phtghtm*h[i-1],axis=1);
    h[i]=pxgh[characters_ints[i]]
    h[i]= h[i]*b
    index[i] = np.argmax(phtghtm*h[i-1],axis=1);
index=index.tolist();
h_max= [0]*len(characters_ints);
h_max[len(index)-1]= np.argmax(h[-1]);
# print(index)
for i in range(len(characters_ints)-1,0,-1):
    h_max[i-1]=int(index[i][h_max[i]]);
y_max= []
# print(ph1_flat[0][int(h_max[0])])
# print(pygh[:,h_max[0]])
# y_max[0]=np.argmax(pygh[:,h_max[0]]*ph1[h_max[0]]);
yyy_mul = ph1_flat[h_max[0]]
# yy_append
yyy=[]
for i in range(len(pygh)):
    yyy.append(pygh[i][h_max[0]]*yyy_mul)
# yyy.append(pygh[:][h_max[0]]*yyy_mul)
# yyy=[x*yyy_mul[0] for x in ph1_flat[0][h_max[0]]]
y_max.append(np.argmax(yyy))
for i in range(1,len(characters_ints)):
    yyy_mul = phtghtm[h_max[i]][h_max[i-1]]
    # print(i)
    yyy=[]
    for ii in range(0,len(pygh)):
        yyy.append(pygh[ii][h_max[i]]*yyy_mul)
    y_max.append(np.argmax(yyy))
    # y_max.append(np.argmax(pygh[:][h_max[i]]*phtghtm[h_max[i]][h_max[i-1]]))
y_max_char=''.join(str(x) for x in y_max)
y_max_char= y_max_char.replace('0','A')
y_max_char= y_max_char.replace('1','C')
y_max_char= y_max_char.replace('2','G')
y_max_char= y_max_char.replace('3','T')

print(" The Output Y sequence obtained is - ")
print(y_max_char)
# print("hiii")
