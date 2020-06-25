import matplotlib.pyplot as plt
import numpy as np
x=[1,2,3,4]
x=np.array(x,dtype=float)
y=[1,2,3,4]
y=np.array(y,dtype=float)
z=[2,3,4,5]
z=np.array(z,dtype=float)
plt.figure()
plt.plot(np.arange(len(x)),y,'g-')
plt.plot(np.arange(1,len(x)+1),y,'r-')
plt.show()