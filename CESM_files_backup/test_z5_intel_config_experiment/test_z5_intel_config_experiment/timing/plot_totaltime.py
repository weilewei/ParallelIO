#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

#plt.style.use('seaborn-talk')
print(plt.style.available)
write_pnetcdf = np.array([478.134,474.176,473.535,469.775,472.138,469.858])
write_z5 = np.array([609.241,475.514,552.091,478.917,587.988,471.414,478.562])
mean_write_pnetcdf=np.mean(write_pnetcdf)
mean_write_z5 = np.mean(write_z5)
std_write_pnetcdf = np.std(write_pnetcdf)
std_write_z5 = np.std(write_z5)

x_name = ['PnetCDF', 'Z5']
x_pos = np.arange(len(x_name))
CTEs = [mean_write_pnetcdf, mean_write_z5]
error = [std_write_pnetcdf, std_write_z5]

fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, width=0.4, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10, color=['purple', 'red'])
ax.set_xlabel('File I/O Backends', fontsize=15)
ax.set_ylabel('Time (s)')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_name)
ax.set_title('Total CESM Simulation Time', fontsize=15)
#ax.yaxis.grid(True)
plt.autoscale()
plt.tight_layout()
plt.savefig('totaltime.png')
plt.show()

#y_axis=[2016,2017,2018,2019]
#xi=[i for i in range(0,len(y_axis))]
#x_axis = [25,50,102,210]
#bar_width=0.1
##plt.rcParams['axes.linewidth']=4
#plt.bar(xi,x_axis,bar_width,alpha=1)
#plt.xticks(xi,y_axis)
#plt.show
#
#
#points = np.array([(0,29),(1,58),(2,104),(3,218)])
#x = points[:,0]
#y = points[:,1]
#
#z = np.polyfit(x,y,2)
#f = np.poly1d(z)
#print f
#x_new = np.linspace(x[0],x[-1],50)
#y_new = f(x_new)
#plt.plot(x,y,' ',x_new,y_new)
#plt.show
#plt.savefig( 'z5_performance.png',format='png')
