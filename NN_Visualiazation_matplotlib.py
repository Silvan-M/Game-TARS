import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as matplotlib
from matplotlib.animation import FuncAnimation
global color 
color = 'white'
fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
def sigmoid(x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
def neuron(coord, radius, color, text):
    circle2 = plt.Circle((coord[0], coord[1]), radius, color='black', fill = True)
    circle1 = plt.Circle((coord[0], coord[1]), radius, color=color, fill = False)
    ax.add_artist(circle2)
    ax.add_artist(circle1)
    ax.text(coord[0], coord[1],text, horizontalalignment='center',verticalalignment='center', size = 'smaller', color = 'white')
def draw(raw_data, weights, nodes, error, inp, val,n): #
    global color

    text = ['Input','Input Layer','Hidden Layer', 'Output Layer']
    ax.text(4, 4, text[0], horizontalalignment='center',verticalalignment='center', size = 'smaller', color = 'white')
    ax.text(8, 4, text[1], horizontalalignment='center',verticalalignment='center', size = 'smaller', color = 'white')
    ax.text(12, 4, text[2], horizontalalignment='center',verticalalignment='center', size = 'smaller', color = 'white')
    ax.text(16.5, 4, text[3], horizontalalignment='center',verticalalignment='center', size = 'smaller', color = 'white')
    
    neuron([4,8],0.5,'white',raw_data[2])
    neuron([4,11],0.5,'white',raw_data[1])
    neuron([4,14],0.5,'r',raw_data[0])
    neuron([8,8],1,'white',inp[2])
    neuron([8,11],1,'white',inp[1])
    neuron([8,14],1,'white',inp[0])
    neuron([12,11],1,'white','{:.3f}'.format(nodes[0]))
    neuron([16.5,11],1,'white','{:.3f}'.format(val))

    factor = 1/10
    color1 = lighten_color('b',1-(weights[2])*factor)
    color2 =lighten_color('b',1-(weights[1])*factor)
    color3 = lighten_color('b',1-(weights[0])*factor)
    plt.plot([9,11], [8,11],color = color1, linewidth=4, zorder = 1)
    plt.plot([9,11], [11,11],color = color2, linewidth=4, zorder = 1)
    plt.plot([9,11], [14,11],color = color3, linewidth=4, zorder = 1)
    ax.text(9.5, 14, '{:.3f}'.format(weights[0]), horizontalalignment='left',verticalalignment='center', size = 4, color = 'white')
    ax.text(9.5, 11.5,'{:.3f}'.format(weights[1]), horizontalalignment='left',verticalalignment='center', size = 4, color = 'white')
    ax.text(9.5, 8, '{:.3f}'.format(weights[2]), horizontalalignment='left',verticalalignment='center', size = 4, color = 'white')
    plt.plot([20,20], [2,18],color = 'white', linewidth=1)
     
    shift = 2
    ax.text(23, 14-shift, 'Wanted Output:', horizontalalignment='center',verticalalignment='center', size = 'x-small', color = 'white')
    ax.text(23, 13-shift, raw_data[0], horizontalalignment='center',verticalalignment='center', size = 'x-small', color = 'white')
    ax.text(23, 11-shift, 'Actual Output:', horizontalalignment='center',verticalalignment='center', size = 'x-small', color = 'white')
    ax.text(23, 10-shift, '{:.6f}'.format(val), horizontalalignment='center',verticalalignment='center', size = 'x-small', color = 'white')
    ax.text(23, 8-shift, 'Error:', horizontalalignment='center',verticalalignment='center', size = 'x-small', color = 'white')
    ax.text(23, 7-shift, '{:.6f}'.format(error) , horizontalalignment='center',verticalalignment='center', size = 'x-small', color = 'white')
    ax.text(23, 17-shift, 'Episode:', horizontalalignment='center',verticalalignment='center', size = 'x-small', color = 'white')
    ax.text(23, 16-shift, '{:6}'.format(n) , horizontalalignment='center',verticalalignment='center', size = 'x-small', color = 'white')
     
    plt.axis('off')
def i(template):
    plt.cla()
    ax.set_xlim((0, 25))
    ax.set_ylim((0, 20))
    a = ax.arrow(4.5,8,2,0, head_width = 0.2, color = 'white')
    b = ax.arrow(4.5,11,2,0, head_width = 0.2, color = 'white')
    c = ax.arrow(4.5,14,2,0, head_width = 0.2, color = 'white')
    d = ax.arrow(13,11,2,0, head_width = 0.2, color = 'white')
    plt.rcParams['axes.facecolor'] = 'black'
    ax.add_artist(a)
    ax.add_artist(b)
    ax.add_artist(c)
    ax.add_artist(d)
    draw(template[0],template[1], template[2],template[3],template[4],template[5],template[6])
    # template = [[input1,input2,input3],[weights1, weights2, weights3],[nodes],error,[nodevalue1,nodevalue2,nodevalue3], output value]
    plt.axis('off')
    name = '\\'+str(template[-1])
    appendix = '.jpg'
    path = r'C:\Users\41763\Desktop\pictures'
    plt.savefig(path+name+appendix,facecolor = 'black', dpi = 400)

x = [[1,0,0],[0.5,0.5,0.5],[0.5],0.377540,[1,0,0],0.622459,0]
i(x)