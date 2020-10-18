import csv
import matplotlib.pyplot as plt
import numpy as np
# settings #####################################################
file_name = 'test.txt' #input filename
label = ['n','total_reward','epsilon','avg_reward', 'losses', 'win_count', 'lose_count', 'illegal_moves'] # all saved data in csv file
plot =['win_count','lose_count'] # put in here what should be processed
color_mode = 'cyanred' #choose from gray, blue, red, yellow, cyanred, gremag, yelblue
regression = True # make a regression 
reg_dim = 1 # dimension of regression
################################################################
def color_brightener(color): #brightens the color
    first = color[1:3]
    snd = color[3:5]
    thrd = color[5:7]
    # splits input color into 3 subparts each in hex form
    first = hex(int(first,16) + 50)
    snd = hex(int(snd,16) + 50)
    thrd = hex(int(thrd,16) + 50)
    # convert all number to int with int(hex_number,16) and add 50 in decimal
    hex_number ='#'+ str(first[2:])+ str(snd[2:])+ str(thrd[2:])
    # add all subparts together
    print(f'Brightened {color} to {hex_number}.')
    return(hex_number)

def color_generator(amount, scale): # generates equally distributed colors

    colors = []
    difference = round((255/amount)-0.5)
    # hex numbers have 3 values  (red,gree,blue) each in hex in order to make colors just all numbers from 0 to 255 are valid
    # difference is 255/amount so we get the step we need to add to get the next color
    for i in range(amount+1):
        number = (i)* difference
        rev_number = 255 - number
        rev_number = format(rev_number,'02x')
        number = hex(number)
        # depending on the scale the numbers are differently used
        if scale == 'gray':
            hex_number ='#'+ str(number[2:])+ str(number[2:])+ str(number[2:])
        elif scale == 'blue':
            hex_number ='#'+ str(00)+ str(number[2:])+ str(number[2:])
        elif scale == 'red':
            hex_number ='#'+ str(number[2])+ str(00)+ str(number[2:])
        elif scale == 'yellow':
            hex_number ='#'+ str(number[2:])+ str(number[2:])+ str(00)
        elif scale == 'cyanred' or scale == 'gremag' or scale == 'yelblu' :
            if scale == 'cyanred':
                hex_number ='#'+ str(rev_number)+ str(number[2:])+ str(number[2:])
            elif scale == 'gremag':
                hex_number ='#'+ str(number[2:])+ str(rev_number)+ str(number[2:])
            else:
                hex_number ='#'+ str(number[2:])+ str(number[2:])+ str(rev_number)
        colors.append(hex_number)
    colors = colors[1:]
    print(f'Generated {amount-1} colors in the {scale} mode. The colors are: {colors[1:]}')
    return(colors)

        

row_data = []
amount_of_rows = 0
whole_data = []
plot_data = []
with open('test.txt') as csv_file:
    # read csv file
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        print(row)
        line_count += 1
        amount_of_rows = len(row)
        for i in range(len(row)):
            row_data.append(row[i])
        whole_data.append(row_data)
        row_data = []                   
    print(f'Processed {line_count} lines with {amount_of_rows} entries.')
color = color_generator(len(plot)+1, 'cyanred')
x=[]
for i in range(line_count):
    x.append(int(whole_data[i][0]))
plt.style.use('fivethirtyeight')
for q in range(len(plot)):
    i = label.index(plot[q])
    for y in range(line_count):
        plot_data.append(float(whole_data[y][i]))
    plt.plot(x,plot_data ,color = color[q],label = label[i],linewidth = 0.75)
    #plt.ylim(0,5)
    if regression:
        dim = reg_dim
        coef = np.polyfit(x,plot_data,dim)
        poly1d_fn = np.poly1d(coef) 
        plt.plot(x, poly1d_fn(x), color = color_brightener(color[q]), label = "Avg. Incr. = "+str(round(coef[0]*len(x)*100,3))+"%",linewidth = 0.75, linestyle = (0,(5,5)))
    plot_data=[]
plt.title("File: "+str(file_name))
plt.xlabel("Episodes")
plt.ylabel("Value")
plt.legend(loc="upper right",fontsize = 'x-small')
#plt.savefig('fig.'+file_name+".pdf")
plt.show()

