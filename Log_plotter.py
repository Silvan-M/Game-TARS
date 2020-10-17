import csv
import matplotlib.pyplot as plt
import numpy as np
# settings #####################################################
file_name = 'test.txt'
label = ['n','total_reward','epsilon','avg_reward', 'losses', 'win_count', 'lose_count', 'illegal_moves']
plot =['win_count', 'lose_count']
color_mode = 'cyanred' #choose from gray, blue, red, yellow, cyanred, gremag, yelblue
################################################################

def color_generator(amount, scale):

    colors = []
    difference = round((255/amount)-0.5)

    for i in range(amount+1):
        number = (i)* difference
        rev_number = 255 - number
        rev_number = format(rev_number,'02x')
        number = hex(number)
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
print(whole_data)
color = color_generator(len(plot)+1, 'cyanred')
x=[]
for i in range(line_count):
    x.append(whole_data[i][0])
plt.style.use('fivethirtyeight')
for q in range(len(plot)):
    i = label.index(plot[q])
    for y in range(line_count):
        plot_data.append(whole_data[y][i])
    print(plot_data)
    plt.plot(x,plot_data, color = color[q],marker="o",label = label[i])
    #plt.ylim(0,5)

    plot_data=[]
plt.title("File: "+str(file_name))
plt.xlabel("Episodes")
plt.ylabel("Value")
plt.legend(loc="upper right",fontsize = 'x-small')
#plt.savefig('fig.'+file_name+".pdf")
plt.show()

