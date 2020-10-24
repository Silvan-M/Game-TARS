import csv
import matplotlib.pyplot as plt
import numpy as np
# settings #####################################################
file_name = 'C:/Users/41763/Desktop/logs/SpaceInvadersFirstTrainingN500.txt' #input filename
name = 'Snake - 5000 Ep - 2020.10.21' # name the plot

#label = ['n','total_reward','epsilon','avg_reward', 'losses', 'win_count', 'lose_count', 'illegal_moves'] # all saved data in csv file
label = ['n','total_reward','epsilon','avg_reward', 'losses', 'avg_points']
plot =['n'] # put in here what should be processed
color_mode = 'cyanred' #choose from gray, blue, red, yellow, cyanred, gremag, yelblue
regression = False # make a regression 
reg_dim = 1 # dimension of regression
reg_mode = 'normal' #normal
predict = False # if the prediction should be plottet
range_predict = 10 # range of the prediction
reg_func_inp =[1000000]
save_file = True
################################################################
def color_brightener(color, dim =0): #brightens the color
    first = color[1:3]
    snd = color[3:5]
    thrd = color[5:7]
    # splits input color into 3 subparts each in hex form
    if dim == 0:
        if int(first,16) > 204 or int(snd,16) > 204 or int(thrd,16) > 204:
            return(color)
        else:
            first = hex(int(first,16) + 50)
            snd = hex(int(snd,16) + 50)
            thrd = hex(int(thrd,16) + 50)
    else:
        if int(first,16) > 154 or int(snd,16) > 154 or int(thrd,16) > 154:
            return(color)
        else:
            first = hex(int(first,16) + 100)
            snd = hex(int(snd,16) + 100)
            thrd = hex(int(thrd,16) + 100)
    # convert all number to int with int(hex_number,16) and add 50 in decimal
    hex_number ='#'+ str(first[2:])+ str(snd[2:])+ str(thrd[2:])
    # add all subparts together
    print(f'Brightened {color} to {hex_number}.')
    return(hex_number)

def color_generator(amount, scale): # generates equally distributed colors
    wins = []
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

        
wins = []
row_data = []
amount_of_rows = 0
whole_data = []
plot_data = []
with open(file_name) as csv_file:
    # read csv file
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        line_count += 1
        amount_of_rows = len(row)
        for i in range(len(row)):
            row_data.append(row[i])
        whole_data.append(row_data)
        wins.append(row_data[5])        
        row_data = []           
    print(f'Processed {line_count} lines with {amount_of_rows} entries.')
color = color_generator(len(plot)+1, 'cyanred')
x=[]

beg_av = (float(wins[0])+float(wins[2]))/2
end_av = (float(wins[-1])+float(wins[-2]))/2
print(f'Win in % first {beg_av} in the end {end_av}.')
for i in range(line_count):
    x.append(int(whole_data[i][0]))
plt.style.use('fivethirtyeight')
params = {'legend.fontsize': 20,
         'axes.labelsize':  9,
         'axes.titlesize':  9,
         'xtick.labelsize': 7,
         'ytick.labelsize': 7}
plt.rcParams.update(params)
for q in range(len(plot)):
    i = label.index(plot[q])
    for y in range(line_count):
        plot_data.append(float(whole_data[y][i]))
    plt.plot(x,plot_data ,color = color[q],label = label[i],linewidth = 1.2)
    #plt.ylim(0,5)
    if regression:
        dim = reg_dim
        coef = np.polyfit(x,plot_data,dim)
        poly1d_fn = np.poly1d(coef) 

        if dim == 1 and reg_mode == 'normal':
            plt.plot(x, poly1d_fn(x), color = color_brightener(color[q]), label = "Total Incr. = "+str(round(coef[0]*100*len(x),3))+"%",linewidth = 1.2, linestyle = (0,(5,5)))
        else:
            function = 'f(x) = '
            for i in range(len(coef)):
                h = len(coef) -i-1
                if h == 0:
                    function = function +str(round(coef[i],4))
                elif h == 1:
                    function = function + str(round(coef[i],4)) + ' x +'
                else:
                    function = function + str(round(coef[i],4)) + ' x^'+str(h)+'+'
            if dim > 12:
                function =f'Regression function of {plot[q]}'

            plt.plot(x, poly1d_fn(x), color = color_brightener(color[q]), label = function,linewidth = 2, linestyle = (0,(5,5)))
            if predict:
                step = float(whole_data[3][0])-float(whole_data[2][0])
                new_x = []
                end = float(whole_data[-1][0])
                for i in range(range_predict):
                    new_x.append(end + i*step)
                plt.plot(new_x, poly1d_fn(new_x), color = color_brightener(color[q],dim=1) , label = function+'(prediction)',linewidth = 2, linestyle = (0,(5,5)))
                
            print(f'The regression function is: {function}')
            out = poly1d_fn(reg_func_inp)
            for i in range(len(reg_func_inp)):
                print(f'Prediction for {reg_func_inp[i]} is {out[i]}')
                

    plot_data=[]
plt.title(str(name), fontsize=12)
plt.xlabel("Episodes")
plt.ylabel("Value")
plt.legend(loc="upper right",fontsize = 'x-small')
if save_file:
    plt.savefig('fig.'+name+".pdf")
plt.show()

