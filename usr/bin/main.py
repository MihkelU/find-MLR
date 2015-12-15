#!/usr/bin/env python
#-*- coding: utf-8 -*-
from tkinter import *
from tkinter import filedialog

import csv
import threading
import queue
import GeneticAlgorithm
import SimulatedAnnealing
import time
import sys

class mainWindow(Frame):

    def __init__(self, master):
        Frame.__init__(self, master)
        self.master = master
        self.initUI()
        self.queue = queue.Queue()

        self.text = Text(master)
        self.text.grid(row=0, column=0, sticky="nsew")


        self.scroll = Scrollbar(master, command=self.text.yview)
        self.scroll.grid(row=0, column=1, sticky="nsew")
        self.text['yscrollcommand'] = self.scroll.set
        self.periodiccall()


    def periodiccall(self):
        self.checkqueue()
        if threading.main_thread().is_alive():
            self.after(10, self.periodiccall)
        else:
           pass


    def checkqueue(self):

        while self.queue.qsize():
            try:
                msg = self.queue.get(0)
                self.text.configure(state=NORMAL)
                self.text.insert(INSERT, msg)
                self.text.yview(END)
                self.text.config(state=DISABLED)

                print(msg)
            except queue.Empty:
                pass

    def initUI(self):

        # Reading in the data matrix
        self.master.title("Descriptor finder")
        #self.pack(fill=BOTH, expand=1)

        # Creating the menus
        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open", command=self.readin_DP_Matrix)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=sys.exit)
        menubar.add_cascade(label="File", menu=fileMenu)

        calculateMenu = Menu(menubar)
        calculateMenu.add_command(label="GA", command=self.Run_GA)
        calculateMenu.add_command(label="SA", command=self.Run_SA)
        calculateMenu.add_command(label="Full Search", command=self.Run_FS)
        menubar.add_cascade(label="calculate", menu=calculateMenu)


    def readin_DP_Matrix(self):

        ftypes = [('Comma Separated Values','*.csv'),("Text files","*.txt"),('Python files', '*.py'),('All files','*')]
        filename = filedialog.askopenfilename(filetypes=ftypes)

        with open(filename,"r") as f:
            matrix_object = csv.reader(f, delimiter=';',quotechar='|')
            self.DP_Matrix = [row for row in matrix_object]


    def Run_GA(self):

        self.GA_window = Toplevel(self.master)
        self.GA_window_obj = GAWindow(self.GA_window, self.DP_Matrix, self.queue)

    def Run_SA(self):

        self.SA_window = Toplevel(self.master)
        label = Label(self.SA_window, text = "valmimisel!")
        label.pack()

    def Run_FS(self):

        self.FS_window = Toplevel(self.master)
        label = Label(self.FS_window, text = "valmimisel!")
        label.pack()


class GAWindow(Frame):

    def __init__(self, master, matrix, queue):
        Frame.__init__(self, master)

        self.queue = queue
        self.master = master
        self.DP_Matrix = matrix

        self.initUI()


    def initUI(self):

        self.master.title("Run GA")
        self.pack(fill=BOTH, expand=1)

        # Checkbutton for reducing number of descriptors
        self.var = BooleanVar()
        cb = Checkbutton(self, text="Reduce the number of descriptors", variable=self.var)
        cb.select()
        cb.grid(row=0)

        #Creating widgets for GA parameters
        self.label1 = Label(self, text="Genetic Algorithm parameters")
        self.label2 = Label(self, text="mutation rate:")
        self.label3 = Label(self, text="crossover rate:")
        self.label4 = Label(self, text="number of generations:")
        self.label5 = Label(self, text="population size:")
        self.label6 = Label(self, text="number of features:")
        self.label7 = Label(self, text="number of best models:")
        self.label8 = Label(self, text="orthogonality bound:")

        self.label1.grid(row=1, column=0, columnspan=2)
        self.label2.grid(row=2, column=0, sticky=E)
        self.label3.grid(row=3, column=0, sticky=E)
        self.label4.grid(row=4, column=0, sticky=E)
        self.label5.grid(row=5, column=0, sticky=E)
        self.label6.grid(row=6, column=0, sticky=E)
        self.label7.grid(row=7, column=0, sticky=E)
        self.label8.grid(row=8, column=0, sticky=E)

        entry_width = 10
        self.entry1 = Entry(self, width=entry_width)
        self.entry2 = Entry(self, width=entry_width)
        self.entry3 = Entry(self, width=entry_width)
        self.entry4 = Entry(self, width=entry_width)
        self.entry5 = Entry(self, width=entry_width)
        self.entry6 = Entry(self, width=entry_width)
        self.entry7 = Entry(self, width=entry_width)
        self.entry1.grid(row=2, column=1)
        self.entry2.grid(row=3, column=1)
        self.entry3.grid(row=4, column=1)
        self.entry4.grid(row=5, column=1)
        self.entry5.grid(row=6, column=1)
        self.entry6.grid(row=7, column=1)
        self.entry7.grid(row=8, column=1)
        self.entry1.focus()

        mutation_rate, crossover_rate, n_of_generations, pop_size, n_of_features, n_of_best_models, orthogonality\
            = 0.001, 0.7, 10, 500, 3, 10, 0.1 # Default values of the variables
        self.entry1.insert(END, mutation_rate)
        self.entry2.insert(END, crossover_rate)
        self.entry3.insert(END, n_of_generations)
        self.entry4.insert(END, pop_size)
        self.entry5.insert(END, n_of_features)
        self.entry6.insert(END, n_of_best_models)
        self.entry7.insert(END, orthogonality)

        # Run the Genetic Algorithm
        self.button1 = Button(self, text="Run", fg="white", bg="red", command=self.spawn_thread)
        self.button1.grid(row=9, column=1, sticky=(N,S,W,E))


    def spawn_thread(self):

        self.reduce = self.var.get()
        self.mutation_rate = float(self.entry1.get())
        self.crossover_rate = float(self.entry2.get())
        self.N_of_generations = int(self.entry3.get())
        self.population_size = int(self.entry4.get())
        self.N_of_features = int(self.entry5.get())
        self.N_of_models = int(self.entry6.get())
        self.orthogonality = float(self.entry7.get())
        self.button1.config(state="disabled")

        self.thread = GA_Thread(self.queue, self.DP_Matrix, self.reduce, self.mutation_rate,
                                                 self.crossover_rate, self.N_of_generations,
                                                 self.population_size, self.N_of_features,
                                                 self.N_of_models, self.orthogonality)
        self.thread.start()
        self.periodiccall()

    def periodiccall(self): # For turning button1 active
        if self.thread.is_alive():
            self.after(10, self.periodiccall)
        else:
            self.button1.config(state="active")



class GA_Thread(threading.Thread):

    def __init__(self, queue, matrix, reduce, mutation_rate, crossover_rate, N_of_generations,
                 population_size, N_of_features, N_of_models, orthogonality):
        threading.Thread.__init__(self)
        self.queue = queue

        self.DP_Matrix = matrix
        self.reduce = reduce
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.N_of_generations = N_of_generations
        self.population_size = population_size
        self.N_of_features = N_of_features
        self.N_of_models = N_of_models
        self.orthogonality = orthogonality

    # overrididing Thread.run() method
    def run(self):
        self.queue.put(" Genetic Algorithm started! ".center(50,"#"))
        self.queue.put("\n\n")
        self.GA = GeneticAlgorithm.Evolution(matrix=self.DP_Matrix, reduce=self.reduce,mutation_rate=self.mutation_rate,
                                                 crossover_rate=self.crossover_rate, N_of_generations=self.N_of_generations,
                                                 population_size=self.population_size, N_of_features=self.N_of_features,
                                                 N_of_models=self.N_of_models, orthogonality=self.orthogonality, queue=self.queue)


# class SAWindow():
#
#     pass
# class SAThread():
#
#     pass


def main():

    root = Tk()
    app = mainWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()


