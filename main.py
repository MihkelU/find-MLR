#!/usr/bin/env python
#-*- coding: utf-8 -*-
from tkinter import *
from tkinter import filedialog

import csv
import GeneticAlgorithm

class mainWindow(Frame):

    def __init__(self, master):
        Frame.__init__(self, master)

        self.master = master
        self.initUI()

    def initUI(self):
        # Reading in the data matrix
        self.master.title("Descriptor finder")
        self.pack(fill=BOTH, expand=1)

        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open", command=self.readin_DP_Matrix)
        menubar.add_cascade(label="File", menu=fileMenu)
        fileMenu.add_command(label="Exit", command=quit)

        # Checkbutton for reducing number of descriptors
        self.var = BooleanVar()
        cb = Checkbutton(self, text="Reduce the number of descriptors", variable=self.var)
        cb.select()
        cb.grid(row=0)

        # Creating widgets for GA parameters
        self.label1 = Label(self, text="Genetic Algorithm parameters")
        self.label2 = Label(self, text="mutation rate:")
        self.label3 = Label(self, text="crossover rate:")
        self.label4 = Label(self, text="number of generations:")
        self.label5 = Label(self, text="population size:")
        self.label6 = Label(self, text="number of features:")
        self.label7 = Label(self, text="number of best models:")

        self.label1.grid(row=1, column=0, columnspan=2)
        self.label2.grid(row=2, column=0, sticky=E)
        self.label3.grid(row=3, column=0, sticky=E)
        self.label4.grid(row=4, column=0, sticky=E)
        self.label5.grid(row=5, column=0, sticky=E)
        self.label6.grid(row=6, column=0, sticky=E)
        self.label7.grid(row=7, column=0, sticky=E)

        entry_width = 10
        self.entry1 = Entry(self, width=entry_width)
        self.entry2 = Entry(self, width=entry_width)
        self.entry3 = Entry(self, width=entry_width)
        self.entry4 = Entry(self, width=entry_width)
        self.entry5 = Entry(self, width=entry_width)
        self.entry6 = Entry(self, width=entry_width)
        self.entry1.grid(row=2, column=1)
        self.entry2.grid(row=3, column=1)
        self.entry3.grid(row=4, column=1)
        self.entry4.grid(row=5, column=1)
        self.entry5.grid(row=6, column=1)
        self.entry6.grid(row=7, column=1)
        self.entry1.focus()

        mutation_rate, crossover_rate, n_of_generations, pop_size, n_of_features, n_of_best_models\
            = 0.05, 0.7, 10, 500, 3, 10 #Default values of the variables
        self.entry1.insert(END, mutation_rate)
        self.entry2.insert(END, crossover_rate)
        self.entry3.insert(END, n_of_generations)
        self.entry4.insert(END, pop_size)
        self.entry5.insert(END, n_of_features)
        self.entry6.insert(END, n_of_best_models)

        # Run the Genetic Algorithm
        self.button1 = Button(self, text="Run", fg="white", bg="red", command=self.runGA)
        self.button1.grid(row=8, column=1, sticky=(N,S,W,E))

    def readin_DP_Matrix(self):

        ftypes = [('Comma Separated Values','*.csv'),("Text files","*.txt"),('Python files', '*.py'),('All files','*')]
        filename = filedialog.askopenfilename(filetypes=ftypes)

        with open(filename,"r") as f:
            matrix_object = csv.reader(f, delimiter=';',quotechar='|')
            self.DP_Matrix = [row for row in matrix_object]

    def runGA(self):

        reduce = self.var.get()
        mutation_rate = float(self.entry1.get())
        crossover_rate = float(self.entry2.get())
        N_of_generations = int(self.entry3.get())
        population_size = int(self.entry4.get())
        N_of_features = int(self.entry5.get())
        N_of_models = int(self.entry6.get())

        self.GA = GeneticAlgorithm.Evolution(matrix=self.DP_Matrix, reduce=reduce,mutation_rate=mutation_rate,
                                             crossover_rate=crossover_rate, N_of_generations=N_of_generations,
                                             population_size=population_size, N_of_features=N_of_features,
                                             N_of_models=N_of_models)
        self.master.quit()


def main():

    root = Tk()
    root.geometry("300x300+300+300")
    app = mainWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()

