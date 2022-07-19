# This file holds the mean and standard deviation for each task
class Task:
    def __init__(self, name):
        if name == "solar":
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2023, 0.1994, 0.2010]
        else:
            raise NotImplementedError("Task not implemented")
