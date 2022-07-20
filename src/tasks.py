# This file holds the mean and standard deviation for each task
class Task:
    def __init__(self, name):
        if name == "solar":
            self.mean = [0.507, 0.513, 0.462]
            self.std = [0.172, 0.133, 0.114]
        else:
            raise NotImplementedError("Task not implemented")
