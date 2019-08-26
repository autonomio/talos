from keras.callbacks import Callback


class ExperimentLogCallback(Callback):

    def __init__(self,
                 experiment_name,
                 params):

        '''Takes as input the name of the experiment which will be
        used for creating a .log file with the outputs and the params
        dictionary from the input model in `Scan()`

        experiment_name | str | must match the experiment_name in `Scan()`

        '''

        super(ExperimentLogCallback, self).__init__()

        import glob
        import os

        # get the experiment id first
        list_of_files = glob.glob('./' + experiment_name + '/*.csv')
        try:
            latest_file = max(list_of_files, key=os.path.getmtime)
        except ValueError:
            print("`experiment_name` has to match `Scan(experiment_name)`")

        self.name = latest_file.replace('.csv', '') + '.log'

        # rest of the config variables
        self.params = params
        self.counter = 1
        self.new_file = True

    def on_train_begin(self, logs={}):

        import random
        self.hash = hex(abs(hash(str(random.random()))))
        self.final_out = []

    def on_train_end(self, logs={}):

        f = open(self.name, 'a+')
        [f.write(','.join(map(str, i)) + '\n') for i in self.final_out]
        f.close()

    def on_epoch_begin(self, epoch, logs={}):

        self.epoch_out = []

    def on_epoch_end(self, epoch, logs={}):

        if len(self.final_out) == 0:

            try:
                open(self.name, 'r')
            except FileNotFoundError:

                self.epoch_out.append('id')
                self.epoch_out.append('epoch')

                for key in logs.keys():

                    # add to the epoch out list
                    self.epoch_out.append(key)

                self.final_out.append(self.epoch_out)
                self.epoch_out = []

        self.epoch_out.append(self.hash)
        self.epoch_out.append(epoch + 1)

        for key in logs.keys():

            # round the values
            rounded = round(logs[key], 4)

            # add to the epoch out list
            self.epoch_out.append(rounded)

        # add to the final out list
        self.final_out.append(self.epoch_out)
