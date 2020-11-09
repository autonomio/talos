from tensorflow.keras.callbacks import Callback


class PowerDrawCallback(Callback):

    '''A callback for recording GPU power draw (watts) on epoch begin and end.

    Example use:

    power_draw = PowerDrawCallback()

    model.fit(...callbacks=[power_draw]...)

    print(power_draw.logs)

    '''

    def __init__(self):

        super(PowerDrawCallback, self).__init__()

        import os
        import time

        self.os = os
        self.time = time.time
        self.command = "nvidia-smi -i 0 -q | grep -i 'power draw' | tr -s ' ' | cut -d ' ' -f5"

    def on_train_begin(self, logs={}):
        self.log = {}
        self.log['epoch_begin'] = []
        self.log['epoch_end'] = []
        self.log['seconds'] = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_start_time = self.time()
        temp = self.os.popen(self.command).read()
        temp = float(temp.strip())
        self.log['epoch_begin'].append(temp)

    def on_epoch_end(self, batch, logs=None):
        temp = self.os.popen(self.command).read()
        temp = float(temp.strip())
        self.log['epoch_end'].append(temp)
        seconds = round(self.time() - self.epoch_start_time, 3)
        self.log['seconds'].append(seconds)
