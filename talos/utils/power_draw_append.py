def power_draw_append(history, power_draw):

    '''For appending the data from PowerDrawCallback to the history object
    and allowing the data to be captured in the experiment log in Talos.

    history | object | tf.keras model history object
    power_draw | object | PowerDrawCallback object

    '''

    import numpy as np

    joined = power_draw.log['epoch_begin'] + power_draw.log['epoch_end']
    history.history['watts_min'] = [min(joined)]
    history.history['watts_max'] = [max(joined)]
    history.history['seconds'] = [sum(power_draw.log['seconds'])]

    # get average watts per epoc
    epoch_begin = np.array(power_draw.log['epoch_begin'])
    epoch_end = np.array(power_draw.log['epoch_end'])
    avg_watts = (epoch_begin + epoch_end) / 2

    watt_seconds = round(sum(avg_watts * np.array(power_draw.log['seconds'])), 2)
    history.history['Ws'] = [watt_seconds]

    return history
