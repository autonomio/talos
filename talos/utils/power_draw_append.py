def power_draw_append(history, power_draw):

    '''For appending the data from PowerDrawCallback to the history object
    and allowing the data to be captured in the experiment log in Talos.'''

    import numpy as np

    joined = power_draw.log['epoch_begin'] + power_draw.log['epoch_end']
    avg_watts = (np.array(joined)) / 2

    history.history['watts_min'] = [min(joined)]
    history.history['watts_max'] = [max(joined)]
    history.history['seconds'] = [sum(power_draw.log['seconds'])]

    watt_seconds = round(sum(avg_watts * np.array(power_draw.log['seconds'])), 2)
    history.history['Ws'] = [watt_seconds]

    return history
