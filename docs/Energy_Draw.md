# Energy Draw Callback

A callback for recording GPU power draw (watts) on epoch begin and end. The callback allows: 

- record and analyze model energy consumption
- optimize towards energy efficient models

### how-to-use

Before `model.fit()` in the input model:

`power_draw = PowerDrawCallback()`

Then use `power_draw` as you would callbacks in general:

`model.fit(...callbacks=[power_draw]...)`

To get the energy draw data into the experiment log:

`history = talos.utils.power_draw_append(history, power_draw)`

NOTE: this line has to be after `model.fit()`.

