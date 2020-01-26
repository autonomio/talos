# Energy Draw Callback

A callback for recording GPU power draw (watts) on epoch begin and end. The callback allows: 

- record and analyze model energy consumption
- optimize towards energy efficient models

### how-to-use

Use it as you would use any other Callback in Tensorflow or Keras.

`
power_draw = PowerDrawCallback()

model.fit(...callbacks=[power_draw]...)
`

It's possible to read the energy draw data:

`print(power_draw.logs)`

