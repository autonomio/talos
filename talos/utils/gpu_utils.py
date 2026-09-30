def parallel_gpu_jobs(allow_growth=True, fraction=.5, backend='tensorflow'):
    if isinstance(allow_growth, (int, float)) and not isinstance(allow_growth, bool):
        fraction, allow_growth = allow_growth, False
    if not 0 < fraction <= 1:
        raise ValueError('fraction must lie between zero and one.')
    if backend in ('torch', 'pytorch'):
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction)
        return
    import tensorflow as tf
    devices = tf.config.list_physical_devices('GPU')
    for device in devices:
        if fraction < 1:
            import subprocess
            index = devices.index(device)
            result = subprocess.run(['nvidia-smi', '-i', str(index), '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                                    check=True, capture_output=True, text=True)
            limit = float(result.stdout.strip()) * fraction
            tf.config.set_logical_device_configuration(device, [tf.config.LogicalDeviceConfiguration(memory_limit=limit)])
        elif allow_growth:
            tf.config.experimental.set_memory_growth(device, True)


def multi_gpu(model, gpus=None, cpu_merge=True, cpu_relocation=False):
    from talos.backends import backend_for
    if backend_for(model).name == 'torch':
        import torch
        return torch.nn.DataParallel(model, device_ids=gpus if isinstance(gpus, list) else None)
    import tensorflow as tf
    devices = tf.config.list_logical_devices('GPU')
    if isinstance(gpus, int):
        devices = devices[:gpus]
    if len(devices) < 2:
        return model
    strategy = tf.distribute.MirroredStrategy(devices=[device.name for device in devices])
    with strategy.scope():
        cloned = tf.keras.models.clone_model(model)
        cloned.set_weights(model.get_weights())
    return cloned


def force_cpu(backend='tensorflow'):
    if backend in ('torch', 'pytorch'):
        return 'cpu'
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
