def predict(experience, w, b):
    if not isinstance(experience, (int, float)):
        return IOError('Your experience must be a number')
    return w * experience  + b