import numpy as np
from time import process_time
from utils import get_label

def critic(c, m, X, y, model):
    s, yp = None, None
    if c:
        yp = m.predict(X)
        s = m.score(X, y)
        c = 0
    else:
        m = model(X, y)
        c = 1
    return c, m, s, yp

def evolution(X, y, model, start=10000, batch=10):
    score = []
    predictions = []

    gmax = len(X)
    m = model()
    g = start
    c = 0

    _start = process_time()

    while g < gmax:
        gnext = g+batch
        yg = y[g:gnext]
        Xg = X[g:gnext]

        c, m, s, yp = critic(c, m, Xg, yg, model)
            
        g = gnext

        if not c:
            score.append(s)
            predictions.append(yp)

    _end = process_time()
    _cost = _end - _start


    return score, predictions, _cost

