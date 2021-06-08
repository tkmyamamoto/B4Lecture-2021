import numpy as np


def get_framing_data(input, win_size, overlap=0.5):
    '''
    get_framing_data is framing data

    Parameters
    ----------
    input: ndarray
        input data
    win_size: int
        window size
    overlap: float
        overlap value

    Returns
    -------
    framing_data: ndarray
        framing data
    '''

    input_length = len(input)
    shift_size = int(win_size*overlap)

    # n_frame = (input_length - win_size) // shift_size
    # framing_data = np.zeros((n_frame, win_size))

    # for i in range(n_frame):
    #     framing_data[i] = input[i*shift_size:i*shift_size + win_size]

    framing_data = []
    for i in range(0,  input_length, shift_size):
        x = input[i:i+win_size]
        if win_size > len(x):
            break
        framing_data.append(x)

    return np.array(framing_data)


def get_lb_method_error(ac, a, dim):
    e = 0
    for i in range(dim-1):
        e += ac[i]*a[i]

    return e
