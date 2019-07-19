import keras.backend as K

MARGIN = 1.

# Refer to https://github.com/maciejkula/triplet_recommendations_keras

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def triplet_loss(vects):
    # f_anchor.shape = (batch_size, 256)
    f_anchor, f_positive, f_negative = vects

    # Implement the Triplet Loss by your self.
    f_anchor = K.l2_normalize(f_anchor,axis=-1) # np.sum(np.square(f_anchor)) == 1
    f_positive = K.l2_normalize(f_positive,axis=-1)
    f_negative = K.l2_normalize(f_negative,axis=-1)

    distance_ap = K.sum(K.square(K.abs(f_anchor - f_positive)),axis=-1,keepdims=True)
    
    distance_an = K.sum(K.square(K.abs(f_anchor - f_negative)),axis=-1, keepdims=True)
    
    loss = distance_ap - distance_an + MARGIN

    return loss
