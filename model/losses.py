import tensorflow as tf
import keras.backend as K


def gener_loss(crossEntropy, z):
    z = tf.reduce_mean(z, keepdims=True, axis=2)
    sparsity_factor = 5e-5  # 0.3
    coherent_ratio = 1e-4  # 2.0
    coherent_factor = sparsity_factor * coherent_ratio

    def losses(y_true, y_pred):
        predDiff = tf.square(y_true - y_pred)
        logPzSum = tf.reduce_sum(crossEntropy, axis=1)
        Zsum = tf.reduce_sum(z, axis=1)
        Zdiff = tf.reduce_sum(tf.abs(z[:, 1:] - z[:, :-1]), axis=1)
        costVec = predDiff + Zsum * sparsity_factor + Zdiff * coherent_factor
        costLogPz = tf.reduce_mean(costVec * logPzSum)
        return costLogPz

    return losses


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def binary_focal_loss_fixed(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """
    gamma = 2.
    alpha = .25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    epsilon = K.epsilon()
    # clip to prevent NaN's and Inf's
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
           - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def integrated_loss(z, crossEntropy, sparsity):
    z = tf.reduce_mean(z, keepdims=True, axis=2)
    sparsity_factor = sparsity
    coherent_ratio = 0.1  # 1e-4 #2.0
    coherent_factor = sparsity_factor * coherent_ratio

    def losses(y_true, y_pred):
        predDiff = binary_focal_loss_fixed(y_true, y_pred)
        # predDiff = K.categorical_crossentropy(y_true, y_pred)
        logPzSum = 1.0  # tf.reduce_sum(crossEntropy, axis=1)
        Zsum = tf.reduce_sum(z, axis=1)
        Zdiff = tf.reduce_sum(tf.abs(z[:, 1:] - z[:, :-1]), axis=1)
        costVec = 10 * predDiff + Zsum * sparsity_factor + Zdiff * coherent_factor
        costLogPz = tf.reduce_mean(costVec * logPzSum)
        return costLogPz

    return losses
