import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential


def carlini_wagner_l2(model, x, **kwargs):
    return CarliniWagnerL2(model, **kwargs).attack(x)

class CarliniWagnerL2Exception(Exception):
    pass

class CarliniWagnerL2(object):
    def __init__(
        self,
        model,
        y=None,
        batch_size=128,
        binary_search_steps=5,
        max_iterations=1_000,
        abort_early=True,
        confidence=0.0,
        initial_const=1e-2,
        learning_rate=5e-3,
    ):
        self.model = model
        self.batch_size = batch_size
        self.y = y
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.learning_rate = learning_rate
        self.confidence = confidence
        self.initial_const = initial_const
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        super(CarliniWagnerL2, self).__init__()

    def attack(self, x):
        adv_ex = np.zeros_like(x)
        adv_ex= self._attack(x).numpy()
        return adv_ex

    def _attack(self, x):
        original_x = tf.cast(x, tf.float32)
        shape = original_x.shape
        y = tf.one_hot(self.y, 43)
        y = tf.reshape(y,(1,43))
        y = tf.cast(y, tf.float32)
        x = original_x
        x = tf.clip_by_value(x, 0.0, 1.0)
        x = (x * 2.0) - 1.0
        x = tf.atanh(x * 0.999999)
        lower_bound = tf.zeros(shape[:1])
        upper_bound = tf.ones(shape[:1]) * 1e10
        const = tf.ones(shape) * self.initial_const
        best_l2 = tf.fill(shape[:1], 1e10)
        best_score = tf.fill(shape[:1], -1)
        best_score = tf.cast(best_score, tf.int32)
        best_attack = original_x
        modifier = tf.Variable(tf.zeros(shape, dtype=x.dtype), trainable=True)
        for outer_step in range(self.binary_search_steps):
            modifier.assign(tf.zeros(shape, dtype=x.dtype))
            for var in self.optimizer.variables():
                var.assign(tf.zeros(var.shape, dtype=var.dtype))
            prev = None
            for iteration in range(self.max_iterations):
                x_new, loss, preds, l2_dist = self.attack_step(x, y, modifier, const)
                if (
                    self.abort_early
                    and iteration % ((self.max_iterations // 10) or 1) == 0
                ):
                    if prev is not None and loss > prev * 0.9999:
                        break
                    prev = loss
                target_label = tf.argmax(y, axis=1)
                pred_with_conf = preds - self.confidence
                pred_with_conf = tf.argmax(pred_with_conf, axis=1)
                pred = tf.argmax(preds, axis=1)
                pred = tf.cast(pred, tf.int32)
                mask = tf.math.logical_and(
                    tf.less(l2_dist, best_l2), tf.equal(pred_with_conf, target_label)
                )
                if(mask):
                    best_l2 = l2_dist
                    best_score = pred
                    best_attack = x_new
            target_label = tf.argmax(y, axis=1)
            target_label = tf.cast(target_label, tf.int32)
            upper_mask = tf.math.logical_and(
                tf.equal(best_score, target_label),
                tf.not_equal(best_score, -1),
            )
            
            if(upper_mask):
                upper_bound = tf.math.minimum(upper_bound, const)
            const_mask = tf.math.logical_and(
                upper_mask,
                tf.less(upper_bound, 1e9),
            )
            if(const_mask.numpy().all()):
                const = (lower_bound + upper_bound) / 2.0
            lower_mask = tf.math.logical_not(upper_mask)
            if(lower_mask):
                lower_bound = tf.math.maximum(lower_bound, const)
            const_mask = tf.math.logical_and(
                lower_mask,
                tf.less(upper_bound, 1e9),
            )
            if(const_mask.numpy().all()):
                const = (lower_bound + upper_bound) / 2
            const_mask = tf.math.logical_not(const_mask)
            if(const_mask.numpy().all()):
                const = const * 10
        return best_attack

    def attack_step(self, x, y, modifier, const):
        x_new, grads, loss, preds, l2_dist = self.gradient(x, y, modifier, const)
        self.optimizer.apply_gradients([(grads, modifier)])
        return x_new, loss, preds, l2_dist

    @tf.function
    def gradient(self, x, y, modifier, const):
        with tf.GradientTape() as tape:
            adv_image = modifier + x
            x_new = clip_tanh(adv_image)
            preds = self.model(x_new)
            loss, l2_dist = loss_fn(
                x=x,
                x_new=x_new,
                y_true=y,
                y_pred=preds,
                confidence=self.confidence,
                const=const,
            )
        grads = tape.gradient(loss, adv_image)
        return x_new, grads, loss, preds, l2_dist


def l2(x, y):
    return tf.reduce_sum(tf.square(x - y), list(range(1, len(x.shape))))


def loss_fn(
    x,
    x_new,
    y_true,
    y_pred,
    confidence,
    const=0,
):
    other = clip_tanh(x)
    l2_dist = l2(x_new, other)
    real = tf.reduce_sum(y_true * y_pred, 1)
    other = tf.reduce_max((1.0 - y_true) * y_pred - y_true * 10000, 1)
    loss_1 = tf.maximum(0.0, other - real + confidence)
    loss_2 = tf.reduce_sum(l2_dist)
    loss_1 = tf.reduce_sum(const * loss_1)
    loss = loss_1 + loss_2
    return loss, l2_dist


def clip_tanh(x):
    return (tf.tanh(x) + 1) / 2.0

def preprocess_image(filename):
    img_width, img_height = 224, 224
    img_path = filename
    image = load_img(img_path, target_size=(img_width, img_height))
    image = img_to_array(image)/255
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = tf.convert_to_tensor(image)
    return image

def main(model,img_path,target,**kwargs):
    x = preprocess_image(img_path)
    kwargs = {
        "y": target,
        "batch_size": 1,
        "binary_search_steps": 5,
        "max_iterations": 1000,
        "abort_early": True,
        "confidence": 0.0,
        "initial_const": 1e-2,
        "learning_rate": 5e-3,
    }
    adv = carlini_wagner_l2(model, x, **kwargs)
    return adv


if __name__ == "__main__":
    main(model,img,target,**kwargs)