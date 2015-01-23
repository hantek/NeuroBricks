import numpy
from layer import LinearLayer


def test_draw_weight():
    test_model = LinearLayer(3072, 3)

    test_weight2 = numpy.ones((3072, 3))
    test_weight2[:1024, 0] = 255
    test_weight2[1025:2048, 1] = 255
    test_weight2[2049:3072, 2] = 255
    test_model.w.set_value(test_weight2.astype('float32'))

    test_model.draw_weight(patch_shape=(32,32,3))


