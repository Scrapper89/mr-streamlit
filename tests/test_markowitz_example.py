import markowitz_example as mw

class TestMarkowitz:
    def test_rand_weights(self):
        assert mw.rand_weights(1)
