#generator class
class Generator(tf.keras.Model):
    def __init__(self, z_dim=100):
        super().__init__()
        self.model = tf.keras.Sequential([
            layers.Dense(256, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(1024, activation="relu"),
            layers.Dense(3 * 64 * 64, activation="tanh"),
            layers.Reshape((64, 64, 3))
        ])

    def call(self, z):
        return self.model(z)
    
#discriminator class
#discriminator
class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(1024),
            layers.LeakyReLU(0.2),
            layers.Dense(512),
            layers.LeakyReLU(0.2),
            layers.Dense(256),
            layers.LeakyReLU(0.2),
            layers.Dense(1, activation="sigmoid")
        ])

    def call(self, img):
        return self.model(img)