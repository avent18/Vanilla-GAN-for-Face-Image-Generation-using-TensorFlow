#traning fnct
@tf.function
def train_step(real_images, batch_size, z_dim=100):

    noise = tf.random.normal([batch_size, z_dim])

    valid = tf.ones((batch_size, 1))
    fake = tf.zeros((batch_size, 1))

    # ---------------------
    # Train Discriminator
    # ---------------------
    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        real_loss = cross_entropy(valid, real_output)
        fake_loss = cross_entropy(fake, fake_output)
        d_loss = (real_loss + fake_loss) / 2

    grads_D = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(
        zip(grads_D, discriminator.trainable_variables)
    )

    # ---------------------
    # Train Generator
    # ---------------------
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        g_loss = cross_entropy(valid, fake_output)

    grads_G = gen_tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(
        zip(grads_G, generator.trainable_variables)
    )

    return d_loss, g_loss
