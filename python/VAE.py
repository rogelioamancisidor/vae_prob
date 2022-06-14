import numpy as np
import tensorflow as tf 

import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


class VAE(tfk.Model):
    def __init__(self,
                 pz,
                 encoder,
                 decoder,
                 latent_size,
                 name = 'VAEprob',
                 **kwargs):
        super(VAE,self).__init__(name=name, **kwargs)
        self.pz = pz
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size
        self.params = encoder.trainable_variables + decoder.trainable_variables

    def call(self, inputs, L=1):
        qz_x = self.encoder(inputs)
        zs  = qz_x.sample(L)
        for i in range(zs.shape[0]):
            z = zs[i,...] 
            px_z = self.decoder(z)
            
            pz = self.pz(loc=tf.zeros_like(qz_x.mean()), scale_diag=tf.ones_like(qz_x.mean()))
           
            # pz, qz_x, and px_z are all pdf's!
            elbo = px_z.log_prob(inputs) - tfd.kl_divergence(qz_x,pz) 
            
            self.loss = -tf.reduce_mean(elbo)

        return qz_x, px_z

    # use posterior to generate. returns a density function 
    def reconstruct(self,x):
        qz_x = self.encoder(x)
        z   = qz_x.mean()
        
        px_z = self.decoder(z)
        return px_z 
    
    # use prior to generate. returns a density function 
    def generate(self, L=10):
        pz = self.pz(loc=0, scale_diag=tf.ones(self.latent_size))
        z = pz.sample(L) 
        px_z = self.decoder(z)
        
        return px_z 

    @tf.function
    def train(self,x, optimizer, L=1):
        with tf.GradientTape() as tape:
            enc_dec = self.call(x, L=L)
        gradients = tape.gradient(self.loss, self.params)
        optimizer.apply_gradients(zip(gradients, self.params))
        
        return enc_dec, self.loss

