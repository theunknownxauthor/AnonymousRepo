import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import concatenate

from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.initializers import LecunNormal

from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Activation
from config import *


def create_backbone( patch_size, patch_size_global, latent_dim, bands_context, bands, mask_dim=6):
    _, _, backbone, _ = create_GeoKRL( patch_size=patch_size, patch_size_global=patch_size_global, latent_dim=latent_dim,
                                            bands_context=bands_context, bands=bands, mask_dim=mask_dim )
    return backbone

def load_pretrained_backbone( weights_file, patch_size, patch_size_global, latent_dim, bands_context, bands, mask_dim=6):

    encoder, decoder, backbone, pretraining_model = create_GeoKRL( patch_size=patch_size, patch_size_global=patch_size_global,
                                                                    latent_dim=latent_dim, bands_context=bands_context, 
                                                                    bands=bands, mask_dim=mask_dim )
    pretraining_model.load_weights(weights_file)
    print()
    print("Pretrained weights loaded:")
    print(weights_file)
    return backbone
    
def create_task_model( backbone, task, freeze_backbone=True, bottleneck_ratio=4, n_blocks=2, dropout_rate=0.2):

    # ======================================================
    # Freeze / Unfreeze Backbone
    # ======================================================
    for layer in backbone.layers:
        layer.trainable = not freeze_backbone
    # ======================================================
    # Backbone Representation
    # ======================================================
    representation = backbone.output
    # ======================================================
    # GPS Input
    # ======================================================
    gps_input = Input( shape=(4,), name="input_gps")
    # ======================================================
    # Prediction Head
    # ======================================================
    prediction = create_prediction_head( representation=representation, gps=gps_input, task=task,
                                            bottleneck_ratio=bottleneck_ratio, n_blocks=n_blocks, dropout_rate=dropout_rate )
    # ======================================================
    # Complete Model
    # ======================================================
    model = Model( inputs=backbone.inputs + [gps_input], outputs=prediction, name=f"GeoKRL_{task}")
    return model

def create_prediction_head( representation, gps, task, bottleneck_ratio=4, n_blocks=2, dropout_rate=0.2):
    # ======================================================
    # GPS Embedding
    # ======================================================
    gps = Dense( 16, kernel_initializer=HeNormal(), name=f"{task}_gps_dense" )(gps)
    gps = Activation( "gelu", name=f"{task}_gps_gelu" )(gps)
    gps = LayerNormalization( name=f"{task}_gps_norm" )(gps)
    # ======================================================
    # Representation + GPS Fusion
    # ======================================================
    x = concatenate( [representation, gps], name=f"{task}_representation_with_gps" )
    d = K.int_shape(x)[-1]
    # ======================================================
    # Initial Layer Normalization
    # ======================================================
    x = LayerNormalization( name=f"{task}_neck_norm" )(x)
    # ======================================================
    # Residual Adapter Neck
    # ======================================================
    shortcut = x
    x = Dense( max(d // bottleneck_ratio, 8), kernel_initializer=HeNormal(), name=f"{task}_neck_dense1" )(x)
    x = Activation( "gelu", name=f"{task}_neck_gelu" )(x)
    x = Dense( d, kernel_initializer=HeNormal(), name=f"{task}_neck_dense2" )(x)
    x = Add( name=f"{task}_neck_add" )([shortcut, x])
    # ======================================================
    # Residual Prediction Blocks
    # ======================================================
    for block in range(n_blocks):
        shortcut = x
        y = Dense( d, kernel_initializer=HeNormal(), name=f"{task}_block{block+1}_dense1")(x)
        y = Activation( "gelu", name=f"{task}_block{block+1}_gelu" )(y)
        y = Dropout( dropout_rate, name=f"{task}_block{block+1}_dropout" )(y)
        y = Dense( d, kernel_initializer=HeNormal(), name=f"{task}_block{block+1}_dense2" )(y)
        x = Add( name=f"{task}_block{block+1}_add" )([shortcut, y])
    # ======================================================
    # Final Normalization
    # ======================================================
    x = LayerNormalization( name=f"{task}_head_norm" )(x)
    x = Dropout( dropout_rate, name=f"{task}_head_dropout")(x)
    prediction = Dense( 1, activation=None, kernel_initializer=LecunNormal(), name=f"{task}_logit" )(x)
    # ======================================================
    # Task-specific Output
    # ======================================================
    if task == TASK_POPULATION:
        prediction = Activation( "softplus", name="population_output" )(prediction)
    elif task == TASK_BIOMASS:
        prediction = Activation( "softplus", name="biomass_output")(prediction)
    elif task == TASK_BUILDING:
        prediction = Activation( "sigmoid", name="building_output" )(prediction)
    else:
        raise ValueError(f"Unknown task: {task}")
    return prediction
    
class Sampling(tf.keras.layers.Layer):

    def __init__(self, seed=1234, **kwargs):
        super().__init__(**kwargs)
        self.generator = tf.random.Generator.from_seed(seed)

    def call(self, inputs, training=None):
        z_mean, z_log_var = inputs

        if training:
            epsilon = self.generator.normal(tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        return z_mean 
    
def conv_block(x, filters, kernel_size=3, padding='same'):
    x = Conv2D(filters, kernel_size, padding=padding, activation='relu', kernel_initializer=HeNormal())(x)
    return x

def residual_block(x, filters):
    shortcut = x
    
    x = conv_block(x, filters)
    x = conv_block(x, filters)
    
    # Match dimensions if needed (not required here as we use 'same' padding)
    if K.int_shape(shortcut)[-1] != K.int_shape(x)[-1]:
        x = Conv2D(K.int_shape(shortcut)[-1], (1, 1), padding='same', kernel_initializer=HeNormal())(x)
    
    x = Add()([x, shortcut])
    return x

def create_GeoKRL(patch_size, patch_size_global, latent_dim, bands_context, bands, mask_dim=6):

    # Inputs
    xc = Input(shape=(patch_size, patch_size, bands_context), name='input_xc')
    xg = Input(shape=(patch_size_global, patch_size_global, bands_context), name='input_xg')
    input_xp = Input(shape=(1, 1, bands), name='input_xp')
    modality_mask = Input(shape=(mask_dim,), name='modality_mask')

    # =========================
    # VAE Encoder
    # =========================

    x = conv_block(xc, 16)
    x = residual_block(x, 32)

    x = Flatten(name='encoder_flatten')(x)

    x = Dense(32, activation='relu', kernel_initializer=HeNormal(), name='encoder_dense')(x)

    z_mean = Dense(latent_dim, kernel_initializer=HeNormal(), name='z_mean')(x)
    z_log_var = Dense(latent_dim, kernel_initializer=HeNormal(), name='z_log_var')(x)

    # z = Lambda(sampling, output_shape=(latent_dim,), name='latent_sampling')([z_mean, z_log_var])
    z = Sampling(name="latent_sampling")([z_mean, z_log_var])
    # z = z_mean
    # =========================
    # VAE Decoder
    # =========================

    decoder_input = Input(shape=(latent_dim,), name='decoder_input')

    d = Dense(32, activation='relu', kernel_initializer=HeNormal(), name='decoder_dense1')(decoder_input)

    d = Dense(patch_size * patch_size * 32, activation='relu', kernel_initializer=HeNormal(), name='decoder_dense2')(d)

    d = Reshape((patch_size, patch_size, 32), name='decoder_reshape')(d)

    d = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal(), name='decoder_deconv1')(d)

    d = Conv2DTranspose(16, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal(), name='decoder_deconv2')(d)

    xc_prim = Conv2DTranspose(bands_context, (3, 3), activation='sigmoid', padding='same', kernel_initializer=LecunNormal(), name='decoder_output')(d)

    encoder = Model(xc, [z_mean, z_log_var, z], name='vae_encoder')

    decoder = Model(decoder_input, xc_prim, name='vae_decoder')

    vae_output = decoder(encoder(xc)[2])

    # =========================
    # Medium Context Gate Gm
    # =========================

    z_flattened = Flatten(name='z_flattened')(z)

    gate_weights = Dense(latent_dim, activation='sigmoid', kernel_initializer=HeNormal(), name='G_m')(z_flattened)

    z_modulated = Multiply(name='z_modulated')([z_flattened, gate_weights])

    z_reshaped = Reshape((1, 1, latent_dim), name='z_reshaped')(z_modulated)

    # =========================
    # Global Context Branch
    # =========================

    atrous_features = []

    for rate in [1, 3, 11, 17]:

        feat = Conv2D(8, (3, 3), dilation_rate=rate, padding='same', activation='relu', kernel_initializer=HeNormal(), name=f'atrous_conv_r{rate}')(xg)

        feat = GlobalAveragePooling2D(name=f'gap_r{rate}')(feat)

        atrous_features.append(feat)

    xg_combined = concatenate(atrous_features, axis=-1, name='global_concat')

    gate_global = Dense(K.int_shape(xg_combined)[-1], activation='sigmoid', kernel_initializer=HeNormal(), name='G_g')(xg_combined)

    xg_modulated = Multiply(name='global_modulated')([xg_combined, gate_global])

    xg_reshaped = Reshape((1, 1, K.int_shape(xg_combined)[-1]), name='global_reshaped')(xg_modulated)

    # =========================
    # Representation Vector
    # =========================

    xp_flat = Flatten(name='xp_flat')(input_xp)

    z_flat = Flatten(name='z_flat')(z_reshaped)

    xg_flat = Flatten(name='xg_flat')(xg_reshaped)

    representation = concatenate([xp_flat, z_flat, xg_flat], axis=-1, name='representation')
    # representation =  z_flat

    # =========================
    # Modality Mask Embedding
    # =========================

    # mask_embedding = Dense(16, activation='relu', kernel_initializer=HeNormal(), name='mask_embedding')(modality_mask)

    # representation_with_mask = concatenate([representation, mask_embedding], axis=-1, name='representation_with_mask')
    representation_with_mask = representation
    # =========================
    # Task Classification Head
    # =========================

    h = Dense(128, activation='relu', kernel_initializer=HeNormal(), name='task_dense1')(representation_with_mask)

    h = Dropout(0.3, name='task_dropout')(h)

    h = Dense(64, activation='relu', kernel_initializer=HeNormal(), name='task_dense2')(h)

    task_output = Dense(3, activation='softmax', name='task_classifier')(h)

    # =========================
    # Models
    # =========================

    # backbone = Model(inputs=[input_xp, xc, xg, modality_mask], outputs=representation_with_mask, name='GeoKRL_Backbone')
    backbone = Model(inputs=[input_xp, xc, xg], outputs=representation_with_mask, name='GeoKRL_Backbone')

    # pretraining_model = Model(inputs=[input_xp, xc, xg, modality_mask], outputs=[task_output, vae_output, z_mean, z_log_var], 
    # name='GeoKRL_Pretraining')
    pretraining_model = Model(inputs=[input_xp, xc, xg], outputs=[task_output, vae_output, z_mean, z_log_var], 
    name='GeoKRL_Pretraining')

    return encoder, decoder, backbone, pretraining_model
    
def main():

    patch_size = 11
    patch_size_global = 21
    latent_dim = 20
    bands_context = 10
    bands = 10
    mask_dim = 6

    encoder, decoder, backbone, pretraining_model = create_GeoKRL(
        patch_size,
        patch_size_global,
        latent_dim,
        bands_context,
        bands,
        mask_dim
    )

    print("\n========== ENCODER ==========")
    encoder.summary()

    print("\n========== DECODER ==========")
    decoder.summary()

    print("\n========== BACKBONE ==========")
    backbone.summary()

    print("\n========== PRETRAINING MODEL ==========")
    pretraining_model.summary()

    batch_size = 8

    dummy_xp = np.random.rand(batch_size, 1, 1, bands).astype(np.float32)

    dummy_xc = np.random.rand(
        batch_size,
        patch_size,
        patch_size,
        bands_context
    ).astype(np.float32)

    dummy_xg = np.random.rand(
        batch_size,
        patch_size_global,
        patch_size_global,
        bands_context
    ).astype(np.float32)

    dummy_mask = np.random.randint(
        0,
        2,
        size=(batch_size, mask_dim)
    ).astype(np.float32)

    print("\n========== FORWARD PASS ==========")

    representation = backbone.predict(
        [dummy_xp, dummy_xc, dummy_xg, dummy_mask],
        verbose=0
    )

    print("Backbone output shape:", representation.shape)

    task_pred, reconstructed = pretraining_model.predict(
        [dummy_xp, dummy_xc, dummy_xg, dummy_mask],
        verbose=0
    )

    print("Task prediction shape:", task_pred.shape)
    print("Reconstruction shape:", reconstructed.shape)

    print("\nInference successful.")


if __name__ == "__main__":
    main()
    
"""
The modality mask is transformed into a learnable modality embedding that provides the backbone with explicit information regarding the ancillary modality configuration used to generate each training sample.
"""