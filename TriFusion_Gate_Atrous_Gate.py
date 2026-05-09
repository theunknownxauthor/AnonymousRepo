#Three-Branch Fusion for Building Detection (Pixel, Local, Global)
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add,Input, GlobalAveragePooling2D, Dense, Multiply, Conv2D, Conv2DTranspose, Flatten, Reshape, Lambda, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import rasterio


from tensorflow.keras.initializers import HeNormal,LecunNormal

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, indices, input_data_paths, target_data_paths, patch_size, patch_size_global, bands, bands_context, batch_size):
        assert len(input_data_paths) == len(target_data_paths), "Input and target paths must be the same length"
        
        self.input_data_paths = input_data_paths
        self.target_data_paths = target_data_paths
        self.patch_size = patch_size
        self.patch_size_global = patch_size_global
        self.bands = bands
        self.bands_context = bands_context
        self.batch_size = batch_size
        self.indices = indices
        self.half_patch_size = patch_size // 2
        self.half_patch_size_global = patch_size_global // 2

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_c_batch, X_g_batch, X_p_batch, P_gt_batch = self.__data_generation(batch_indices)
        return [X_c_batch, X_g_batch, X_p_batch, P_gt_batch], [X_c_batch, P_gt_batch]

    def __data_generation(self, batch_indices):
        X_c_batch, X_g_batch, X_p_batch, P_gt_batch = [], [], [], []

        for file_idx, i, j in batch_indices:
            input_path = self.input_data_paths[file_idx]
            target_path = self.target_data_paths[file_idx]

            # Local and global windows
            window_local = rasterio.windows.Window(j - self.half_patch_size, i - self.half_patch_size,
                                                   self.patch_size, self.patch_size)
            window_global = rasterio.windows.Window(j - self.half_patch_size_global, i - self.half_patch_size_global,
                                                    self.patch_size_global, self.patch_size_global)

            # Read input patches
            with rasterio.open(input_path) as src:
                patch_local = src.read(window=window_local)  # (C, H, W)
                patch_global = src.read(window=window_global)

            # Read ground truth label
            with rasterio.open(target_path) as src_tgt:
                label = src_tgt.read(1, window=rasterio.windows.Window(j, i, 1, 1))[0, 0]
                
            label = label.astype(np.float32)

            # Format inputs
            patch_local = np.moveaxis(patch_local, 0, -1)  # (H, W, C)
            patch_global = np.moveaxis(patch_global, 0, -1)  # (H, W, C)
            if (self.bands_context==5):
                center_pixel = patch_local[self.half_patch_size, self.half_patch_size, :]  # (C,)
                min_values = np.array([0.0, 0.0, 0.0, -20.0, 0.0])
                max_values = np.array([255.0, 255.0, 255.0, 194.0, 53.0])
            if (self.bands_context==4):
                center_pixel = patch_local[self.half_patch_size, self.half_patch_size, :-1]  # (C,)
                min_values = np.array([0.0, 0.0, 0.0, -20.0])
                max_values = np.array([255.0, 255.0, 255.0, 194.0])

            if (self.bands_context==3):
                center_pixel = patch_local[self.half_patch_size, self.half_patch_size, :-2]  # (C,)
                min_values = np.array([0.0, 0.0, 0.0])
                max_values = np.array([255.0, 255.0, 255.0])

            # Context bands
            X_c = patch_local[:, :, -self.bands_context:]  # local context
            X_g = patch_global[:, :, -self.bands_context:]  # global context

            # Normalize local context
            X_c = (X_c - min_values) / (max_values - min_values + 1e-8)  # Add small epsilon for stability

            # Normalize global context
            X_g = (X_g - min_values) / (max_values - min_values + 1e-8)
            
            # Normalize pixel context
            center_pixel = (center_pixel - min_values) / (max_values - min_values + 1e-8)
            
            X_c_batch.append(X_c)
            X_g_batch.append(X_g)
            X_p_batch.append(center_pixel)
            label=label/255.0
            P_gt_batch.append(label)
            # print("label:", label)
        return (np.array(X_c_batch), np.array(X_g_batch), np.array(X_p_batch), np.array(P_gt_batch).astype(np.float32))

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VAELossLayer(tf.keras.layers.Layer):
    def __init__(self, patch_size, bands_context, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.bands_context = bands_context

    def call(self, inputs):
        xc, xc_prim, z_mean, z_log_var, y_true, p_pred = inputs
        reconstruction_loss = MeanSquaredError()(xc, xc_prim)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        prediction_loss = BinaryCrossentropy(from_logits=False)(y_true, p_pred)
        λ1 = 1.0   # reconstruction
        λ2 = 0.01  # KL loss
        λ3 = 1.0   # prediction
        total_loss = K.mean(λ1 * reconstruction_loss + λ2 * kl_loss + λ3 * prediction_loss)
        total_loss = K.mean(reconstruction_loss + kl_loss + prediction_loss)
        self.add_loss(total_loss)
        
        return total_loss

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

def residual_dense_block(x, units, activation='relu'):
    shortcut = x
    x = Dense(units, activation=activation, kernel_initializer=HeNormal())(x)
    x = Dense(units, activation=activation, kernel_initializer=HeNormal())(x)
    return Add()([x, shortcut])

def create_TriFusion(patch_size, patch_size_global, latent_dim, bands, atrous):     

    # VAE Encoder
    xc = Input(shape=(patch_size, patch_size, bands), name='input_xc')
    xg = Input(shape=(patch_size_global, patch_size_global, bands), name='input_xg')
    x = conv_block(xc, 16)  # Reduced filters from 16 to 8
    x = residual_block(x, 32)  # Reduced filters and simplified residual block

    x = Flatten(name='encoder_flatten')(x)
    x = Dense(32, activation='relu', kernel_initializer=HeNormal(), name='encoder_dense' )(x)  # Reduced Dense layer size
    z_mean = Dense(latent_dim, kernel_initializer=HeNormal(), name='z_mean')(x)
    z_log_var = Dense(latent_dim, kernel_initializer=HeNormal(), name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='latent_sampling')([z_mean, z_log_var])

    # VAE Decoder
    decoder_input = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(32, activation='relu', kernel_initializer=HeNormal(), name='decoder_dense1')(decoder_input)  # Reduced Dense layer size
    x = Dense(patch_size * patch_size * 32, activation='relu', kernel_initializer=HeNormal(), name='decoder_dense2')(x)  # Reduced size
    x = Reshape((patch_size, patch_size, 32), name='decoder_reshape')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal(), name='decoder_deconv1')(x)  # Reduced filters
    x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal(), name='decoder_deconv2')(x)  # Reduced filters
    xc_prim = Conv2DTranspose(bands, (3, 3), activation='sigmoid', padding='same', kernel_initializer=LecunNormal(), name='decoder_output')(x)

    # VAE Models
    encoder = Model(xc, [z_mean, z_log_var, z], name='vae_encoder')
    decoder = Model(decoder_input, xc_prim, name='vae_decoder')
    vae_output = decoder(encoder(xc)[2])

    # Prediction Pipeline
    input_xp = Input(shape=(1, 1, bands), name='input_xp')

    # Flatten z and apply gating mechanism G_m
    z_flattened = Flatten()(z)
    gate_weights = Dense(latent_dim, activation='sigmoid', kernel_initializer=HeNormal(), name='G_m')(z_flattened)
    z_modulated = Multiply()([z_flattened, gate_weights])
    z_reshaped = Reshape((1, 1, latent_dim))(z_modulated)

    if (atrous==1):
       # Atrous convolution branch on global input
        atrous_rates = [1, 3, 11, 17]
        atrous_features = []
    
        for rate in atrous_rates:
            feat = Conv2D(8, (3, 3), padding='same', dilation_rate=rate, activation='relu',
                          kernel_initializer=HeNormal(), name=f'atrous_conv_r{rate}')(xg)
            feat = GlobalAveragePooling2D()(feat)  # Reduce spatial dimension
            atrous_features.append(feat)
    
        xg_combined = concatenate(atrous_features, axis=-1)
        
        # Apply gating mechanism G_g
        gate_global = Dense(xg_combined.shape[-1], activation='sigmoid', kernel_initializer=HeNormal(), name='G_g')(xg_combined)
        xg_modulated = Multiply()([xg_combined, gate_global])  # Element-wise modulation
        xg_reshaped = Reshape((1, 1, -1))(xg_modulated)  # Final shape ready for concat

        concatenated = concatenate([tf.reshape(input_xp, (-1, 1, 1, input_xp.shape[-1])), z_reshaped, xg_reshaped], axis=-1)
    else:
        concatenated = concatenate([tf.reshape(input_xp, (-1, 1, 1, input_xp.shape[-1])), z_reshaped], axis=-1)
    # Apply fewer convolutional layers
    x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal())(concatenated)  # Reduced filters
    x = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal())(x)  # Reduced filters

    # Global Average Pooling to reduce to scalar
    x = GlobalAveragePooling2D()(x)

    # Output scalar for p_pred
    p_pred = Dense(1, activation='sigmoid', kernel_initializer=HeNormal(), name='building_prediction')(x)

    # Define models
    y_true = Input(shape=(1,), name='true_building')
    vae_loss_layer = VAELossLayer(patch_size, bands, name='vae_loss')(
        [xc, vae_output, z_mean, z_log_var, y_true, p_pred]
    )

    TriFusion_model = Model(inputs=[xc, xg, input_xp, y_true], outputs=[vae_output, p_pred], name='TriFusion')
    TriFusion_model.add_loss(vae_loss_layer)

    # Define the learning rate
    learning_rate = 0.001

    # Instantiate the optimizer with the custom learning rate
    optimizer = Adam(learning_rate=learning_rate)

    # Compile the model with the custom optimizer
    TriFusion_model.compile(optimizer=optimizer, loss=None)

    return TriFusion_model
     
def get_all_indices(ratio_dataset, input_data, patch_size, bands):
    half_patch_size = patch_size // 2
    _, height, width = input_data.shape[2], input_data.shape[0], input_data.shape[1]
    
    if height > patch_size and width > patch_size:
        all_indices = [(i, j) for i in range(half_patch_size, height - half_patch_size)
                               for j in range(half_patch_size, width - half_patch_size)]
    else:
        raise ValueError("Patch size is too large for the input data dimensions.")
   
    
    return all_indices, height, width
