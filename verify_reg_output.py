#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Concatenate, Lambda, UpSampling3D
from tensorflow.keras.optimizers import Adam

MODEL_PATH = "/hpc/pahm409/harvard/UOA/3d_reg_results/checkpoints/model_epoch_050_val_loss_0.0000.h5"

def create_basic_3d_registration_model(
    fixed_input_shape=(64, 64, 64, 1),
    moving_input_shape=(64, 64, 64, 1),
    n_base_filters=8,
    depth=3
):
    fixed_input = Input(fixed_input_shape, name='fixed_input')
    moving_input = Input(moving_input_shape, name='moving_input')

    inputs = Concatenate(axis=-1)([fixed_input, moving_input])

    encoder_layers = []
    x = inputs

    # Encoder
    for i in range(depth):
        n_filters = n_base_filters * (2 ** i)
        x = Conv3D(n_filters, 3, activation='relu', padding='same')(x)
        x = Conv3D(n_filters, 3, activation='relu', padding='same')(x)
        encoder_layers.append(x)

        if i < depth - 1:
            x = Conv3D(n_filters, 3, strides=2, activation='relu', padding='same')(x)

    # Decoder
    for i in range(depth-2, -1, -1):
        n_filters = n_base_filters * (2 ** i)
        x = UpSampling3D()(x)
        x = Conv3D(n_filters, 3, activation='relu', padding='same')(x)
        x = Concatenate()([x, encoder_layers[i]])
        x = Conv3D(n_filters, 3, activation='relu', padding='same')(x)
        x = Conv3D(n_filters, 3, activation='relu', padding='same')(x)

    # Deformation field (3 channels)
    deformation_field = Conv3D(3, 3, activation='tanh', padding='same')(x)
    deformation_field = Lambda(lambda x: x * 5.0)(deformation_field)

    # FINAL OUTPUT = fixed + moving + deformation (1 + 1 + 3 = 5 channels)
    final_output = Concatenate(axis=-1)(
        [fixed_input, moving_input, deformation_field]
    )

    model = Model(inputs=[fixed_input, moving_input], outputs=final_output)

    def deformation_prediction_loss(y_true, y_pred):
        deformation = y_pred[..., -3:]
        dx = deformation[:, 1:, :, :, :] - deformation[:, :-1, :, :, :]
        dy = deformation[:, :, 1:, :, :] - deformation[:, :, :-1, :, :]
        dz = deformation[:, :, :, 1:, :] - deformation[:, :, :, :-1, :]
        smoothness = tf.reduce_mean(tf.square(dx)) + \
                     tf.reduce_mean(tf.square(dy)) + \
                     tf.reduce_mean(tf.square(dz))
        magnitude = tf.reduce_mean(tf.square(deformation))
        return 0.8 * smoothness + 0.2 * magnitude

    model.compile(optimizer=Adam(1e-4), loss=deformation_prediction_loss)
    return model


# ---------------------------------------------------
# Build + load model
# ---------------------------------------------------
model = create_basic_3d_registration_model()
model.load_weights(MODEL_PATH)

# ---------------------------------------------------
# Print shapes
# ---------------------------------------------------
print("\n=== MODEL INPUT / OUTPUT SHAPES ===")
print("Number of inputs:", len(model.inputs))

for i, inp in enumerate(model.inputs):
    print(f"Input {i} ({inp.name}):", inp.shape)

print("Output:", model.output.shape)

# ---------------------------------------------------
# Full layer summary
# ---------------------------------------------------
print("\n=== MODEL SUMMARY ===")
model.summary()

# ---------------------------------------------------
# Dummy forward pass test
# ---------------------------------------------------
fixed_dummy  = tf.random.normal((1, 64, 64, 64, 1))
moving_dummy = tf.random.normal((1, 64, 64, 64, 1))

out = model.predict([fixed_dummy, moving_dummy])
print("\nForward pass output tensor shape:", out.shape)
print("Deformation field slice shape:", out[..., -3:].shape)

