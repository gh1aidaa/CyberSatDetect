import keras
from keras.layers import Dense


orig_dense_from_config = Dense.from_config.__func__

@classmethod
def dense_from_config(cls, config):
    if isinstance(config, dict):
        config = dict(config)
        config.pop("quantization_config", None)
    return orig_dense_from_config(cls, config)

Dense.from_config = dense_from_config

keras.config.enable_unsafe_deserialization()

model = keras.models.load_model(
    "backend/app/best_model_qc_filtered.keras",
    compile=False,
    safe_mode=False,
)

model.save("backend/app/best_model_render.keras")

print("DONE")