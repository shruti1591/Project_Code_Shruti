
import os
import numpy as np
import tensorflow as tf
from SF0 import*
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)



class KimiDeltaAttention(layers.Layer):
    def __init__(self,embed_dim=128,num_heads=4,**kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads")
        self.head_dim = embed_dim // num_heads
        self.q_proj = layers.Dense(embed_dim,name="q_projection")
        self.k_proj = layers.Dense(embed_dim,name="k_projection")
        self.v_proj = layers.Dense(embed_dim,name="v_projection")
        self.alpha_proj = layers.Dense(embed_dim,name="alpha_projection")
        self.beta_proj = layers.Dense(embed_dim,name="beta_projection")
        self.out_proj = layers.Dense(embed_dim,name="kda_output_projection")

    def call(self, x):
        B = tf.shape(x)[0]
        T = self.sequence_length
        H = self.num_heads
        K = self.head_dim
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = tf.reshape(q,[B, T, H, K])
        k = tf.reshape(k,[B, T, H, K])
        v = tf.reshape(v,[B, T, H, K])
        q = tf.transpose(q,[0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        alpha = tf.nn.sigmoid(self.alpha_proj(x))
        beta = tf.nn.sigmoid(self.beta_proj(x))
        alpha = tf.reshape(alpha,[B, T, H, K])
        beta = tf.reshape(beta,[B, T, H, K])
        alpha = tf.transpose(alpha,[0, 2, 1, 3])
        beta = tf.transpose(beta,[0, 2, 1, 3])
        q = tf.math.l2_normalize(q,axis=-1)
        k = tf.math.l2_normalize(k,axis=-1)
        S = tf.zeros([B, H, K, K],dtype=x.dtype)
        outputs = []
        I = tf.eye(K,dtype=x.dtype)
        I = I[None, None, :, :]
        for t in range(self.sequence_length):
            q_t = q[:, :, t, :]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            alpha_t = alpha[:, :, t, :]
            beta_t = beta[:, :, t, :]
            q_t = tf.expand_dims(q_t,-1)
            k_t = tf.expand_dims(k_t,-1)
            v_t = tf.expand_dims(v_t,-1)
            alpha_t = tf.expand_dims(alpha_t,-1)
            beta_t = tf.expand_dims(beta_t,-1)
            S = alpha_t * S
            kkT = tf.matmul(k_t,k_t,transpose_b=True)
            I_minus_beta_kkT = ( I -beta_t * kkT)
            S = tf.matmul(I_minus_beta_kkT,S)
            S = (S +beta_t *tf.matmul(k_t,v_t,transpose_b=True))
            o_t = tf.matmul(S,q_t,transpose_a=True)
            o_t = tf.squeeze(o_t,axis=-1)
            outputs.append(o_t)
        out = tf.stack(outputs,axis=2)
        out = tf.transpose(out,[0, 2, 1, 3])
        out = tf.reshape(out,[B, T, self.embed_dim])
        return self.out_proj(out)

    def build(self, input_shape):
        self.sequence_length = input_shape[1]
        super().build(input_shape)


# ============================================================
# 2. LIQUID NEURAL NETWORK CELL
# ============================================================

class LiquidCell(layers.Layer):
    def __init__(self,hidden_dim,dt=0.1,**kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.w_in = layers.Dense(hidden_dim,name="liquid_input")
        self.w_h = layers.Dense( hidden_dim,name="liquid_recurrent")

    def build(self, input_shape):
        self.tau = self.add_weight(shape=(self.hidden_dim,),initializer="ones",
            trainable=True,name="tau")
        super().build(input_shape)
    def call(self,x,h_prev):
        tau = (tf.nn.softplus(self.tau)+ 1e-3)
        f = tf.nn.tanh(self.w_in(x)+self.w_h(h_prev))
        dh = ( -h_prev / tau+f)
        h_new = (h_prev+self.dt * dh)
        return h_new
    
class LiteKDCNet_text(Model):
    def __init__(self,input_shape, num_classes,hidden_dim=128,proj_dim=64,
        num_heads=4,sequence_length=32, dropout=0.3,**kwargs):
        super().__init__(
            name="LiteKDC-Net",
            **kwargs
        )
        self.original_input_shape = input_shape
        self.original_input_dim = input_shape[-1]
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        if self.original_input_dim % sequence_length != 0:
            raise ValueError(
                f"Input dimension {self.original_input_dim} "
                f"cannot be divided into sequence length "
                f"{sequence_length}."
            )

        self.features_per_step = (self.original_input_dim // sequence_length)
        self.sequence_projection = layers.Dense(sequence_length * self.features_per_step,
            activation="relu",name="feature_sequence_projection")
        self.input_projection = layers.Dense(hidden_dim, activation="relu",
            name="audio_feature_projection")
        self.liquid_cell = LiquidCell(hidden_dim=hidden_dim,dt=0.1,name="liquid_cell")
        self.kda = KimiDeltaAttention(embed_dim=hidden_dim,num_heads=num_heads,
            name="kimi_delta_attention")
        self.norm = layers.LayerNormalization(name="kda_normalization")
        self.representation = layers.Dense( hidden_dim,activation="gelu",name="kda_representation")
        self.dropout = layers.Dropout(
            dropout,
            name="representation_dropout"
        )

     
        self.classifier = layers.Dense(num_classes,activation="softmax",name="classifier")
        self.contrastive_head = tf.keras.Sequential([layers.Dense( hidden_dim,activation="relu"
                ),layers.Dense(proj_dim)], name="contrastive_head")

    def call(self,x,training=False):

        x = self.sequence_projection(x)
        x = tf.reshape(x,[tf.shape(x)[0],self.sequence_length,self.features_per_step])
        x = self.input_projection(x)
        h = tf.zeros([tf.shape(x)[0],self.hidden_dim],dtype=x.dtype)
        liquid_outputs = []
        for t in range(self.sequence_length):
            x_t = x[:, t, :]
            h = self.liquid_cell(x_t,h)
            liquid_outputs.append(h)
        liquid_sequence = tf.stack(liquid_outputs, axis=1)
        kda_output = self.kda(liquid_sequence)
        kda_output = self.norm(liquid_sequence + kda_output)
        representation = tf.reduce_mean(kda_output,axis=1)
        representation = self.representation( representation)
        representation = self.dropout(representation,training=training)
        logits = self.classifier(representation)
        projections = self.contrastive_head(representation,training=training)
        projections = tf.math.l2_normalize(projections,axis=1)
        return logits, projections

# ============================================================
# 3. LiteKDC-Net
# ============================================================
class LiteKDCNetDual(Model):

    def __init__(self,text_input_shape=None, audio_input_shape=None,text_num_classes=None,
        audio_num_classes=None,hidden_dim=128, proj_dim=64,num_heads=4,text_sequence_length=32,
        dropout=0.3,**kwargs):
        super().__init__(name="LiteKDC-Net-Dual",**kwargs)
        self.text_input_shape = text_input_shape
        self.audio_input_shape = audio_input_shape
        self.text_num_classes = text_num_classes
        self.audio_num_classes = audio_num_classes
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.text_sequence_length = text_sequence_length
        self.dropout_rate = dropout
        if text_input_shape is not None:
            self.text_input_dim = text_input_shape[-1]
            if (self.text_input_dim % text_sequence_length != 0):
                raise ValueError(
                    f"Text input dimension "
                    f"{self.text_input_dim} cannot be "
                    f"divided by text sequence length "
                    f"{text_sequence_length}"
                )

            self.text_features_per_step = (self.text_input_dim //text_sequence_length)
            self.text_sequence_projection = layers.Dense(self.text_sequence_length *
                self.text_features_per_step, activation="relu",
                name="text_sequence_projection")
            self.text_projection = layers.Dense(hidden_dim,activation="relu",
                name="text_feature_projection")
            self.text_liquid = LiquidCell(hidden_dim=hidden_dim,
                dt=0.1,name="text_liquid_cell" )
            self.text_kda = KimiDeltaAttention(embed_dim=hidden_dim,num_heads=num_heads,
                name="text_kimi_delta_attention"
            )
            self.text_norm = layers.LayerNormalization(
                name="text_kda_normalization")
            self.text_representation = layers.Dense(hidden_dim,
                activation="gelu",name="text_representation")
            self.text_dropout = layers.Dropout(dropout,name="text_dropout")
            self.text_classifier = layers.Dense(text_num_classes, activation="softmax",
                name="text_classifier")
            self.text_contrastive_head = tf.keras.Sequential([layers.Dense(hidden_dim,
                        activation="relu" ),layers.Dense(proj_dim)],
                name="text_contrastive_head")

        # ====================================================
        # AUDIO BRANCH
        # ====================================================

        if audio_input_shape is not None:
            self.audio_sequence_length = (audio_input_shape[0])
            self.audio_feature_dim = (audio_input_shape[1])
            self.audio_projection = layers.Dense(hidden_dim,activation="relu",
                name="audio_feature_projection")
            self.audio_liquid = LiquidCell(hidden_dim=hidden_dim,dt=0.1,
                name="audio_liquid_cell")
            self.audio_kda = KimiDeltaAttention(
                embed_dim=hidden_dim,num_heads=num_heads,
                name="audio_kimi_delta_attention")
            self.audio_norm = layers.LayerNormalization(
                name="audio_kda_normalization")
            self.audio_representation = layers.Dense(hidden_dim, activation="gelu",
                name="audio_representation")
            self.audio_dropout = layers.Dropout(
                dropout,
                name="audio_dropout")
            self.audio_classifier = layers.Dense(audio_num_classes,activation="softmax",
                name="audio_classifier")
            self.audio_contrastive_head = tf.keras.Sequential([layers.Dense(hidden_dim,
                        activation="relu"),
                    layers.Dense(
                        proj_dim)], name="audio_contrastive_head")

 
    def forward_text(self,text,training=False):
        x= self.text_sequence_projection(text)
        x = tf.reshape(x, [ tf.shape(x)[0],
                self.text_sequence_length,
                self.text_features_per_step])
        x = self.text_projection(x)
        h = tf.zeros(
            [tf.shape(x)[0],self.hidden_dim],dtype=x.dtype)
        outputs = []
        for t in range(self.text_sequence_length):
            h = self.text_liquid(x[:, t, :],h)
            outputs.append(h)
        liquid_sequence = tf.stack(outputs,axis=1 )
        kda_output = self.text_kda(liquid_sequence)
        x = self.text_norm(liquid_sequence +kda_output)
        representation = tf.reduce_mean(x,axis=1)
        representation = self.text_representation(representation)
        representation = self.text_dropout(representation,training=training)
        logits = self.text_classifier(representation)
        projection = self.text_contrastive_head(representation,
            training=training)
        projection = tf.math.l2_normalize(projection,axis=1)
        return logits, projection


    def forward_audio(self,audio,training=False):
        x = self.audio_projection(audio)
        h = tf.zeros([tf.shape(x)[0],self.hidden_dim],dtype=x.dtype)
        outputs = []
        for t in range(
            self.audio_sequence_length):
            h = self.audio_liquid(x[:, t, :], h)
            outputs.append(h)
        liquid_sequence = tf.stack(outputs,axis=1)
        kda_output = self.audio_kda(liquid_sequence)
        x = self.audio_norm(liquid_sequence + kda_output )
        representation = tf.reduce_mean(x,axis=1)
        representation = self.audio_representation(representation)
        representation = self.audio_dropout( representation,training=training)
        logits = self.audio_classifier(representation)
        projection = self.audio_contrastive_head(representation,training=training)
        projection = tf.math.l2_normalize(projection,axis=1)
        return logits, projection


    def call(self,
        text=None,audio=None,training=False):
        if text is None and audio is None:
            raise ValueError("Provide text or audio input." )
        text_output = None
        audio_output = None
        if text is not None:
            text_output = self.forward_text(text,training=training)
        if audio is not None:
            audio_output = self.forward_audio(audio,training=training)
        return {
            "text": text_output,
            "audio": audio_output}


class SupConLoss:
    def __init__(self,temperature=0.07):
        self.temperature = temperature
    def __call__(self,labels,features):
        labels = tf.cast(tf.reshape(labels, [-1]),tf.int32)
        features = tf.math.l2_normalize(features,axis=1)
        similarity = tf.matmul(features,features,transpose_b=True)
        logits = (similarity /self.temperature)
        logits = (logits -tf.reduce_max(logits,axis=1,keepdims=True))
        batch_size = tf.shape(features)[0]

        labels_i = tf.expand_dims(labels,axis=1)
        labels_j = tf.expand_dims(labels,axis=0)
        positive_mask = tf.cast(tf.equal(labels_i,labels_j),tf.float32)
        identity = tf.eye(batch_size)
        positive_mask = (positive_mask -identity)

        logits_mask = (1.0 - identity)
        exp_logits = (tf.exp(logits) *logits_mask)
        log_prob = (logits -tf.math.log(tf.reduce_sum(exp_logits,axis=1, keepdims=True)
                +1e-8))
        positive_count = tf.reduce_sum(positive_mask,axis=1)
        mean_log_prob_pos = (tf.reduce_sum( positive_mask * log_prob,axis=1)
            / tf.maximum(positive_count,1.0))
        valid = tf.cast(positive_count > 0,tf.float32)
        loss = (-tf.reduce_sum(mean_log_prob_pos *valid)/tf.maximum(tf.reduce_sum(valid),
                1.0))
        return loss


# ============================================================
# 5. TRAINING FUNCTION
# ============================================================
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# 5. TRAINING FUNCTION
# ============================================================

import numpy as np
import tensorflow as tf


def train_dual_litekdc(
    model,
    X_text_train=None,
    y_text_train=None,
    X_audio_train=None,
    y_audio_train=None,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    contrastive_weight=0.1,
    temperature=0.07,
):
    # ========================================================
    # CHECK INPUTS
    # ========================================================
    if X_text_train is None and X_audio_train is None:
        raise ValueError(
            "At least text or audio data must be provided."
        )

    # ========================================================
    # PREPARE TEXT DATA
    # ========================================================
    text_dataset = None

    if X_text_train is not None:

        X_text_train = np.asarray(
            X_text_train,
            dtype=np.float32
        )

        y_text_train = np.asarray(
            y_text_train,
            dtype=np.int32
        )

        if len(X_text_train) != len(y_text_train):
            raise ValueError(
                "Text X and y have different number of samples."
            )

        text_dataset = (
            tf.data.Dataset
            .from_tensor_slices(
                (
                    X_text_train,
                    y_text_train
                )
            )
            .shuffle(
                len(X_text_train)
            )
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    # ========================================================
    # PREPARE AUDIO DATA
    # ========================================================
    audio_dataset = None

    if X_audio_train is not None:

        X_audio_train = np.asarray(
            X_audio_train,
            dtype=np.float32
        )

        y_audio_train = np.asarray(
            y_audio_train,
            dtype=np.int32
        )

        if len(X_audio_train) != len(y_audio_train):
            raise ValueError(
                "Audio X and y have different number of samples."
            )

        audio_dataset = (
            tf.data.Dataset
            .from_tensor_slices(
                (
                    X_audio_train,
                    y_audio_train
                )
            )
            .shuffle(
                len(X_audio_train)
            )
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    # ========================================================
    # LOSS FUNCTIONS
    # ========================================================
    ce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    supcon_loss_fn = SupConLoss(
        temperature=temperature
    )


    text_optimizer = SFO(
        learning_rate=lr
    )

    audio_optimizer = SFO(
        learning_rate=lr
    )



    if X_text_train is not None:

        _ = model(
            text=tf.convert_to_tensor(
                X_text_train[:1]
            ),
            training=False
        )

    if X_audio_train is not None:

        _ = model(
            audio=tf.convert_to_tensor(
                X_audio_train[:1]
            ),
            training=False
        )

    # ========================================================
    # GET TEXT VARIABLES
    # ========================================================
    text_variables = [
        var
        for var in model.trainable_variables
        if (
            var.name.startswith("text_")
            or "text_" in var.path
        )
    ]

    # ========================================================
    # GET AUDIO VARIABLES
    # ========================================================
    audio_variables = [
        var
        for var in model.trainable_variables
        if (
            var.name.startswith("audio_")
            or "audio_" in var.path
        )
    ]

    # ========================================================
    # DEBUG INFORMATION
    # ========================================================
    print("\n")
    print("=" * 60)
    print("          LiteKDC-Net Dual Training")
    print("=" * 60)

    print(
        "Text trainable variables :",
        len(text_variables)
    )

    print(
        "Audio trainable variables:",
        len(audio_variables)
    )

    print(
        "Text optimizer           : SFO"
    )

    print(
        "Audio optimizer          : SFO"
    )

    print(
        "Learning rate            :",
        lr
    )

    print(
        "Contrastive weight       :",
        contrastive_weight
    )

    print(
        "Temperature              :",
        temperature
    )

    print("=" * 60)

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(1, epochs + 1):

        # ----------------------------------------------------
        # INITIALIZE EPOCH METRICS
        # ----------------------------------------------------

        text_loss_total = 0.0
        audio_loss_total = 0.0

        text_correct = 0
        text_total = 0

        audio_correct = 0
        audio_total = 0

        # ====================================================
        # TEXT BRANCH
        # ====================================================

        if text_dataset is not None:

            for batch_text, batch_y in text_dataset:

                # --------------------------------------------
                # FORWARD + LOSS
                # --------------------------------------------

                with tf.GradientTape() as tape:

                    output = model(
                        text=batch_text,
                        training=True
                    )

                    logits, projections = output["text"]

                    # Classification loss
                    ce_loss = ce_loss_fn(
                        batch_y,
                        logits
                    )

                    # Supervised contrastive loss
                    con_loss = supcon_loss_fn(
                        batch_y,
                        projections
                    )

                    # Total loss
                    loss = (
                        ce_loss
                        +
                        contrastive_weight * con_loss
                    )

                # --------------------------------------------
                # GRADIENTS
                # --------------------------------------------

                gradients = tape.gradient(
                    loss,
                    text_variables
                )

                # --------------------------------------------
                # GRADIENT CLIPPING
                # --------------------------------------------

                gradients = [
                    tf.clip_by_norm(
                        g,
                        5.0
                    )
                    if g is not None
                    else None
                    for g in gradients
                ]

                # --------------------------------------------
                # SFO OPTIMIZATION
                # --------------------------------------------

                text_optimizer.apply_gradients(
                    zip(
                        gradients,
                        text_variables
                    )
                )

                # --------------------------------------------
                # ACCURACY
                # --------------------------------------------

                predictions = tf.argmax(
                    logits,
                    axis=1,
                    output_type=tf.int32
                )

                correct = tf.reduce_sum(
                    tf.cast(
                        tf.equal(
                            predictions,
                            batch_y
                        ),
                        tf.int32
                    )
                )

                text_correct += int(correct)

                n = int(batch_y.shape[0])

                text_total += n

                text_loss_total += (
                    float(loss) * n
                )

        # ====================================================
        # AUDIO BRANCH
        # ====================================================

        if audio_dataset is not None:

            for batch_audio, batch_y in audio_dataset:

                # --------------------------------------------
                # FORWARD + LOSS
                # --------------------------------------------

                with tf.GradientTape() as tape:

                    output = model(
                        audio=batch_audio,
                        training=True
                    )

                    logits, projections = output["audio"]

                    # Classification loss
                    ce_loss = ce_loss_fn(
                        batch_y,
                        logits
                    )

                    # Supervised contrastive loss
                    con_loss = supcon_loss_fn(
                        batch_y,
                        projections
                    )

                    # Total loss
                    loss = (
                        ce_loss
                        +
                        contrastive_weight * con_loss
                    )

                # --------------------------------------------
                # GRADIENTS
                # --------------------------------------------

                gradients = tape.gradient(
                    loss,
                    audio_variables
                )

                # --------------------------------------------
                # GRADIENT CLIPPING
                # --------------------------------------------

                gradients = [
                    tf.clip_by_norm(
                        g,
                        5.0
                    )
                    if g is not None
                    else None
                    for g in gradients
                ]

                # --------------------------------------------
                # SFO OPTIMIZATION
                # --------------------------------------------

                audio_optimizer.apply_gradients(
                    zip(
                        gradients,
                        audio_variables
                    )
                )

                # --------------------------------------------
                # ACCURACY
                # --------------------------------------------

                predictions = tf.argmax(
                    logits,
                    axis=1,
                    output_type=tf.int32
                )

                correct = tf.reduce_sum(
                    tf.cast(
                        tf.equal(
                            predictions,
                            batch_y
                        ),
                        tf.int32
                    )
                )

                audio_correct += int(correct)

                n = int(batch_y.shape[0])

                audio_total += n

                audio_loss_total += (
                    float(loss) * n
                )

        # ====================================================
        # EPOCH RESULTS
        # ====================================================

        message = (
            f"Epoch {epoch:03d}/{epochs}"
        )

        # ----------------------------------------------------
        # TEXT RESULTS
        # ----------------------------------------------------

        if text_total > 0:

            text_accuracy = (
                text_correct /
                text_total
            ) * 100.0

            text_loss = (
                text_loss_total /
                text_total
            )

            message += (
                f" | Text Loss: "
                f"{text_loss:.4f}"
                f" | Text Acc: "
                f"{text_accuracy:.2f}%"
            )

        # ----------------------------------------------------
        # AUDIO RESULTS
        # ----------------------------------------------------

        if audio_total > 0:

            audio_accuracy = (
                audio_correct /
                audio_total
            ) * 100.0

            audio_loss = (
                audio_loss_total /
                audio_total
            )

            message += (
                f" | Audio Loss: "
                f"{audio_loss:.4f}"
                f" | Audio Acc: "
                f"{audio_accuracy:.2f}%"
            )

        # ----------------------------------------------------
        # PRINT EPOCH
        # ----------------------------------------------------

        print(message)

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print("\n")
    print("=" * 60)
    print("Training completed successfully.")
    print("=" * 60)

    return model

# ============================================================
# 6. TEST FUNCTION
# ============================================================


def test_litekdc(model, X_test, y_test, label_encoder=None):
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)

    logits, projections = model(X_test, training=False)
    prediction_class = tf.argmax(logits, axis=1).numpy()
    return prediction_class, logits.numpy(), cm

