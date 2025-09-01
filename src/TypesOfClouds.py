import os
import numpy as np
import tensorflow as tf
import matplotlib
import tf2onnx
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import AdamW
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import EfficientNetB4, efficientnet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.TypesOfClouds_extra.CustomDataManager import CustomDataManager


class TypesOfClouds:
    def __init__(self):
        # Define the images directory
        images_dir = "resources/TypesOfClouds"
        output_dir = "results/TypesOfClouds"

        # Avoid explosions
        matplotlib.use("Agg")  # use a non-interactive backend

        # Functions
        self.download_data(images_dir)
        self.train(images_dir, output_dir)

    @staticmethod
    def download_data(images_dir):
        # Create the directory if it doesn't exist
        os.makedirs(images_dir, exist_ok=True)
        # Does nothing else on purpose, you must load your data to resources


    def train(self, images_dir, output_dir):

        # Prepare for K-fold cross validation
        filepaths_dict = self._create_filepaths_dictionary(images_dir)
        X_all, y_all_encoded, y_all_one_hot, label_encoder = self._prepare_for_stratification(filepaths_dict)

        # Compute class weights, instead of bootstrapping to balance the classes
        num_classes = len(label_encoder.classes_)
        class_weights_array = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_all_encoded)
        class_weights = {i: w for i, w in enumerate(class_weights_array)}

        # Perform stratified 5-fold cross-validation
        splitter = StratifiedKFold(n_splits=5, shuffle=True)
        for fold_no, (train_val_idx, test_idx) in enumerate(splitter.split(X_all, y_all_encoded), 1):
            print(f"Processing Fold {fold_no}...")

            # Split train_val_idx into training and validation (80/20)
            train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, stratify=y_all_encoded[train_val_idx])

            # Split filepaths and one-hot encoded labels into training, validation, and test for this fold
            X_train = X_all[train_idx]
            y_train = y_all_one_hot[train_idx]
            X_val = X_all[val_idx]
            y_val = y_all_one_hot[val_idx]
            X_test = X_all[test_idx]
            y_test = y_all_one_hot[test_idx]

            # Create TensorFlow datasets for training, validation, and test
            train_ds = CustomDataManager().make_dataset(X_train, y_train, is_train_set=True)
            val_ds = CustomDataManager().make_dataset(X_val, y_val, is_train_set=False)
            test_ds = CustomDataManager().make_dataset(X_test, y_test, is_train_set=False)

            # Build the base model with frozen weights
            efficient_net_B4_model = EfficientNetB4(include_top=False, weights="imagenet",pooling="avg")
            efficient_net_B4_model.trainable = False  # Freeze all layers in the base model

            # Build the model (sequential)
            # model = models.Sequential([
            #     layers.Input(shape=(380, 380, 3)),
            #     efficient_net_B4_model,
            #     layers.Dense(1024, activation="relu"),
            #     layers.Dropout(0.3),
            #     layers.Dense(512, activation="relu"),
            #     layers.Dropout(0.2),
            #     layers.Dense(256, activation="relu"),
            #     layers.Dropout(0.1),
            #     layers.Dense(num_classes, activation="softmax")
            # ])

            # Build the model (functional)
            inputs = layers.Input(shape=(380, 380, 3))
            x = efficient_net_B4_model(inputs)
            x = layers.Dense(1024, activation="relu")(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(512, activation="relu")(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(256, activation="relu")(x)
            x = layers.Dropout(0.1)(x)
            outputs = layers.Dense(num_classes, activation="softmax")(x)
            model = Model(inputs=inputs, outputs=outputs, name="cloud_classifier")

            # Print the model summary
            model.summary()

            # Compile the model
            model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5), loss="categorical_crossentropy",
                          metrics=["accuracy", "precision", "recall"])

            # Callback
            early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
            reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6)
            callbacks = [early_stopping, reduce_on_plateau]

            # Fit the model
            history = model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=callbacks, class_weight=class_weights, verbose=1)

            # Save the model
            os.makedirs(output_dir, exist_ok=True)
            model.save(os.path.join(output_dir, f"model_fold{fold_no}.keras"))
            onnx_model_filepath = os.path.join(output_dir, f"model_fold{fold_no}.onnx")
            input_signature = [tf.TensorSpec([None, 380, 380, 3], tf.float32, name="input")]
            onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, output_path=onnx_model_filepath, opset=13)

            # Evaluation
            test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_ds)
            self._generate_loss_curves(history, output_dir, fold_no)
            self._generate_confusion_matrix(model, label_encoder, test_ds, output_dir, fold_no)
            self._generate_roc_auc_curves(model, label_encoder, test_ds, output_dir, fold_no)

            print(f"End of fold {fold_no} - Training samples: {len(X_train)}, "
                  f"Validation samples: {len(X_val)}, Test samples: {len(X_test)}, "
                  f"Accuracy: {test_accuracy:.4f}")



    @staticmethod
    def _create_filepaths_dictionary(images_dir):
        filepaths_dict = {}
        for class_name in os.listdir(images_dir):
            class_path = os.path.join(images_dir, class_name)
            if os.path.isdir(class_path):
                filepaths = []
                for filename in os.listdir(class_path):
                    file_path = os.path.join(class_path, filename)
                    filepaths.append(file_path)
                    filepaths_dict[class_name] = filepaths
        return filepaths_dict


    @staticmethod
    def _prepare_for_stratification(filepaths_dict):
        # Combine all filepaths and labels for stratified k-fold
        X_all = []  # List of filepaths
        y_all = []  # List of corresponding class labels
        for class_name, filepaths in filepaths_dict.items():
            X_all.extend(filepaths)
            y_all.extend([class_name] * len(filepaths))

        # Convert to numpy arrays for easier handling
        X_all = np.array(X_all)
        y_all = np.array(y_all)

        # Encode labels
        label_encoder = LabelEncoder()
        y_all_encoded = label_encoder.fit_transform(y_all)

        # Convert the encoded labels to one hot vectors
        num_classes = len(label_encoder.classes_)
        y_all_one_hot = to_categorical(y_all_encoded, num_classes=num_classes)

        return X_all, y_all_encoded, y_all_one_hot, label_encoder


    @staticmethod
    def _generate_loss_curves(history, output_dir, fold_no):
        """Plots training & validation loss and accuracy curves."""
        os.makedirs(output_dir, exist_ok=True)

        # Plot Loss
        plt.figure()
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.title(f"Fold {fold_no} Loss", fontweight='bold')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"loss_fold{fold_no}.png"))
        plt.close()

        # Plot Accuracy
        plt.figure()
        plt.plot(history.history["accuracy"], label="train_acc")
        plt.plot(history.history["val_accuracy"], label="val_acc")
        plt.title(f"Fold {fold_no} Accuracy", fontweight='bold')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"accuracy_fold{fold_no}.png"))
        plt.close()

    @staticmethod
    def _generate_confusion_matrix(model, label_encoder, test_ds, output_dir, fold_no):
        """Generates and saves the confusion matrix."""
        os.makedirs(output_dir, exist_ok=True)

        # Predictions
        y_prob = model.predict(test_ds)
        y_pred = np.argmax(y_prob, axis=1)

        # True labels
        y_true = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds], axis=0)

        # Confusion matrix
        cm = confusion_matrix(y_pred, y_true)
        class_names = label_encoder.classes_

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"size": 14}
        )
        plt.title(f"Confusion Matrix Fold {fold_no}", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("True Label", fontsize=14, fontweight='bold')
        plt.ylabel("Predicted Label", fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12, rotation=45, ha='right')
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_fold{fold_no}.png"))
        plt.close()

    @staticmethod
    def _generate_roc_auc_curves(model, label_encoder, test_ds, output_dir, fold_no):
        """Generates One-vs-One and One-vs-Rest ROC-AUC curves and saves plots."""
        os.makedirs(output_dir, exist_ok=True)

        # Get predictions
        y_prob = model.predict(test_ds)

        # Extract true labels from dataset
        y_true = np.concatenate([y for x, y in test_ds], axis=0)

        # Count the number of classes
        n_classes = y_prob.shape[1]

        # Get class names from label encoder
        class_names = label_encoder.classes_

        # Compute micro-average ROC-AUC
        y_true_flat = y_true.ravel()  # Flatten for micro-average
        y_prob_flat = y_prob.ravel()
        fpr_micro, tpr_micro, _ = roc_curve(y_true_flat, y_prob_flat)
        auc_micro = auc(fpr_micro, tpr_micro)

        # Compute macro-average ROC-AUC
        auc_macro = 0
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            auc_macro += auc(fpr, tpr)
        auc_macro /= n_classes

        # One-vs-Rest
        plt.figure(figsize=(6.4, 5))  # Default is 6.4, 4.8
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'One-vs-One ROC Fold {fold_no}', fontweight='bold', fontsize=12)
        plt.tight_layout(rect=[0.05, 0.12, 1, 1])  # Leave space on the bottom
        plt.figtext(0.5, -0.20, f'AUC Micro Avg = {auc_micro:.2f}, AUC Macro Avg = {auc_macro:.2f}',
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right', fontsize=9)
        plt.savefig(os.path.join(output_dir, f'roc_ovr_fold{fold_no}.png'))
        plt.close()

        # one-vs-one
        plt.figure(figsize=(14, 6))
        for i, j in combinations(range(n_classes), 2):
            # Select only samples of class i or j
            mask = np.logical_or(y_true[:, i] == 1, y_true[:, j] == 1)
            y_binary = y_true[mask, i]  # class i = 1, class j = 0
            y_score = y_prob[mask, i]  # probability of class i
            fpr, tpr, _ = roc_curve(y_binary, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_names[i]} vs {class_names[j]} (AUC={roc_auc:.2f})')
        plt.title(f'One-vs-One ROC Fold {fold_no}', fontweight='bold')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, ncol=2)
        plt.tight_layout(rect=[0, 0, 1, 1])  # leave space on the right
        plt.savefig(os.path.join(output_dir, f'roc_ovo_fold{fold_no}.png'))
        plt.close()
