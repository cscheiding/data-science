import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.models import Model
from keras.src.models.functional import Functional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers
from Tools import preprocessing, regression_and_metrics

# Define custom types:
type Function = float

def regularization_function(
                l1: float | None = 1e-4,
                l2: float | None = 1e-4,
                ) -> l1 | l2 | l1_l2 | None:
    """
    Return a regularization function, depending on the values of L1 and L2:
    1. If L1 != NONE and L2 != NONE, return an Elastic Net regularization
    function with the given L1 and L2 regularization parameters
    2. If L1 != NONE and L2 = NONE, return an L1 regularization function with
    the given L1 regularization parameter
    3. If L1 = NONE and L2 != NONE, return an L2 regularization function with
    the given L2 regularization parameter

    Parameters
    -----------
    - L1: L1 regularization parameter
    - L2: L2 regularization parameter

    Returns
    --------
    - REGULARIZER: regularization function
    """
    # Determine the regularizer to use (if any):
    if l1 and l2:
        regularizer = l1_l2(l1 = l1, l2 = l2)
    elif l1:
        regularizer = l1(l1)
    elif l2:
        regularizer = l2(l2)
    else:
        regularizer = None
    return regularizer

def create_encoder_model(
        input_dimension: int,
        encoding_dimension: int,
        activation_function: str = 'relu',
        hidden_layers_dimensions: tuple[int] | None = None,
        regularization__batch_normalization: bool = True,
        regularization__dropout_rate: float | None = 0.2,
        regularization__kernel_regularizer: l1 | l2 | l1_l2 | None = None
        ) -> Model:
    """
    Create and return an encoder Model.

    Parameters
    -----------
    - INPUT_DIMENSION: encoder's input dimension
    - ENCODING_DIMENSION: to how many components one intends to encode the
    features in ENCODER_INPUT_LAYER
    - ACTIVATION_FUNCTION: activation function to use in the Dense layers
    - HIDDEN_LAYERS_DIMENSIONS: dimensions (no. of units) of the hidden layers.
    It is assumed that the hidden layers will go from smallest to biggest. If
    HIDDEN_LAYERS_DIMENSIONS = NONE, no hidden layers are used
    - REGULARIZATION__BATCH_NORMALIZATION: whether to use Batch Normalization
    after each hidden layer of the encoder
    - REGULARIZATION__DROPOUT_RATE: if provided, it represents the percentage of
    neurons that are dropped out after each hidden layer of the encodder. If
    REGULARIZATION__DROPOUT_RATE = NONE, no dropout is used
    - REGULARIZATION__KERNEL_REGULARIZER: kernel regularizer function to be
    applied to each Dense layer. Can be an L1, L2 or Elastic Net (L1 + L2)
    regularization penalty. REGULARIZATION__KERNEL_REGULARIZER = NONE, no
    kernel regularization is applied
    
    Returns
    --------
    ENCODER_MODEL
    """
    # Encoder's input layer:
    input_layer = Input(shape = (input_dimension,))
    # Start adding layers from this layer onward:
    previous_layer = input_layer
    # Add the hidden layers if their dimensions are specified:
    if hidden_layers_dimensions:
        # The dimensions go from smallest to biggest:
        for dimension in sorted(hidden_layers_dimensions):
            hidden_layer = Dense(
                        dimension,
                        activation = activation_function,
                        kernel_regularizer = regularization__kernel_regularizer
                        )(previous_layer)
            if regularization__batch_normalization == True:
                hidden_layer = BatchNormalization()(hidden_layer)
            if regularization__dropout_rate:
                hidden_layer = Dropout(
                                    regularization__dropout_rate
                                    )(hidden_layer)
            
            previous_layer = hidden_layer
    # Add the encoded layer and define the encoder Model:
    encoded_layer = Dense(
                        encoding_dimension,
                        activation = activation_function,
                        kernel_regularizer = regularization__kernel_regularizer
                        )(previous_layer)
    encoder_model = Model(
                        inputs = input_layer,
                        outputs = encoded_layer,
                        name = 'encoder'
                        )
    return encoder_model

def create_decoder_model(
        output_dimension: int,
        encoding_dimension: int,
        activation_function: str = 'relu',
        hidden_layers_dimensions: tuple[int] | None = None,
        regularization__kernel_regularizer: l1 | l2 | l1_l2 | None = None
        ):
    """
    Create and return a decoder Model.

    Parameters
    -----------
    - OUTPUT_DIMENSION: dimension of the decoder's output
    - ENCODING_DIMENSION: dimension of the encoded layer
    - ACTIVATION_FUNCTION: activation function to use in the Dense layers
    - HIDDEN_LAYERS_DIMENSIONS: dimensions (no. of units) of the hidden layers.
    It is assumed that the hidden layers will go from biggest to smallest. If
    HIDDEN_LAYERS_DIMENSIONS = NONE, no hidden layers are used
    - REGULARIZATION__KERNEL_REGULARIZER: kernel regularizer function to be
    applied to each Dense layer. Can be an L1, L2 or Elastic Net (L1 + L2)
    regularization penalty. REGULARIZATION__KERNEL_REGULARIZER = NONE, no
    kernel regularization is applied
    
    Returns
    --------
    DECODER_MODEL
    """
    # Decoder's input layer:
    encoded_layer = Input(
                        shape = (encoding_dimension,),
                        name = 'encoding'
                        )
    # Start adding layers from the DECODER_INPUT_LAYER onward:
    previous_layer = encoded_layer
    # Add the hidden layers if their dimensions are specified:
    if hidden_layers_dimensions:
        # The dimensions go from biggest to smallest:
        for dimension in sorted(hidden_layers_dimensions, reverse = True):
            hidden_layer = Dense(
                        dimension,
                        activation = activation_function,
                        kernel_regularizer = regularization__kernel_regularizer
                        )(previous_layer)
            previous_layer = hidden_layer
    ## Add the decoded layer and define the decoder model:
    decoded_layer = Dense(
                        output_dimension,
                        activation = 'linear',
                        kernel_regularizer = regularization__kernel_regularizer
                        )(previous_layer)
    # Define the decoder model:
    decoder_model = Model(
                        inputs = encoded_layer,
                        outputs = decoded_layer,
                        name = 'decoder'
                        )
    return decoder_model

def create_autoencoder_model(
            input_dimension: int,
            encoding_dimension: int,
            activation_function: str = 'relu',
            hidden_layers_dimensions: tuple[int] | None = None,
            optimizer: str = 'adam',
            initial_learning_rate: float = 1e-3,
            loss_function: str = 'binary_crossentropy',
            regularization__batch_normalization: bool = True,
            regularization__dropout_rate: float | None = 0.2,
            regularization__kernel_regularizer: l1 | l2 | l1_l2 | None = None
            ) -> tuple[Model]:
    """
    - Create an autoencoder model.
    - It includes the following callbacks after LOSS_FUNCTION stops decreasing:
    1. Early stopping
    2. Reduction of learning rate
    - Return a tuple containing the autoencoder, encoder and decoder.
    
    Parameters
    -----------
    - INPUT_DIMENSION: dimension of the encoder's input and decoder's output
    - ENCODING_DIMENSION: to how many components one intends to encode the
    features
    - ACTIVATION_FUNCTION: activation function to use in the Dense layers
    - HIDDEN_LAYERS_DIMENSIONS: dimensions (no. of units) of the hidden layers
    of the encoder and decoder. If HIDDEN_LAYERS_DIMENSIONS = NONE, no hidden
    layers are used
    - OPTIMIZER: optimizer for compiling the model
    - INITIAL_LEARNING_RATE: initial learning rate for the optimizer
    - LOSS_FUNCTION: loss function for compiling the model
    - REGULARIZATION__BATCH_NORMALIZATION: whether to use Batch Normalization
    after each hidden layer of the encoder
    - REGULARIZATION__DROPOUT_RATE: if provided, it represents the percentage of
    neurons that are dropped out after each hidden layer of the encodder. If
    REGULARIZATION__DROPOUT_RATE = NONE, no dropout is used
    - REGULARIZATION__KERNEL_REGULARIZER: kernel regularizer function to be
    applied to each Dense layer. Can be an L1, L2 or Elastic Net (L1 + L2)
    regularization penalty. REGULARIZATION__KERNEL_REGULARIZER = NONE, no
    kernel regularization is applied

    Returns
    --------
    - AUTOENCODER, ENCODER, DECODER
    """
    # Get the optimizer object corresponding to the OPTIMIZER string:
    optimizer_object = optimizers.get({
                        'class_name': optimizer,
                        'config': {'learning_rate': initial_learning_rate}
                        })
    # Create an encoder Model:
    encoder_model = create_encoder_model(
        input_dimension = input_dimension,
        encoding_dimension = encoding_dimension,
        activation_function = activation_function,
        hidden_layers_dimensions = hidden_layers_dimensions,
        regularization__batch_normalization = (
                                    regularization__batch_normalization
                                    ),
        regularization__dropout_rate = regularization__dropout_rate,
        regularization__kernel_regularizer = regularization__kernel_regularizer
    )
    # Create a decoder Model:
    decoder_model = create_decoder_model(
        output_dimension = input_dimension, # Reconstruct the input dimension
        encoding_dimension = encoding_dimension,
        activation_function = activation_function,
        hidden_layers_dimensions = hidden_layers_dimensions,
        regularization__kernel_regularizer = regularization__kernel_regularizer
    )
    # Define and compile an autoencoder Model:
    input_layer = Input(shape = (input_dimension,))
    output_layer = decoder_model(encoder_model(input_layer))
    autoencoder_model = Model(input_layer, output_layer, name = 'autoencoder')
    autoencoder_model.compile(
                            optimizer = optimizer_object,
                            loss = loss_function
                            )
    autoencoder_model.summary()
    return autoencoder_model, encoder_model, decoder_model


def fit_autoencoder_model(
                autoencoder: Model,
                X_train: pd.DataFrame,
                epochs: int = 5,
                batch_size: int = 32,
                shuffle: bool = True,
                validation_split: float = 0.1,
                early_stopping__monitor: str = 'val_loss',
                early_stopping__patience: int = 5,
                early_stopping__min_delta: float = 1e-4,
                reduce_learning_rate__monitor: str = 'val_loss',
                reduce_learning_rate__patience: int = 3,
                reduce_learning_rate__min_delta: float = 1e-4,
                reduce_learning_rate__factor: float = 0.5,
                ) -> None:
    """
    Fit the autoencoder.

    Parameters
    -----------
    - AUTOENCODER: previously created and compiled autoencoder Model
    - X_TRAIN: training split of the features
    - EPOCHS: epochs for fitting the model
    - BATCH_SIZE: batch size for fitting the model
    - SHUFFLE: whether to shuffle the order in which the batches are used
    - VALIDATION_SPLIT: fraction of the training split to use for validation
    - REGULARIZATION__BATCH_NORMALIZATION: whether to use Batch Normalization
    after each hidden layer of the encoder
    - EARLY_STOPPING__MONITOR: which loss function must stop decreasing, for
    EarlyStopping to be triggered
    - EARLY_STOPPING__PATIENCE: during how many epochs EARLY_STOPPING__MONITOR
    must stop decreasing for EarlyStopping to be triggered
    - EARLY_STOPPING__MIN_DELTA: minimum change in EARLY_STOPPING__MONITOR for
    EarlyStopping not to be triggered
    - REDUCE_LEARNING_RATE__MONITOR: which loss function must stop decreasing,
    for ReduceLROnPlateau to be triggered
    - REDUCE_LEARNING_RATE__PATIENCE: during how many epochs
    REDUCE_LEARNING_RATE__MONITOR must stop decreasing for ReduceLROnPlateau to
    be triggered
    - REDUCE_LEARNING_RATE__MIN_DELTA: minimum change i
    REDUCE_LEARNING_RATE__MONITOR for ReduceLROnPlateau not to be triggered
    - REDUCE_LEARNING_RATE__FACTOR: percentage by which to multiply the current
    learning rate when ReduceLROnPlateau is triggered
    """
    # Pre-processing:
    ## ColumnTransformer that applies One-Hot encoding for categorical features
    ## and Standard Scaling for numeric features:
    preprocessor = preprocessing.create_preprocessor(X = X_train)
    ## Apply pre-processing to the features' training split:
    X_train_processed = preprocessor.fit_transform(X_train) # NumPy array
    # Stop training if the evaluation loss stops improving for 5 epochs. Restore
    # the weights before it stopped improving:
    early_stopping = EarlyStopping(
                        monitor = early_stopping__monitor,
                        patience = early_stopping__patience,
                        min_delta = early_stopping__min_delta,
                        restore_best_weights = True
                        )
    # Reduce the learning rate to half its previous value, if the evaluation
    # loss stops improving for 1 epoch:
    reduce_learning_rate = ReduceLROnPlateau(
                        monitor = reduce_learning_rate__monitor,
                        patience = reduce_learning_rate__patience,
                        min_delta = reduce_learning_rate__min_delta,
                        factor = reduce_learning_rate__factor
                        )
    # Fit the autoencoder model:
    autoencoder.fit(
                X_train_processed,
                X_train_processed,
                shuffle = shuffle,
                epochs = epochs,
                batch_size = batch_size,
                validation_split = validation_split,
                callbacks = [early_stopping, reduce_learning_rate]
                )

def create_and_fit_autoencoder(
                X_train: np.array,
                encoding_dimension: int,
                hidden_layers_dimensions: tuple[int] | None = None,
                activation_function: str = 'relu',
                optimizer: str = 'adam',
                initial_learning_rate: float = 1e-3,
                loss_function: str = 'binary_crossentropy',
                training__epochs: int = 5,
                training__batch_size: int = 32,
                training__shuffle: bool = True,
                training__validation_split: float = 0.1,
                regularization__batch_normalization: bool = True,
                regularization__dropout_rate: float | None = 0.2,
                regularization__l1: float | None = 1e-4,
                regularization__l2: float | None = 1e-4,
                early_stopping__monitor: str = 'val_loss',
                early_stopping__patience: int = 5,
                early_stopping__min_delta: float = 1e-4,
                reduce_learning_rate__monitor: str = 'val_loss',
                reduce_learning_rate__patience: int = 1,
                reduce_learning_rate__min_delta: float = 1e-4,
                reduce_learning_rate__factor: float = 0.5
                ) -> tuple[Functional]:
    """
    - Create and fit an autoencoder Model.
    - It includes the following callbacks after a specified loss function stops
    decreasing:
    1. Early stopping
    2. Reduction of learning rate
    - Return a tuple containing the autoencoder, encoder and decoder.
    
    Parameters
    -----------
    - X_TRAIN: training split of the features
    - ENCODING_DIMENSION: to how many components one intends to encode the
    features
    - HIDDEN_LAYERS_DIMENSIONS: dimensions (no. of units) of the hidden layers.
    They must be ordered from biggest to smallest. If 'None', it is assumed that
    there are no hidden layers
    - ACTIVATION_FUNCTION: activation function to use in the Dense layers
    - OPTIMIZER: optimizer for fitting the model
    - INITIAL_LEARNING_RATE: initial learning rate for the optimizer
    - LOSS_FUNCTION: loss function for fitting the model
    - TRAINING__EPOCHS: epochs for fitting the model
    - TRAINING__BATCH_SIZE: batch size for fitting the model
    - TRAINING__SHUFFLE: whether to shuffle the order in which the batches are
    used
    - TRAINING__VALIDATION_SPLIT: fraction of the training split to use for
    validation
    - REGULARIZATION__BATCH_NORMALIZATION: whether to use Batch Normalization
    after each hidden layer of the encoder
    - REGULARIZATION__DROPOUT_RATE: if provided, it represents the percentage of
    neurons that are dropped out after each hidden layer of the encodder. If
    REGULARIZATION__DROPOUT_RATE = NONE, no dropout is used
    - REGULARIZATION__L1: L1 regularization penalty to be applied on Dense
    layers. If REGULARIZATION__L1 = NONE, no L1 regularization is used
    - REGULARIZATION__L2: L2 regularization penalty to be applied on Dense
    layers. If REGULARIZATION__L2 = NONE, no L2 regularization is used
    - EARLY_STOPPING__MONITOR: which loss function must stop decreasing, for
    EarlyStopping to be triggered
    - EARLY_STOPPING__PATIENCE: during how many epochs EARLY_STOPPING__MONITOR
    must stop decreasing for EarlyStopping to be triggered
    - EARLY_STOPPING__MIN_DELTA: minimum change in EARLY_STOPPING__MONITOR for
    EarlyStopping not to be triggered
    - REDUCE_LEARNING_RATE__MONITOR: which loss function must stop decreasing,
    for ReduceLROnPlateau to be triggered
    - REDUCE_LEARNING_RATE__PATIENCE: during how many epochs
    REDUCE_LEARNING_RATE__MONITOR must stop decreasing for ReduceLROnPlateau to
    be triggered
    - REDUCE_LEARNING_RATE__MIN_DELTA: minimum change i
    REDUCE_LEARNING_RATE__MONITOR for ReduceLROnPlateau not to be triggered
    - REDUCE_LEARNING_RATE__FACTOR: percentage by which to multiply the current
    learning rate when ReduceLROnPlateau is triggered

    Returns
    --------
    AUTOENCODER, ENCODER, DECODER
    """
    # Determine the kernel regularizer to use (if any) in all the Dense layers:
    kernel_regularizer = regularization_function(
                                    l1 = regularization__l1,
                                    l2 = regularization__l2
                                    )
    # Pre-processing: One-Hot encoding for categorical features and Standard
    # Scaling for numeric features:
    preprocessor = preprocessing.create_preprocessor(X = X_train)
    X_train_processed = preprocessor.fit_transform(X_train) # Apply processing
    input_dimension = X_train_processed.shape[1] # Dimension after processing
    # Create and compile an autoencoder Model:
    autoencoder, encoder, decoder = create_autoencoder_model(
        input_dimension = input_dimension,
        encoding_dimension = encoding_dimension,
        activation_function = activation_function,
        hidden_layers_dimensions = hidden_layers_dimensions,
        optimizer = optimizer,
        initial_learning_rate = initial_learning_rate,
        loss_function = loss_function,
        regularization__batch_normalization = (
                                            regularization__batch_normalization
                                            ),
        regularization__dropout_rate = regularization__dropout_rate,
        regularization__kernel_regularizer = kernel_regularizer
    )
    # Fit the autoencoder Model:
    fit_autoencoder_model(
        autoencoder = autoencoder,
        X_train = X_train,
        epochs = training__epochs,
        batch_size = training__batch_size,
        shuffle = training__shuffle,
        validation_split = training__validation_split,
        early_stopping__monitor = early_stopping__monitor,
        early_stopping__patience = early_stopping__patience,
        early_stopping__min_delta = early_stopping__min_delta,
        reduce_learning_rate__monitor = reduce_learning_rate__monitor,
        reduce_learning_rate__patience = reduce_learning_rate__patience,
        reduce_learning_rate__min_delta = reduce_learning_rate__min_delta,
        reduce_learning_rate__factor = reduce_learning_rate__factor
    )
    return autoencoder, encoder, decoder

def evaluate_encoder(
            encoder: Functional,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: np.array,
            y_test: np.array,
            fraudulent_threshold: float = 0.8,
            regression__search__cv: int = 5,
            regression__search__scoring: str = 'f1',
            regression__search__n_iter: int = 20,
            regression__search__c_range: tuple[float] = np.logspace(-3, 3, 10) 
            ) -> None:
    """
    - Use an ENCODER model to make predictions and compare them to the real
    data.
    - Calculate and show model metrics and the confusion matrix.

    Parameters
    -----------
    - ENCODER: previously-trained encoder model
    - X_TRAIN: training split of the features
    - X_TEST: testing split of the features
    - Y_TRAIN: training split of the target column
    - Y_TEST: testing split of the target column
    - FRAUDULENT_THRESHOLD: only job ads with this probability of being
    fraudulent are categorized as such
    - REGRESSION__SEARCH__CV: number of cross-validation folds for the
    randomized search over the logistic regression's hyper parameters
    - REGRESSION__SEARCH__SCORING: scoring for the randomized search over the
    logistic regression's hyper parameters 
    - REGRESSION__SEARCH__N_ITER: no. of parameter settings that are sampled
    during the randomized search over the logistic regression's hyper parameters
    - REGRESSION__SEARCH__C_RANGE: possible values for the C hyper parameter
    (inverse of regularization strength) over which to perform a randomized
    search
    """
    # ColumnTransformer that applies One-Hot encoding for categorical features
    # and Standard Scaling for numeric features:
    preprocessor = preprocessing.create_preprocessor(X = X_train)
    # Apply pre-processing to the features' training and testing splits:
    X_train_processed = preprocessor.fit_transform(X_train)#.astype('float32')
    X_test_processed = preprocessor.transform(X_test)#.astype('float32')
    # Extract latent representations of the training and testing splits:
    X_train_latent = encoder.predict(X_train_processed, verbose = 0)
    X_test_latent = encoder.predict(X_test_processed, verbose = 0)
    # Best LogisticRegression model after a randomized cross-validation search
    # over the space of hyper parameters:
    classifier = regression_and_metrics.create_and_fit_regression(
        X_train = X_train_latent,
        y_train = y_train,
        preprocess = False, # The data was already pre-processed
        search__cv = regression__search__cv,
        search__scoring = regression__search__scoring,
        search__n_iter = regression__search__n_iter,
        search__c_range = regression__search__c_range
        )
    # Make a prediction with the latent representation of the testing split:
    y_pred_proba = classifier.predict_proba(X_test_latent)[:, 1]
    y_pred_latent = (y_pred_proba >= fraudulent_threshold).astype(int)
    # Show metrics and confusion matrix:
    regression_and_metrics.show_model_metrics(y_test, y_pred_latent)
    regression_and_metrics.show_confusion_matrix(
                y_pred = y_pred_latent,
                y_true = y_test
                )
