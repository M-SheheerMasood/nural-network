import numpy as np
import pickle
import os
from PIL import Image

TARGET_IMG_SIZE = (64, 64)
CLASSES = ["cat", "not cat"]
OUTPUT_SIZE = len(CLASSES)
INPUT_SIZE = TARGET_IMG_SIZE[0] * TARGET_IMG_SIZE[1]
HIDDEN_SIZE = 128
LEARNING_RATE = 0.01
MODEL_FILENAME = "my_model_state.pkl"

def activation(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def derivative_activation(a):
    return a * (1.0 - a)

def derivative_softmax(a, y_true):
    return a - y_true

def xavier_init(dim_in, dim_out):
    low = -np.sqrt(6.0 / (dim_in + dim_out))
    high = np.sqrt(6.0 / (dim_in + dim_out))
    return np.random.uniform(low, high, (dim_in, dim_out))

def forward_propagation(X, w1, b1, w2, b2):
    z1 = np.dot(X, w1) + b1
    a1 = activation(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def backward_propagation(X, y_one_hot, a1, a2, w2, z1):
    m = X.shape[0]
    dz2 = derivative_softmax(a2, y_one_hot)
    dw2 = (1 / m) * np.dot(a1.T, dz2)
    db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * derivative_activation(activation(z1))
    dw1 = (1 / m) * np.dot(X.T, dz1)
    db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)
    return dw1, db1, dw2, db2

def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    return w1, b1, w2, b2

def load_and_process_image(filepath, target_size):
    try:
        img = Image.open(filepath)
        img = img.convert('L')
        img = img.resize(target_size)
        img_array = np.array(img)
        flattened_img = img_array.flatten()
        normalized_img = flattened_img / 255.0
        final_input = normalized_img.reshape(1, -1)
        return final_input
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def save_model_state(w1, b1, w2, b2, filename):
    state = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    print(f"-> Model state saved to {filename}")

def load_model_state(filename, input_size, hidden_size, output_size):
    if os.path.exists(filename):
        print(f"Found existing model state at {filename}.")
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        return state['w1'], state['b1'], state['w2'], state['b2']
    else:
        print("No existing model found. Starting with new.")
        w1 = xavier_init(input_size, hidden_size)
        b1 = np.zeros((1, hidden_size))
        w2 = xavier_init(hidden_size, output_size)
        b2 = np.zeros((1, output_size))
        return w1, b1, w2, b2

if __name__ == "__main__":
    w1, b1, w2, b2 = load_model_state(MODEL_FILENAME, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    while True:
        print("\n--- Image Learning Terminal ---")
        print(f"Current Classes: {CLASSES}")
        print("Type exit to quit.")

        img_path = input("Enter path to image file: ").strip()
        if img_path.lower() == 'exit': break

        if not os.path.exists(img_path):
            print("File not found.")
            continue

        X_input = load_and_process_image(img_path, TARGET_IMG_SIZE)
        if X_input is None: continue

        _, _, _, a2_pred = forward_propagation(X_input, w1, b1, w2, b2)
        predicted_idx = np.argmax(a2_pred)
        confidence = a2_pred[0][predicted_idx]
        print(f"\nModel currently thinks this is: '{CLASSES[predicted_idx]}' (Confidence: {confidence:.2f})")

        label_str = input(f"What is the actual label for this image? {CLASSES}: ").strip()
        if label_str.lower() == 'exit': break

        if label_str not in CLASSES:
            print(f"Invalid class. Please choose from {CLASSES}")
            continue

        label_idx = CLASSES.index(label_str)
        y_one_hot = np.zeros((1, OUTPUT_SIZE))
        y_one_hot[0, label_idx] = 1.0

        print("Training on this image...")

        z1, a1, z2, a2 = forward_propagation(X_input, w1, b1, w2, b2)

        dw1, db1, dw2, db2 = backward_propagation(X_input, y_one_hot, a1, a2, w2, z1)

        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, LEARNING_RATE)

        a2_clipped = np.clip(a2, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_one_hot * np.log(a2_clipped))
        print(f"Done. Loss on this sample: {loss:.4f}")

        save_model_state(w1, b1, w2, b2, MODEL_FILENAME)

    print("\nExiting program.")