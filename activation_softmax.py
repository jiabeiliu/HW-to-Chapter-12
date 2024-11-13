# activation_softmax.py
import numpy as np

class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        """
        Forward pass for the softmax activation.
        
        Args:
            inputs (np.array): The input data for which softmax is to be computed.
                               Expected shape (batch_size, number_of_classes)
        
        Returns:
            np.array: Softmax output with values between 0 and 1,
                      summing up to 1 across each row (for each input).
        """
        # Shift inputs to avoid numerical overflow (stabilizes large exponentials)
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        
        # Exponentiate inputs
        exp_values = np.exp(shifted_inputs)
        
        # Normalize by the sum of exponentials for each input in the batch
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        return self.output

    def backward(self, dvalues):
        """
        Backward pass for the softmax activation.
        
        Args:
            dvalues (np.array): The gradient of the loss with respect to
                                the softmax output.
        
        Returns:
            np.array: Gradient of the loss with respect to the input of softmax.
        """
        # Create an array to store the gradients
        self.dinputs = np.empty_like(dvalues)
        
        # For each output, compute the Jacobian matrix and apply the chain rule
        for index, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):
            # Flatten single_output
            single_output = single_output.reshape(-1, 1)
            
            # Compute Jacobian matrix of the softmax function
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            # Calculate the sample-wise gradient and store it
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalue)
        
        return self.dinputs


# Example usage
if __name__ == "__main__":
    # Example input
    inputs = np.array([[1.0, 2.0, 3.0],
                       [1.0, 2.0, 5.0]])

    # Initialize and apply softmax activation
    softmax = Softmax()
    output = softmax.forward(inputs)
    print("Softmax Output:\n", output)

    # Example gradient from next layer (dummy data)
    dvalues = np.array([[1, 0, -1],
                        [0, 1, -1]])

    # Backward pass
    dinputs = softmax.backward(dvalues)
    print("Gradient wrt Inputs:\n", dinputs)
