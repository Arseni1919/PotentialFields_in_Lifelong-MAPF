import numpy as np

def has_consecutive_equal_values(list1, list2):
    # Convert the lists to numpy arrays
    array1 = np.array(list1)
    array2 = np.array(list2)

    # Check if the arrays have the same shape
    if array1.shape != array2.shape:
        raise ValueError("Arrays must have the same shape")

    # Use numpy char.equal for element-wise equality of string arrays
    consecutive_equal_values = np.char.equal(array1[:-1], array2[:-1]) & np.char.equal(array1[1:], array2[1:])

    # Check if there are any consecutive equal values
    return np.any(consecutive_equal_values)

# Example usage with strings:
list1 = ["apple", "banana", "orange", "grape"]
list2 = ["kiwi", "banana", "orange", "grape"]

result = has_consecutive_equal_values(list1, list2)
print(result)
