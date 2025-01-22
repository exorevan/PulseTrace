import numpy as np
from sklearn.svm import SVC
import joblib  # Recommended for saving sklearn models [[2]]

# 1. Prepare XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([0, 1, 1, 0], dtype=np.float64)

# 2. Create and train the SVM (RBF kernel to capture non-linear patterns)
svm_model = SVC(kernel="rbf", gamma=1.0, C=1.0, probability=True)
svm_model.fit(X, y)

# 3. Print relevant parameters
print("Support Vectors:")
print(svm_model.support_vectors_)  # [[3]]
print("Dual Coefficients:")
print(svm_model.dual_coef_)  # [[3]]
print("Intercept:")
print(svm_model.intercept_)  # [[3]]

# 4. Save the trained model
joblib.dump(svm_model, "xor_svm_model.pkl")  # [[2]]
print("Model saved to 'xor_svm_model.pkl'!")
