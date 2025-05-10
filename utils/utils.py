def calculate_accuracy(y_true, y_pred):
    correct = (y_true == y_pred).sum().item()
    return correct / len(y_true)