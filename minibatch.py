import math
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    output = []
    i = len(features)
    for j in range(0,i,batch_size):
        k = j+batch_size
        batch = [features[j:k],labels[j:k]]
        output.append(batch)
    return output
