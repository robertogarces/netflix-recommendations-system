from collections import defaultdict
import numpy as np

def get_top_n(predictions, n=10):
    """
    Group the predictions by user and return the top-N items with the highest estimated rating.
    """
    top_n = defaultdict(list)
    
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    
    # Sort items for each user and keep only the top-N
    # Ordena por estimación y deja solo los N mejores
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

def precision_recall_at_k(predictions, k=10, threshold=4.0):
    """
    Compute Precision@K and Recall@K for each user.
    Calcula Precision@K y Recall@K para cada usuario.
    
    Parameters:
    - predictions: list of Surprise prediction objects
    - k: number of top items to consider
    - threshold: rating threshold to consider an item as relevant
    
    Returns:
    - avg_precision: average precision@k
    - avg_recall: average recall@k
    """
    user_est_true = defaultdict(list)
    
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((iid, true_r, est))
    
    precisions = []
    recalls = []

    for uid, user_ratings in user_est_true.items():
        # Select top-K predictions
        # Selecciona las K predicciones con mayor estimación
        top_k = sorted(user_ratings, key=lambda x: x[2], reverse=True)[:k]
        
        # Count relevant items in top-K
        # Cuenta cuántos en el top-K son relevantes (rating real >= threshold)
        n_rel_and_rec_k = sum((true_r >= threshold) for (_, true_r, _) in top_k)
        
        # Total relevant items for this user
        # Total de ítems relevantes para este usuario
        n_rel = sum((true_r >= threshold) for (_, true_r, _) in user_ratings)

        if n_rel == 0:
            continue  # Avoid division by zero

        precisions.append(n_rel_and_rec_k / k)
        recalls.append(n_rel_and_rec_k / n_rel)
    
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    return avg_precision, avg_recall


