import numpy as np
from multi_grad_desc import multi_gradient_descent, get_bounded_normal, get_bounded_skewed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_features", type=int, default=100)
    parser.add_argument("--skew_param", type=int, default=100)

    args = parser.parse_args()

    print(f"{'epochs:':<12}", args.epochs)
    print(f"{'lr:':<12}", args.lr)
    print(f"{'n_samples:':<12}", args.n_samples)
    print(f"{'n_features:':<12}", args.n_features)
    print(f"{'skew_param:':<12}", args.skew_param)
    print()

    m = args.n_samples
    n = args.n_features
    X1 = np.random.rand(m, n)
    X2 = get_bounded_normal((m, n))
    X3 = get_bounded_skewed((m, n), skew_param=args.skew_param)
    noise = np.random.normal(0, 0.01, (m, n))
    train_idx = int(m * 0.7)
    w_gt = np.random.rand(n)
    b_gt = np.random.rand()
    names = ["uniform", "normal", "skewed"]
    for i, X in enumerate((X1, X2, X3)):
        y = w_gt @ X.T + b_gt
        w, b = multi_gradient_descent(
            X[:train_idx] + noise[:train_idx], y[:train_idx], lr=1e-2, epochs=2000)
        y_hat = X[train_idx:] @ w + b
        print("MSE", f"{names[i]:<8}", np.mean((y_hat - y[train_idx:]) ** 2))
