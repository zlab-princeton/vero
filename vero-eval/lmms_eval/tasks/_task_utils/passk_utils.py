import math

def combinations(n, k):
    if k < 0 or k > n:
        return 0
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def calculate_passk(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - float(combinations(n - c, k)) / float(combinations(n, k))