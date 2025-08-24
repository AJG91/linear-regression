
import torch as tc

def generate_rand_numbers(a, b, N):
    return (b - a) * tc.rand(N) + a

def gaussian_norm(x, std, mean):
    return (x - mean) / std

def inverse_gaussian_norm(x, std, mean):
    return x * std + mean