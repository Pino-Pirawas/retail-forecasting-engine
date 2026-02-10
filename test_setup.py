import mlx.core as mx

# Create a tensor on the GPU/Neural Engine
a = mx.array([1.0, 2.0, 3.0])
b = mx.array([1.0, 2.0, 3.0])
c = a + b

print("MLX is working!")
print(f"Result: {c}")
print(f"Device: {mx.default_device()}")