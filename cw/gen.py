import numpy as np

# Number of points to generate
num_points = 1000

# Parameters for exponential distribution (adjust as needed)
lambda_x = 0.1  # Rate parameter for x
lambda_y = 0.2  # Rate parameter for y

# Generate non-uniform x and y coordinates
x = np.random.exponential(scale=1/lambda_x, size=num_points)
y = np.random.exponential(scale=1/lambda_y, size=num_points)

# Generate random z values (intensities)
z = np.random.rand(num_points) * 10  # Adjust the range as needed

# Create a list of tuples (x, y, z)
data = list(zip(x, y, z))

# Write the data to a CSV file
with open('non_uniform_random_points.csv', 'w') as f:
    f.write('x,y,z\n')
    for row in data:
        f.write(','.join(map(str, row)) + '\n')

print("Non-uniform random points saved to non_uniform_random_points.csv")