import time
import tensorflow as tf


# Create a function to perform a tensor multiplication using the CPU and another function to perform the same operation using the GPU and compare the relative times of execution between the two.
def cpu():

    startTime = time.time()
    with tf.device('/cpu:0'):
        random_matrix_cpu = tf.random.uniform(shape=(30000, 30000), minval=0, maxval=1)
        dot_operation_cpu = tf.matmul(random_matrix_cpu, tf.transpose(random_matrix_cpu))
        sum_operation_cpu = tf.reduce_sum(dot_operation_cpu)

    
    time_diff = time.time() - startTime
    print("\n" * 2)
    print("CPU Time taken:", time_diff)
    print("\n" * 2)

    return time_diff

def gpu():

    startTime = time.time()
    with tf.device('/gpu:0'):
        random_matrix_gpu = tf.random.uniform(shape=(30000, 30000), minval=0, maxval=1)
        dot_operation_gpu = tf.matmul(random_matrix_gpu, tf.transpose(random_matrix_gpu))
        sum_operation_gpu = tf.reduce_sum(dot_operation_gpu)

    
    time_diff = time.time() - startTime
    print("\n" * 2)
    print("GPU Time taken:", time_diff)
    print("\n" * 2)

    return time_diff

if __name__ == "__main__":
    cpu_time = cpu()
    gpu_time = gpu()
    # print the cpu/gpu fraction to the first decimal place
    print(f'GPU is {cpu_time/gpu_time:.1f}x faster than CPU')