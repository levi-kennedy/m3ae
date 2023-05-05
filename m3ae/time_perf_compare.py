import time
import tensorflow as tf


# Create a function to perform a tensor multiplication using the CPU and another function to perform the same operation using the GPU and compare the relative times of execution between the two.
def cpu():
    with tf.device('/cpu:0'):
        random_matrix_cpu = tf.random.uniform(shape=(1000, 1000), minval=0, maxval=1)
        dot_operation_cpu = tf.matmul(random_matrix_cpu, tf.transpose(random_matrix_cpu))
        sum_operation_cpu = tf.reduce_sum(dot_operation_cpu)

    startTime = time.time()
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation_cpu)
        print(result)
    time_diff = time.time() - startTime
    print("\n" * 2)
    print("CPU Time taken:", time_diff)
    print("\n" * 2)

    return time_diff

def gpu():
    with tf.device('/gpu:0'):
        random_matrix_gpu = tf.random.uniform(shape=(1000, 1000), minval=0, maxval=1)
        dot_operation_gpu = tf.matmul(random_matrix_gpu, tf.transpose(random_matrix_gpu))
        sum_operation_gpu = tf.reduce_sum(dot_operation_gpu)

    startTime = time.time()
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation_gpu)
        print(result)
    time_diff = time.time() - startTime
    print("\n" * 2)
    print("GPU Time taken:", time_diff)
    print("\n" * 2)

    return time_diff

if __name__ == "__main__":
    cpu_time = cpu()
    gpu_time = gpu()
    print("GPU is {}x faster than CPU".format(int(cpu_time/gpu_time)))