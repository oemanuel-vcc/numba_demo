from numba import jit
import random
import time

@jit(nopython=True)
def monte_carlo_pi_numba(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

def monte_carlo_pi_no_numba(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

def main():
    samples = 10000000

    start = time.time()
    monte_carlo_pi_numba(samples)
    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))

    start = time.time()
    est = monte_carlo_pi_numba(samples)
    end = time.time()

    print(f"Elapsed (without compilation) = {end-start}")

    start = time.time()
    monte_carlo_pi_no_numba(samples)
    end = time.time()

    print(f"Elapsed (witout numba) = {end-start}")

    print(f"Estimate: {est}")

if __name__ == "__main__":
    main()
