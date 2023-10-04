import click
from functools import lru_cache
import os
import time
from multiprocessing import Pool
from mpmath import mp

@click.command()
@click.option('--num-digits', default=1000, help='Number of digits of Pi to compute.')
def main(num_digits):
    """Main function to compute Pi."""
    mp.dps = num_digits  # Set decimal places in mpmath
    num_threads = get_optimal_thread_count()
    num_terms = num_digits // 14  # Approximate terms needed

    start_time = time.time()
    pi_value = compute_pi_multiprocessing(num_terms, num_threads)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print_output(str(pi_value), num_digits, elapsed_time)


def print_output(pi_value, num_digits, elapsed_time):
    """
    Print the computed Pi value and timing details, and write the Pi value to a file in y-cruncher format.
    """
    print(f"Final output (first 30 digits): {pi_value[:30]}")
    print(f"Number of digits in the output: {len(pi_value.split('.')[1])}")
    print(f"Time taken to compute {num_digits} digits of Pi: {elapsed_time:.4f} seconds")

    # Write the Pi value to a file in y-cruncher format
    with open("pi.txt", "w") as f:
        # Write metadata
        f.write("PiCalculator Validation File (y-cruncher compatible)\n")
        f.write("===========================\n")
        f.write(f"Algorithm: Chudnovsky\n")
        f.write(f"Decimal Digits: {num_digits}\n")
        f.write(f"Hex Digits: {num_digits * 4 // 10}\n")
        f.write(f"Total Terms: {num_digits // 14}\n")
        f.write("\n")
        f.write("Begin Digits:\n")

        # Write the digits
        pi_digits = pi_value.split('.')[1]  # Exclude the '3.' at the beginning
        for i in range(0, len(pi_digits), 100):  # 100 digits per line
            line = pi_digits[i:i + 100]
            for j in range(0, len(line), 10):  # 10 digits per block
                f.write(line[j:j + 10] + ' ')
            f.write("\n")

        f.write("\nEnd Digits")


@lru_cache(maxsize=None)
def optimized_chudnovsky_bs(a, b):
    """Optimized function to compute Pi using Chudnovsky algorithm."""
    if a + 1 == b:
        return compute_leaf_node(a)

    m = (a + b) // 2
    p1, a1, q1 = optimized_chudnovsky_bs(a, m)
    p2, a2, q2 = optimized_chudnovsky_bs(m, b)

    return compute_internal_node(p1, a1, q1, p2, a2, q2)

def compute_leaf_node(a):
    """Compute the values for a leaf node."""
    ak = mp.mpf((545140134 * a) + 13591409)
    pk = mp.mpf(1)
    qk = mp.fac(6 * a) / (mp.fac(3 * a) * mp.fac(a)**3)
    qk *= (-262537412640768000) ** a
    return pk, ak, qk

def compute_internal_node(p1, a1, q1, p2, a2, q2):
    """Compute the values for an internal node."""
    pk = p1 * p2
    ak = a1 * q2 + a2 * p1
    qk = q1 * q2
    return pk, ak, qk

def get_optimal_thread_count():
    """Get the optimal thread count based on CPU cores."""
    return max(1, os.cpu_count())

def aggregate_thread_results(results):
    """Aggregate results from all threads."""
    p_agg, a_agg, q_agg = mp.mpf(1), mp.mpf(0), mp.mpf(1)
    for p, a, q in results:
        p_agg *= p
        a_agg = a_agg * q + a * p_agg
        q_agg *= q
    return p_agg, a_agg, q_agg

def compute_chunks_for_process(chunks):
    """Compute chunks of terms for a single process."""
    pk_agg, ak_agg, qk_agg = mp.mpf(1), mp.mpf(0), mp.mpf(1)
    for start_term, end_term in chunks:
        pk_chunk, ak_chunk, qk_chunk = optimized_chudnovsky_bs(start_term, end_term)
        pk_agg *= pk_chunk
        ak_agg = ak_agg * qk_chunk + ak_chunk * pk_agg
        qk_agg *= qk_chunk
    return pk_agg, ak_agg, qk_agg

def compute_pi_multiprocessing(num_terms, num_threads):
    """Compute Pi using multiprocessing."""
    # Compute chunk sizes with smaller sizes for later terms
    total_chunks = sum(range(1, num_threads + 1))
    individual_chunks = [i * num_terms // total_chunks for i in range(1, num_threads + 1)]
    chunk_borders = [0] + [sum(individual_chunks[:i+1]) for i in range(len(individual_chunks))]
    all_chunks = [(chunk_borders[i], chunk_borders[i+1]) for i in range(len(chunk_borders) - 1)]

    with Pool(processes=num_threads) as pool:
        results = pool.map(compute_chunks_for_process, [all_chunks[i:i+1] for i in range(len(all_chunks))])

    pk_final, ak_final, qk_final = aggregate_thread_results(results)

    ak_over_qk = ak_final / qk_final

    chudnovsky_scaling_factor = 426880 * mp.sqrt(10005)
    pi = chudnovsky_scaling_factor / ak_over_qk

    return pi


if __name__ == "__main__":
    main()
