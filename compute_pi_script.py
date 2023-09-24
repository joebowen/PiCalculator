import click
import gmpy2
from functools import lru_cache
import os
import time
from multiprocessing import Pool


@click.command()
@click.option('--num-digits', default=1000, help='Number of digits of Pi to compute.')
def main(num_digits):
    """Main function to compute Pi."""
    num_threads = get_optimal_thread_count()
    num_terms = num_digits
    chunk_size = num_terms // num_threads

    start_time = time.time()
    pi_value = compute_pi_multiprocessing(
        num_terms, num_digits, num_threads, chunk_size
    )
    end_time = time.time()

    elapsed_time = end_time - start_time
    print_output(pi_value, num_digits, elapsed_time)


def print_output(pi_value, num_digits, elapsed_time):
    """Print the computed Pi value and timing details."""
    print(f"Final output (first 30 digits): {str(pi_value)[:30]}")
    print(f"Number of digits in the output: {len(str(pi_value).split('.')[1])}")
    print(f"Time taken to compute {num_digits} digits of Pi: {elapsed_time:.4f} seconds")

    # Write the Pi value to a file
    with open("pi.txt", "w") as f:
        f.write(str(pi_value))

@lru_cache(maxsize=None)
def optimized_chudnovsky_bs_gmpy2(a, b):
    """Optimized function to compute Pi using Chudnovsky algorithm."""
    if a + 1 == b:
        return compute_leaf_node(a)

    m = (a + b) // 2
    p1, a1, q1 = optimized_chudnovsky_bs_gmpy2(a, m)
    p2, a2, q2 = optimized_chudnovsky_bs_gmpy2(m, b)

    return compute_internal_node(p1, a1, q1, p2, a2, q2)


def compute_leaf_node(a):
    """Compute the values for a leaf node."""
    ak = gmpy2.mpz((545140134 * a) + 13591409)
    pk = gmpy2.mpz(1)
    qk = gmpy2.divexact(
        gmpy2.fac(6 * a),
        gmpy2.mul(gmpy2.fac(3 * a), gmpy2.fac(a) ** 3)
    )
    qk = gmpy2.mul(qk, (-262537412640768000) ** a)
    return pk, ak, qk


def compute_internal_node(p1, a1, q1, p2, a2, q2):
    """Compute the values for an internal node."""
    pk = gmpy2.mul(p1, p2)
    ak = gmpy2.add(gmpy2.mul(a1, q2), gmpy2.mul(a2, p1))
    qk = gmpy2.mul(q1, q2)
    return pk, ak, qk


def get_optimal_thread_count():
    """Get the optimal thread count based on CPU cores."""
    return max(1, os.cpu_count())


def set_gmpy2_precision(num_digits):
    """Set the precision for gmpy2."""
    precision_bits = int(gmpy2.ceil(num_digits * gmpy2.log2(10)))
    gmpy2.get_context().precision = precision_bits


def aggregate_thread_results(results):
    """Aggregate results from all threads."""
    p_agg, a_agg, q_agg = gmpy2.mpz(1), gmpy2.mpz(0), gmpy2.mpz(1)
    for p, a, q in results:
        p_agg = gmpy2.mul(p_agg, p)
        a_agg = gmpy2.add(gmpy2.mul(a_agg, q), gmpy2.mul(a, p_agg))
        q_agg = gmpy2.mul(q_agg, q)
    return p_agg, a_agg, q_agg


def compute_chunks_for_process(chunks):
    """Compute chunks of terms for a single process."""
    pk_agg, ak_agg, qk_agg = gmpy2.mpz(1), gmpy2.mpz(0), gmpy2.mpz(1)
    for start_term, end_term in chunks:
        pk_chunk, ak_chunk, qk_chunk = optimized_chudnovsky_bs_gmpy2(start_term, end_term)
        pk_agg = gmpy2.mul(pk_agg, pk_chunk)
        ak_agg = gmpy2.add(gmpy2.mul(ak_agg, qk_chunk), gmpy2.mul(ak_chunk, pk_agg))
        qk_agg = gmpy2.mul(qk_agg, qk_chunk)
    return pk_agg, ak_agg, qk_agg


def compute_pi_multiprocessing(num_terms, num_digits, num_threads, chunk_size):
    """Compute Pi using multiprocessing."""
    set_gmpy2_precision(num_digits)
    all_chunks = [
        (start, min(start + chunk_size, num_terms)) for start in range(0, num_terms, chunk_size)
    ]

    chunks_per_process = [all_chunks[i::num_threads] for i in range(num_threads)]

    with Pool(processes=num_threads) as pool:
        results = pool.map(compute_chunks_for_process, chunks_per_process)

    pk_final, ak_final, qk_final = aggregate_thread_results(results)
    pi = gmpy2.div(426880 * gmpy2.sqrt(10005), gmpy2.div(ak_final, qk_final))
    return pi


if __name__ == "__main__":
    main()
