# Pi Calculator with Python and Multiprocessing

## Project Overview

This project started as a nostalgic revisiting of a high school project aimed at calculating digits of Pi. 
The original project ran on a Pentium computer and managed to calculate 100k digits using Visual Basic over 
4 days. Fast forward to today, this project attempts to break that record using modern hardware and 
techniques. It leverages Python's multiprocessing capabilities and now uses the `mpmath` library for 
high-precision arithmetic, providing a significant speed-up over the original `gmpy2` implementation.

## Requirements

- Python 3.x
- mpmath
- click

To install the dependencies, use the `requirements.txt` file provided in the repository:

```bash
pip install -r requirements.txt
```

## Usage

To run the script and calculate digits of Pi:

```bash
python3 compute_pi_script.py --num-digits 1000
```

Replace `1000` with the number of digits you want to calculate.

## Features

- Utilizes Python's `multiprocessing.Pool` for parallel calculations.
- Uses the Chudnovsky algorithm for Pi calculation.
- Employs the `mpmath` library for optimized high-precision arithmetic, providing a faster and more accurate result.

## Performance

The switch to `mpmath` has shown significant improvements in both speed and accuracy. You can calculate up to 10,000 digits in a fraction of the time it used to take with `gmpy2`.

## Author

This project was created with the assistance of ChatGPT by OpenAI.
