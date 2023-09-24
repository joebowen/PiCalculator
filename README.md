# Pi Calculator with Python and Multiprocessing

## Project Overview

This project is a nostalgic revisiting of a high school project aimed at calculating digits of Pi. 
The original project was executed on a Pentium computer and managed to calculate 100k digits using Visual Basic 
and 4 days of computation. Fast forward to today, this project attempts to break that record using modern hardware 
and techniques. Specifically, it leverages Python's multiprocessing capabilities and the gmpy2 library for 
arbitrary-precision arithmetic.

## Requirements

- Python 3.x
- gmpy2
- click

To install the dependencies, use the `requirements.txt` file provided in the repository:

```
pip install -r requirements.txt
```

## Usage

To run the script and calculate digits of Pi:

```
python3 compute_pi_script.py --num-digits 1000
```

Replace `1000` with the number of digits you want to calculate.

## Features

- Utilizes Python's `multiprocessing.Pool` for parallel calculations.
- Uses the Chudnovsky algorithm for Pi calculation.
- Employs the gmpy2 library for optimized arbitrary-precision arithmetic.

## Author

This project was created with the assistance of ChatGPT by OpenAI.
