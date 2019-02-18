# gpuroofperf-toolkit

The gpuroofperf-toolkit is a performance prediction toolkit for CUDA GPU
kernel programs that is comprised of 2 distinct tools:

1. *gpuroofperf-bench*: The CUDA GPU parameter evaluation tool (benchmark)
2. *gpuroofperf-tool*: The performance model tool (performance prediction)

gpuroofperf-bench
--------------

This benchmark tool assesses the performance of a GPU in various aspects
(SP/DP/integer computations, memory bandwidth, load/store operations) and
the results are optionally saved as GPU parameters in a CSV file. These
parameters are intended to be exploited by *gpuroofperf-tool*.

This tool is written in C++ and CUDA, and can be build by using cmake, as
follows:

```
mkdir build
cd build
cmake ../gpuroofperf-bench
```

For a usage example please refer to paragraph *"Typical usage example"* below.

gpuroofperf-tool
--------------

This tool applies the principles as dictated by the performance model referred
at the bottom, and it predicts performance for all GPUs for which there are
benchmark results available.
In practice, it leverages the *nvprof* NVidia profiler for
tracing the GPU metric values that are important in the evaluation of the
performance model parameter values.
Therefore having installed *nvprof* via CUDA toolkit is a prerequisite.

Use of the tool is divided in two stages that can be run either all together or
independently:
1. Kernel parameter extraction

   In this stage, the GPU application is executed, the relevant hardware
   metrics are captured and the required model parameters are evaluated.
   These parameters can be stored in a file for a later use in the next
   stage.

2. Performance prediction

  The stored kernel parameters combined with the GPU parameters (benchmark
  outcome) are used to predict the GPU kernel performance on a particular GPU.
  This procedure involves the designation of the particular kernels as compute
  or memory bound.

This tool is written in Python 3, thus it doesn't require compilation.
See next paragraph for a usage example of the toolchain.

Typical usage example
--------------

Using the benchmark tool without arguments just outputs the available options and
the available CUDA GPUs:
```
$ ./gpuroofperf-bench
gpuroofperf-bench (CUDA) microbenchmarking tool

Syntax: gpuroofbench device_index [options]

Command line options:
-h or --help:            Show this help
-o or --output=filename: Append CSV output to file


Valid device indexes:
[1] GeForce GTX 960
[2] GeForce GTX 660
```

The following example usage specifies the GPU index (GTX-660) and the output
filename for saving the performance results:
```
$ ./gpuroofperf-bench 2 -o gpus.csv
gpuroofperf-bench (CUDA) microbenchmarking tool

------------------------ Device specifications ------------------------
Device:              GeForce GTX 660
CUDA driver version: 10.0
GPU clock rate:      1097 MHz
Memory clock rate:   1502 MHz
Memory bus width:    192 bits
WarpSize:            32
L2 cache size:       384 KB
Total global mem:    1994 MB
ECC enabled:         No
Compute Capability:  3.0
Total SPs:           960 (5 MPs x 192 SPs/MP)
Compute throughput:  2107.2 GFlops (theoretical single precision FMAs)
Memory bandwidth:    144.192 GB/sec
-----------------------------------------------------------------------

Running compute intensive benchmarks...    Done.
Running load/store intensive benchmarks... Done.
Running memory intensive benchmarks...     Done.
Running L2 cache intensive benchmarks...   Done.

Execution results for 'GeForce GTX 660':
GFLOPS - SP:    1863.569
GFLOPS - DP:      81.791
GIOPS - MAD:     359.020
GBSEC - DRAM:    115.943
GBSEC - L2:      179.127
GOPS - SHMEM:    179.729
GIOPS - ADD:     414.935

Results appended to file 'gpus.csv'
```

A typical usage of *gpuroofperf-tool* for the construction of kernel parameter
values on the execution of a sample matrix multiplication program (1st stage)
on e.g. GTX-960:
```
$ ./gpuroofperf-cli.py -o matrixmul.json ../matrixmul
gpuroofperf-tool v.0.9.0 (CLI)


------- Running kernel inspection -------
Retrieving GPU information... Done
Multiple GPUs found. Please choose one of them below:
1. GeForce GTX 960
2. GeForce GTX 660
Select GPU index (1-2):1
Profiling on GPU: GeForce GTX 960
Invocation: "../matrixmul"
Running simple profiling... Done
GPU kernel functions invoked:
1. matrixMul (Time(%):6.64, 2.9151 msecs, 1 invocations)
Please give the subject kernel indices (comma separated) (1-1 or default:all kernels):
Selected kernels: "matrixMul"
Running trace profiling... Done
Running metric profiling (floating point operations, 9 total metrics)... Done
Running metric profiling (memory transactions, 7 total metrics)... Done
Running metric profiling (instruction counts, 24 total metrics)... Done
Running metric profiling (utilization, 13 total metrics)... Done
Kernel inspection done!
Querying GPU driver version (via nvidia-smi)... Done

Kernel parameters saved to  "matrixmul.json"

------- Primary kernel parameters -------
Kernel name: matrixMul(float*, float*, float*, int, int)
        Kernel type:            Floating point (SP)
        Compute operations:     1048576000 ops
        DRAM accesses:          72716160 bytes
        Computation efficiency: 1.00000
        Instruction mix:
                Computations:   34.73%
                Load/Stores:    47.80%
                Other:          17.47%
```

The essential kernel parameters are evident.
If you inspect the produced *"matrixmul.json"* you'll find even more details.
Typical usage for GPU performance prediction (2nd stage) follows:
```
$ ./gpuroofperf-cli.py -i matrixmul.json -g gpus.csv -s mm-prediction.json
gpuroofperf-tool v.0.9.0 (CLI)

Retrieved kernel parameters from "matrixmul.json"

------- Primary kernel parameters -------
Kernel name: matrixMul(float*, float*, float*, int, int)
        Kernel type:            Floating point (SP)
        Compute operations:     1048576000 ops
        DRAM accesses:          72716160 bytes
        Computation efficiency: 1.00000
        Instruction mix:
                Computations:   34.73%
                Load/Stores:    47.80%
                Other:          17.47%


--- Conducting performance estimation ---
Reading GPU specifications from "gpus.csv"
Estimating GPU performance for 2 GPUs (GeForce GTX 660, GeForce GTX 960)

+-----------------+----------------------------------+----------------------------------+
|       GPU       | matrixMul(float*, float*, flo... | *********** Summary ***********  |
|                 | estimated time (msecs) |  bound  |     total estimated time (msecs) |
+-----------------+----------------------------------+----------------------------------+
| GeForce GTX 660 |                5.21397 | Compute |                          5.21397 |
| GeForce GTX 960 |                2.56032 | Compute |                          2.56032 |
+-----------------+----------------------------------+----------------------------------+
Saving prediction results to "mm-prediction.json"
```

As it is evident the performance prediction tool has identified the kernel
as compute bound for both GPUs.
For more information see *"mm-prediction.json"*.

Publications
--------------

This toolkit applies the proposed principles as suggested in the following
research paper. Thus, if you find this toolkit useful in research work please
provide citation to any of the following paper:

Elias Konstantinidis, Yiannis Cotronis,
"A quantitative roofline model for GPU kernel performance estimation using
micro-benchmarks and hardware metric profiling",
Journal of Parallel and Distributed Computing, Volume 107, September 2017,
Pages 37-56, ISSN 0743-7315,
https://doi.org/10.1016/j.jpdc.2017.04.002.  
URL: http://www.sciencedirect.com/science/article/pii/S0743731517301247
