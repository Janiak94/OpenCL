# Assignment report on OpenCL (hpcgp038)

## Program layout

The program is implemented in two files. One for the main program and one for the OpenCL kernels.
Within the main file we have the following functions
- sum_vector(double vector[], size_t len)
  Performs summation of all the elements in vector[] on the CPU with SIMD.
  
- error_handler(char err[], int code)
  Handles the many errors that may arrise from OpenCL.
  
- set_args(int argc, char *argv[], size_t *height, size_t *width, double *initial central, float *diffusion_constant, size_t *n_iterations)
  Parse the command line arguments.
  
- print_result(char str[], double result)
  Prints str and result, formated, to stdout.
  
- initialize_grid(double **input, const size_t height, const size_t width, size_t *grid_height, size_t *grid_width, const double initial)
  Allocates the input array, and makes sure the array size is divisible by GROUP_SIZE. The new width and height are set via
  grid_width and grid_height. This is to ensure that no work item goes out of bounds. After that the initial central value is set.
  
- main(int argc, char *argv[])
  Sets up the OpenCL program, context, command queue and kernels.
  Two kernels are used for the heat diffusion iterations. They switch between the input and output buffers, so that we
  don't have to copy buffers. This effectively halve the run time compared to when copying is used.
  After the iterations are done we calculate the average temperature using a reduction scheme. The grid is divided into
  work groups of size 32 work items. Each work group calculate their local sum using reduction on the GPU. The CPU
  sums together the partial sums with the function sum_vector (SIMD).
  Next the difference from the average temperature is set with the GPU and then its average is calculated with the reduction scheme.
  Finally the output is sent to stdout and all resources are returned.
  
In the kernel file we have the following functions
- local_heat_diffusion(__global const double *h, const float c, const uint width, const uint height, const uint grid_width, __global double *h_updated)
  Performs a single step of the heat diffusion with the input h and outputs to h_updated. This is necessary since we can only
  synchronize inside the work group, but some work items will have to access intries outside their work group.
  Both h and h_updated are stored in global memory.
  c is the heat diffusion constant, width is the effective grid width, height is the effective grid height, i.e the ones set from
  the command line.
  grid_width is the actual grid width.
  
- difference(__global double *input, const int height, const int width, const int grid_width, const double average)
  Calculates the difference from the average temperature. Each work item only accesses the entry in input that they got assigned,
  it is thread safe to save the results to input again and need not to worry about synchonization.
  
- partial_sum(__global const double *input, __local double *local_sums, __global double *partial_sums)
  Performs a reduction scheme summation of the work items in a work group and outputs the results in partial_sums.
  The input is stored in global memory.
  local_sums is stored in local memory, i.e visible inside the work group and this is the array the reduction is performed on.
  After the reduction is completed the results is stored in the work groups index in the partial_sums array, that is in global memory.

## Performance

The program performs well within the time limits as can be seen bellow. What made the largest impact on performance was to use two
kernels for the iterations instead of a single kernel and copy the output back to the input. This doubled the the performance for
that step.
The use of SIMD in the summation on the CPU didn't do that much of a difference. Since the GPU reduction reduces the
elements that needs to be handled by the CPU by a factor of 32. Even with the largest grid used during testing, it won't be
too much work for the CPU to handle. Though if larger grids were being used the benefits from SIMD would probably be noticable.

|width*heigt|100*100|10 000*10 000|100 000*100|
|-----------|-------|-------------|-----------|
|initial value|1e20|1e10|1e10|
|diffusion constant|0.01|0.02|0.6|
|number of iterations|100 000|1000|200|
|goal time|1.7s|98s|1.4s|
|actual run time|0.75s|8.86s|0.69s|
