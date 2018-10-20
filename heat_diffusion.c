#include <CL/cl.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>

#define MAX_SOURCE_SIZE 100000
#define GROUP_SIZE 32

/* SIMD summation of a vector */
static inline double sum_vector(double *vector, size_t len){
	__m512d vsum = _mm512_setzero_pd();
	for(size_t i = 0; i < len; i+=8){
		__m512d v = _mm512_load_pd(&vector[i]);
		vsum = _mm512_add_pd(vsum,v);
	}
	return _mm512_reduce_add_pd(vsum);
}

/* handles the many potential errors from OpenCL */
void error_handler(char err[], int code){
if(code != CL_SUCCESS) {
		printf("%s, Error code:%d\n", err, code);
		exit(EXIT_FAILURE);
	}
}

/* parse and set argument */
void set_args(
	int argc,
	char *argv[],
	size_t *height,
	size_t *width,
	double *initial_central,
	float *diffusion_constant,
	size_t *n_iterations
){
	if(argc != 6){
		printf("Not enough arguments\n");
		exit(1);
	}
	for(unsigned i = 1; i < argc; ++i){
		if(argv[i][0] == '-' && argv[i][1] == 'i'){
			char *ptr;
			short e;
			*initial_central = (double)strtol(&argv[i][2],&ptr,10);
			if(*ptr == 'e'){
				e = (short) strtol(ptr+1,NULL,10);
				for(unsigned short i = 0; i < e; ++i){
					*initial_central*=10.0;
				}
			}else if(*ptr != '\0'){
				printf("Wrong format: %s\n", ptr);
				exit(1);
			}
			continue;
		}else if(argv[i][0] == '-' && argv[i][1] == 'd'){
			*diffusion_constant = atof(&argv[i][2]);
			continue;
		}else if(argv[i][0] == '-' && argv[i][1] == 'n'){
			if((*n_iterations = (size_t) strtol(&argv[i][2],NULL,10)) < 0){
				printf("Number of iterations (-n) must be greater than or equal to 0\n");
				exit(1);
			}
			continue;
		}else if((*height = strtol(argv[i],NULL,10))){
			if(!(*width = strtol(argv[++i],NULL,10))){
				printf("Height not set\n");
				exit(1);
			}
			continue;
		}
		printf("Not a valid argument: %s\n", argv[i]);
		exit(1);
	}
}

/* set the initial center value and allocate grid */
void initialize_grid(
	double **input,
	const size_t height,
	const size_t width,
	size_t *grid_height,
	size_t *grid_width,
	const double initial
){
	/* make sure the grid is divisible in 8x4=32 chunks for work groups */
	*grid_height = (height%4 == 0 ? height : height-height%4 + 4);
	*grid_width = (width%8 == 0 ? width : width-width%8 + 8);
	*input = calloc(*grid_height* *grid_width, sizeof(double));
	if(*input == NULL){
		exit(EXIT_FAILURE);
	}

	if(height%2 && width%2){
		input[0][(*grid_width)*(height/2)+width/2] = initial;
	}else if(height%2 == 0 && width%2 == 0){
		input[0][(*grid_width)*(height/2-1)+width/2] = initial/4;
		input[0][(*grid_width)*(height/2-1)+width/2-1] = initial/4;
		input[0][(*grid_width)*(height/2)+width/2] = initial/4;
		input[0][(*grid_width)*(height/2)+width/2-1] = initial/4;
	}else if(height%2 == 0){
		input[0][(*grid_width)*(height/2)+width/2] = initial/4;
		input[0][(*grid_width)*(height/2-1)+width/2] = initial/4;
	}else{
		input[0][(*grid_width)*(height/2)+width/2] = initial/4;
		input[0][(*grid_width)*(height/2)+width/2-1] = initial/4;
	}
}

int main(int argc, char *argv[]){
	size_t height, grid_height, width, grid_width, n_iterations;
	double initial_central;
	float diffusion_constant;	
 	/* read and set arguments */
	set_args(
		argc,
		argv,
		&height,
		&width,
		&initial_central,
		&diffusion_constant,
		&n_iterations
	);

	/* read .cl source file */	
	char *source_str = (char*) malloc(MAX_SOURCE_SIZE);
	size_t source_size;
	FILE *fp;
	if(!(fp = fopen("cl_heat_diffusion.cl","r"))){
		printf("Error opening source file\n");
		return 1;
	}
	source_size = fread(source_str, sizeof(char), MAX_SOURCE_SIZE, fp);
	fclose(fp);

	cl_int error;
	/* get platform and device info */
  	cl_platform_id platform_id;
	cl_uint nmb_platforms;
  	error = clGetPlatformIDs(1, &platform_id, &nmb_platforms);
	error_handler("Could not get platform id",error);
	
	cl_device_id device_id;
  	cl_uint nmb_devices;
  	error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1,
                &device_id, &nmb_devices);
    	error_handler("cannot get device",error );
	
	/* create context */
	cl_context context;
  	cl_context_properties properties[] =
  	{
    		CL_CONTEXT_PLATFORM,
    		(cl_context_properties) platform_id,
    		0
 	};
  	context = clCreateContext(properties, 1, &device_id, NULL, NULL, &error);
	error_handler("Error creating context",error);

	/* create command queue */
	cl_command_queue command_queue;
  	command_queue = clCreateCommandQueue(context, device_id, 0, &error);
    	error_handler("cannot create context",error);

	/* create program */
	cl_program program;
	program = clCreateProgramWithSource(
		context,
		1,
		(const char **)&source_str,
		(const size_t *)&source_size,
		&error
	);
	error_handler("Error creating program",error);

	/* build program and print build log if build failed */
	error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(error != CL_SUCCESS){
		printf("Cannot build program, log:\n");
		size_t log_size = 0;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,0,
		NULL, &log_size);

		char *log = calloc(log_size, sizeof(char));
		if(log == NULL){
			printf("Could not allocate memory\n");
			return 1;
		}
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,0,
		log, &log_size);
		printf("%s\n",log);
		free(log);
		free(source_str);
		return 1;
	}
	free(source_str);
	
	/* create kernels */
	cl_kernel kernel_heat;
	kernel_heat = clCreateKernel(program, "local_heat_diffusion", &error);
	error_handler("Error while creating heat kernel",error);
	
	cl_kernel kernel_partial;
	kernel_partial = clCreateKernel(program, "partial_sum", &error);
	error_handler("Error while creating partial kernel",error);

	cl_kernel kernel_diff;
	kernel_diff = clCreateKernel(program, "difference", &error);
	error_handler("Error while creating difference kernel",error);

	/* buffers for grid and partial sums */
	double *grid;
	initialize_grid(&grid, height, width, &grid_height, &grid_width, initial_central);
	const size_t global_work_items = grid_height*grid_width;	
	cl_mem grid_buffer, partial_sums_buffer;
	grid_buffer = clCreateBuffer(
		context, CL_MEM_READ_WRITE, sizeof(double)*global_work_items, NULL, &error);
	error_handler("Error while creating input buffer",error);

	const size_t local_size = GROUP_SIZE;
	const size_t n_work_groups = global_work_items/local_size;
	partial_sums_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		sizeof(double)*n_work_groups,NULL,&error);
	error_handler("Error creating partial sums buffer",error);
		
	/* write grid to global memory */
	error = clEnqueueWriteBuffer(command_queue, grid_buffer, CL_TRUE,
  		0, sizeof(double)*global_work_items, grid, 0, NULL, NULL);
	error_handler("Error enqueueing grid",error);
	
	/* set kernel arguments */
	error = clSetKernelArg(kernel_heat, 0, sizeof(cl_mem), &grid_buffer);
	error_handler("heat kernel arg 0",error);
	error = clSetKernelArg(kernel_heat, 1, sizeof(float), &diffusion_constant);
	error_handler("heat kernel arg 1",error);
	error = clSetKernelArg(kernel_heat, 2, sizeof(unsigned int), &width);
	error_handler("heat kernel arg 2",error);
	error = clSetKernelArg(kernel_heat, 3, sizeof(unsigned int), &height);
	error_handler("heat kernel arg 3",error);
	error = clSetKernelArg(kernel_heat, 4, sizeof(unsigned int), &grid_width);
	error_handler("heat kernel arg 4",error);
	error = clSetKernelArg(kernel_heat, 5, sizeof(unsigned int), &grid_height);
	error_handler("heat kernel arg 5",error);

	error = clSetKernelArg(kernel_partial, 0, sizeof(cl_mem), &grid_buffer);
	error_handler("partial kernel arg 0",error);
	error = clSetKernelArg(kernel_partial, 1, local_size*sizeof(cl_double), NULL);
	error_handler("partial kernel arg 1",error);
	error = clSetKernelArg(kernel_partial, 2, sizeof(cl_mem), &partial_sums_buffer);
	error_handler("partial kernel arg 2",error);

	/* calculate heat flow and iterate */
	for(size_t i = 0; i < n_iterations; ++i){
		error = clEnqueueNDRangeKernel(command_queue, kernel_heat, 1, NULL,
  			(const size_t *)&global_work_items, NULL, 0, NULL, NULL);
		error_handler("Error execute",error);
	}
	/* perform reduction to get partial sums */
	error = clEnqueueNDRangeKernel(command_queue, kernel_partial, 1, NULL,
		(const size_t *)&global_work_items, (const size_t *)&local_size, 0, NULL,NULL);
	error_handler("Error execute partial",error);
	
	error = clFinish(command_queue);
	error_handler("Error while finish!!",error);

	/* read buffer and sum partial sums and calculate average temperature */
	double *partial_sums = (double*)aligned_alloc(64,sizeof(double)*n_work_groups);
	double average, average_diff;
	error = clEnqueueReadBuffer(command_queue, partial_sums_buffer, CL_TRUE,
		0, sizeof(double)*n_work_groups,partial_sums, 0, NULL, NULL);
	error_handler("Error reading partial sums buffer", error);
	average = sum_vector(partial_sums,8*(n_work_groups/8));
	if(n_work_groups%8 != 0){
		for(size_t i = 8*(n_work_groups/8); i < n_work_groups; ++i){
			average += partial_sums[i];
		}
	}
	average /= width*height;
	printf("Average temperature: %.0f\n", average);	

	/* set kernel args */
	error = clSetKernelArg(kernel_diff, 0, sizeof(cl_mem), &grid_buffer);
	error_handler("diff kernel arg 0",error);
	error = clSetKernelArg(kernel_diff, 1, sizeof(int), &height);
	error_handler("diff kernel arg 1",error);
	error = clSetKernelArg(kernel_diff, 2, sizeof(int), &width);
	error_handler("diff kernel arg 2",error);
	error = clSetKernelArg(kernel_diff, 3, sizeof(int), &grid_width);
	error_handler("diff kernel arg 3",error);
	error = clSetKernelArg(kernel_diff, 4, sizeof(double), &average);
	error_handler("diff kernel arg 4",error);
	
	/* calculate difference from average for each grid point */
	error = clEnqueueNDRangeKernel(command_queue, kernel_diff, 1, NULL, (const size_t *)&global_work_items,
		NULL, 0, NULL, NULL);
	error_handler("Error while executing diff kernel",error);

	/* perform reduction on average difference, read buffers, sum and calculate average */
	error = clEnqueueNDRangeKernel(command_queue, kernel_partial, 1, NULL,
		(const size_t *)&global_work_items, (const size_t *)&local_size, 0, NULL,NULL);
	error_handler("Error execute partial",error);

	error = clFinish(command_queue);
	error_handler("Error while finish!!",error);

	error = clEnqueueReadBuffer(command_queue, partial_sums_buffer, CL_TRUE,
		0, sizeof(double)*n_work_groups,partial_sums, 0, NULL, NULL);
	error_handler("Error reading partial sums buffer", error);
	average_diff = sum_vector(partial_sums,8*(n_work_groups/8));
	if(n_work_groups%8 != 0){
		for(size_t i = 8*(n_work_groups/8); i < n_work_groups; ++i){
			average_diff += partial_sums[i];
		}
	}
	average_diff /= width*height;
	printf("Average temperature difference: %.0f\n", average_diff);	


	/* release resources */
	error = clReleaseCommandQueue(command_queue);
	error = clReleaseContext(context);
	error = clReleaseProgram(program);
	error = clReleaseMemObject(partial_sums_buffer);
	error = clReleaseMemObject(grid_buffer);
	error = clReleaseKernel(kernel_heat);
	error = clReleaseKernel(kernel_partial);
	error = clReleaseKernel(kernel_diff);
	return 0;
}
