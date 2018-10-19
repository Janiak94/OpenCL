#include <CL/cl.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_SOURCE_SIZE 100000
#define GROUP_SIZE 32 //prefered work group size multiple

void error_handler(char err[], int code){
if(code != CL_SUCCESS) {
		printf("%s, Error code:%d\n", err, code);
		exit(EXIT_FAILURE);
	}
}

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

//set the initial center value and allocate input and output arrays
void initialize_grid(
	double **input,
	const size_t height,
	const size_t width,
	size_t *grid_height,
	size_t *grid_width,
	const double initial
){
	//make sure the grid is divisible in 8x4=32 chunks for work groups
	*grid_height = (height%4 == 0 ? height : height-height%4 + 4);
	*grid_width = (width%8 == 0 ? width : width-width%8 + 8);
	*input = calloc(*grid_height* *grid_width, sizeof(double));
	if(*input == NULL){
		exit(EXIT_FAILURE);
	}

	//this is buggy, fix
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
	/*
 	 * read and set arguments
 	 */
	size_t height, grid_height, width, grid_width, n_iterations;
	double initial_central;
	float diffusion_constant;	
	set_args(
		argc,
		argv,
		&height,
		&width,
		&initial_central,
		&diffusion_constant,
		&n_iterations
	);
	
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
	//get platform info
  	cl_platform_id platform_id;
	cl_uint nmb_platforms;
  	error = clGetPlatformIDs(1, &platform_id, &nmb_platforms);
	error_handler("Could not get platform id",error);
	
	//get device info
	cl_device_id device_id;
  	cl_uint nmb_devices;
  	error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1,
                &device_id, &nmb_devices);
    	error_handler("cannot get device",error );
	
	//setup context
	cl_context context;
  	cl_context_properties properties[] =
  	{
    		CL_CONTEXT_PLATFORM,
    		(cl_context_properties) platform_id,
    		0
 	};
  	context = clCreateContext(properties, 1, &device_id, NULL, NULL, &error);
	error_handler("Error creating context",error);

	//setup command queue
	cl_command_queue command_queue;
  	command_queue = clCreateCommandQueue(context, device_id, 0, &error);
    	error_handler("cannot create context",error);

	//create program
	cl_program program;
	program = clCreateProgramWithSource(
		context,
		1,
		(const char **)&source_str,
		(const size_t *)&source_size,
		&error
	);
	error_handler("Error creating program",error);

	error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(error != CL_SUCCESS){ //get the log from build, and print to stdout
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
	
	//create kernel for heat diffusion
	cl_kernel kernel_heat;
	kernel_heat = clCreateKernel(program, "local_heat_diffusion", &error);
	error_handler("Error while creating heat kernel",error);
	
	//create kernel for calculation of partial sum of temperatures
	cl_kernel kernel_partial;
	kernel_partial = clCreateKernel(program, "partial_sum", &error);
	error_handler("Error while creating partial kernel",error);

	//create kernel for absolute difference
	cl_kernel kernel_diff;
	kernel_diff = clCreateKernel(program, "difference", &error);
	error_handler("Error while creating difference kernel",error);

	//create buffers to hold grid before and after 
	double *grid;
	initialize_grid(&grid, height, width, &grid_height, &grid_width, initial_central);
	const size_t global_work_items = grid_height*grid_width;	
	cl_mem grid_buffer, partial_sums_buffer;
	grid_buffer = clCreateBuffer(
		context, CL_MEM_READ_WRITE, sizeof(double)*global_work_items, NULL, &error);
	error_handler("Error while creating input buffer",error);

	//look this over, how to calculate the appropiate size?
	const size_t local_size = GROUP_SIZE;
	const size_t n_work_groups = global_work_items/local_size;
	partial_sums_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		sizeof(double)*n_work_groups,NULL,&error);
	error_handler("Error creating partial sums buffer",error);
		
	//enqueue input buffer
	error = clEnqueueWriteBuffer(command_queue, grid_buffer, CL_TRUE,
  		0, sizeof(double)*global_work_items, grid, 0, NULL, NULL);
	error_handler("Error enqueueing grid",error);
	
	//set kernel arguments
	if(clSetKernelArg(kernel_heat, 0, sizeof(cl_mem), &grid_buffer) != CL_SUCCESS ||
	clSetKernelArg(kernel_heat, 1, sizeof(float), &diffusion_constant) != CL_SUCCESS ||
	clSetKernelArg(kernel_heat, 2, sizeof(unsigned int), &width) != CL_SUCCESS ||
	clSetKernelArg(kernel_heat, 3, sizeof(unsigned int), &height) != CL_SUCCESS ||
	clSetKernelArg(kernel_heat, 4, sizeof(unsigned int), &grid_width) != CL_SUCCESS ||
	clSetKernelArg(kernel_heat, 5, sizeof(unsigned int), &grid_height) != CL_SUCCESS){
		printf("Error passing argumnets to heat kernal\n");
		free(grid);
		return 1;
	}
	if(clSetKernelArg(kernel_partial, 0, sizeof(cl_mem), &grid_buffer) != CL_SUCCESS ||
	clSetKernelArg(kernel_partial, 1, local_size*sizeof(cl_double), NULL) != CL_SUCCESS ||
	clSetKernelArg(kernel_partial, 2, sizeof(cl_mem), &partial_sums_buffer) != CL_SUCCESS){
		printf("Error passing argumnets to partial sums kernal\n");
		free(grid);
		return 1;
	}
		
	//set length of argumnet vector
	//Execute
	for(size_t i = 0; i < n_iterations; ++i){	//move output to input and start next iteration
		error = clEnqueueNDRangeKernel(command_queue, kernel_heat, 1, NULL,
  			(const size_t *)&global_work_items, NULL, 0, NULL, NULL);
		error_handler("Error execute",error);
	}
	//this kernel seems to access illegal memory
	const size_t test = 32;
	error = clEnqueueNDRangeKernel(command_queue, kernel_partial, 1, NULL,
		(const size_t *)&global_work_items, (const size_t *)&local_size, 0, NULL,NULL);
	error_handler("Error execute partial",error);
	
	if(clFinish(command_queue) != CL_SUCCESS){
		printf("Error while finish!!\n");
		free(grid);
		return 1;
	}

	//read partial sums
	//will yield CL_OUT_OF_RESOURCES if the dimensions get larger
	double *partial_sums = (double*) malloc(sizeof(double)*n_work_groups);
	error = clEnqueueReadBuffer(command_queue, partial_sums_buffer, CL_TRUE,
		0, sizeof(double)*n_work_groups,partial_sums, 0, NULL, NULL);
	error_handler("Error reading partial sums buffer", error);

/*	//read buffer
	error = clEnqueueReadBuffer(command_queue, grid_buffer, CL_TRUE,
  		0, global_work_items*sizeof(double), grid, 0, NULL, NULL);
	error_handler("Error reading temperature buffer",error);
*/
	double average = 0;
	for(size_t ix = 0; ix < n_work_groups; ++ix){
		average += partial_sums[ix];
	}
	average /= width*height;
	printf("Average temperature: %.0f\n", average);	

	error = clSetKernelArg(kernel_diff, 0, sizeof(cl_mem), &grid_buffer);
	error_handler("Arg 1",error);
	error = clSetKernelArg(kernel_diff, 1, sizeof(int), &height);
	error_handler("Arg 2", error);
	error = clSetKernelArg(kernel_diff, 2, sizeof(int), &width);
	error_handler("Arg 3", error);
	error = clSetKernelArg(kernel_diff, 3, sizeof(int), &grid_width);
	error_handler("Arg 4", error);
	error = clSetKernelArg(kernel_diff, 4, sizeof(double), &average);
	error_handler("Arg 5",error);
	
	error = clEnqueueNDRangeKernel(command_queue, kernel_diff, 1, NULL, (const size_t *)&global_work_items,
		NULL, 0, NULL, NULL);
	error_handler("Error while executing diff kernel",error);

	error = clEnqueueNDRangeKernel(command_queue, kernel_partial, 1, NULL,
		(const size_t *)&global_work_items, (const size_t *)&local_size, 0, NULL,NULL);
	error_handler("Error execute partial",error);

	if(clFinish(command_queue) != CL_SUCCESS){
		printf("Error while finish!!\n");
		free(grid);
		return 1;
	}

	error = clEnqueueReadBuffer(command_queue, partial_sums_buffer, CL_TRUE,
		0, sizeof(double)*n_work_groups,partial_sums, 0, NULL, NULL);
	error_handler("Error reading partial sums buffer", error);

	double average_diff = 0;
	for(size_t ix = 0; ix < n_work_groups; ++ix){
		average_diff += partial_sums[ix];
	}
	free(partial_sums);
	average_diff /= width*height;
	printf("Average temperature difference: %.0f\n", average_diff);	

	//release context and command queue
	error = clReleaseCommandQueue(command_queue);
	error = clReleaseContext(context);
	error = clReleaseProgram(program);
	error = clReleaseMemObject(partial_sums_buffer);
	error = clReleaseMemObject(grid_buffer);
	error = clReleaseKernel(kernel_heat);
	error = clReleaseKernel(kernel_partial);
	return 0;
}
