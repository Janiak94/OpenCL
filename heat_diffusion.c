#include <CL/cl.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_SOURCE_SIZE 10000

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
			if(!(*n_iterations = (size_t) strtol(&argv[i][2],NULL,10))){
				printf("Number of iterations (-n) must be greater than 0\n");
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
	double **output,
	size_t height,
	size_t width,
	double initial
){
	*output = calloc(height*width, sizeof(double));
	*input = calloc(height*width, sizeof(double));
	//this is buggy, fix
	if(height%2 && width%2){
		input[0][(height-1)*height/2+width/2] = initial;
	}else if(height%2 == 0 && width%2 == 0){
		input[0][height*(height/2)+width/2] = initial/4;
		input[0][height*(height/2-1)+width/2] = initial/4;
		input[0][height*height/2+width/2-1] = initial/4;
		input[0][height*(height/2-1)+width/2-1] = initial/4;
	}else if(height%2 == 0){
		input[0][height*height/2+width/2] = initial/4;
		input[0][height*(height/2-1)+width/2] = initial/4;
	}else{
		input[0][(height-1)*height/2+width/2-1] = initial/4;
		input[0][height*height/2+width/2-1] = initial/4;
	}
	//print result here
	for(int i = 0; i < height; ++i){
		for(int j = 0; j < width; ++j){
	//		printf("%.2f ", input[0][height*i+j]);
		}
	//	printf("\n");
	}
	//	printf("\n");
}

int main(int argc, char *argv[]){
	/*
 	 * read and set arguments
 	 */
	size_t height, width, n_iterations;
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
	size_t source_size, ix_m = width*height;
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
  	if (clGetPlatformIDs(1, &platform_id, &nmb_platforms) != CL_SUCCESS) {
    		printf( "cannot get platform\n" );
    		return 1;
  	}
	//get device info
	cl_device_id device_id;
  	cl_uint nmb_devices;
  	if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1,
                &device_id, &nmb_devices) != CL_SUCCESS) {
    		printf( "cannot get device\n" );
    		return 1;
  	}
	//setup context
	cl_context context;
  	cl_context_properties properties[] =
  	{
    		CL_CONTEXT_PLATFORM,
    		(cl_context_properties) platform_id,
    		0
 	};
  	context = clCreateContext(properties, 1, &device_id, NULL, NULL, &error);

	//setup command queue
	cl_command_queue command_queue;
  	command_queue = clCreateCommandQueue(context, device_id, 0, &error);
  	if (error != CL_SUCCESS) {
    		printf("cannot create context\n");
    		return 1;
  	}

	//create program
	cl_program program;
	program = clCreateProgramWithSource(
		context,
		1,
		(const char **)&source_str,
		(const size_t *)&source_size,
		&error
	);
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
	if(error != CL_SUCCESS){
		printf("Error while creating heat kernel\n");
	}
	//create kernel for moving output to input between iterations
	cl_kernel kernel_move;
	kernel_move = clCreateKernel(program, "move_buffer", &error);
	if(error != CL_SUCCESS){
		printf("Error while creating move kernel\n");
	}
	//create kernel for calculation of partial sum of temperatures
	cl_kernel kernel_partial;
	kernel_partial = clCreateKernel(program, "partial_sum", &error);
	if(error != CL_SUCCESS){
		printf("Error while creating partial kernel\n");
	}
	//create kernel for calculation of average diff
/*	cl_kernel kernel_diff;
	kernel_diff = clCreateKernel(program, "average_difference", &error);
	if(error != CL_SUCCESS){
		printf("Error while creating diff kernel\n");
	}
*/
	//create buffers to hold grid before and after 
	cl_mem input_buffer, output_buffer;
	input_buffer = clCreateBuffer(
		context, CL_MEM_READ_WRITE, sizeof(double)*ix_m, NULL, &error);
	if(error != CL_SUCCESS){
		printf("Error while creating input buffer\n");
		return 1;
	}
	output_buffer = clCreateBuffer(
		context, CL_MEM_WRITE_ONLY, sizeof(double)*ix_m, NULL, &error);
	if(error != CL_SUCCESS){
		printf("Error while creating output buffer\n");
		return 1;
	}
	//probe the max work group size for the partial sums, so a
	//partial sums buffer can be created
	size_t max_work_group_size;
	if(clGetKernelWorkGroupInfo(kernel_partial, device_id, CL_KERNEL_WORK_GROUP_SIZE,
                         sizeof(size_t), &max_work_group_size, NULL) != CL_SUCCESS){
		printf("Error when fetching max work group size\n");
		return 1;
	}

	// create local sum buffer	
	cl_mem local_sums_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*max_work_group_size,NULL,&error);
	if(error != CL_SUCCESS){
		printf("Error creating local buffer\n");
		return 1;
	}
	size_t n_work_groups = (ix_m/max_work_group_size == 0 ? 1024/32 : ix_m/max_work_group_size);
	cl_mem partial_sums_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		sizeof(double)*n_work_groups,NULL,&error);
	if(error != CL_SUCCESS){
		printf("Error creating partial sums buffer\n");
		return 1;
	}
	
	//create input and output arrays
	double *input, *output;
	initialize_grid(&input,&output, height, width, initial_central);
	
	
	//enqueue input buffer
	if(clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE,
  		0, ix_m*sizeof(double), input, 0, NULL, NULL) != CL_SUCCESS){
		printf("Error enqueueing input\n");
		free(input);
		free(output);
		return 1;
	}
	
	//set kernel arguments
	if(clSetKernelArg(kernel_heat, 0, sizeof(cl_mem), &input_buffer) != CL_SUCCESS ||
	clSetKernelArg(kernel_heat, 1, sizeof(float), &diffusion_constant) != CL_SUCCESS ||
	clSetKernelArg(kernel_heat, 2, sizeof(unsigned int), &width) != CL_SUCCESS ||
	clSetKernelArg(kernel_heat, 3, sizeof(unsigned int), &height) != CL_SUCCESS ||
	clSetKernelArg(kernel_heat, 4, sizeof(cl_mem), &output_buffer) != CL_SUCCESS){
		printf("Error passing argumnets to heat kernal\n");
		free(input);
		free(output);
		return 1;
	}
	if(clSetKernelArg(kernel_move, 0, sizeof(cl_mem), &input_buffer) != CL_SUCCESS ||
	clSetKernelArg(kernel_move, 1, sizeof(cl_mem), &output_buffer) != CL_SUCCESS){
		printf("Error passing argumnets to move kernal\n");
		free(input);
		free(output);
		return 1;
	}
	if(clSetKernelArg(kernel_partial, 0, sizeof(cl_mem), &output_buffer) != CL_SUCCESS ||
	clSetKernelArg(kernel_partial, 1, sizeof(cl_mem), NULL) != CL_SUCCESS ||
	clSetKernelArg(kernel_partial, 2, sizeof(cl_mem), &partial_sums_buffer) != CL_SUCCESS){
		printf("Error passing argumnets to partial sums kernal\n");
		free(input);
		free(output);
		return 1;
	}
		
	//set length of argumnet vector
	const size_t global = ix_m;
	//Execute
	if(clEnqueueNDRangeKernel(command_queue, kernel_heat, 1, NULL,
  		(const size_t *)&global, NULL, 0, NULL, NULL) != CL_SUCCESS){
		printf("Error execute\n");
		free(input);
		free(output);
		return 1;
	}
	for(size_t i = 1; i < n_iterations; ++i){	//move output to input and start next iteration
		if(clEnqueueNDRangeKernel(command_queue, kernel_move, 1, NULL,
  			(const size_t *)&global, NULL, 0, NULL, NULL) != CL_SUCCESS){
			printf("Error execute\n");
			free(input);
			free(output);
			return 1;
		}
		if(clEnqueueNDRangeKernel(command_queue, kernel_heat, 1, NULL,
  			(const size_t *)&global, NULL, 0, NULL, NULL) != CL_SUCCESS){
			printf("Error execute\n");
			free(input);
			free(output);
			return 1;
		}
	}
	if(clEnqueueNDRangeKernel(command_queue, kernel_partial, 1, NULL,
		(const size_t *)&global, NULL, 0, NULL,NULL) != CL_SUCCESS){
			printf("Error execute partial\n");
			free(input);
			free(output);
			return 1;
	}
	//read partial sums
	double *partial_sums = (double*) malloc(sizeof(double)*n_work_groups);
	if(clEnqueueReadBuffer(command_queue, partial_sums_buffer, CL_TRUE,
		0, sizeof(double)*n_work_groups,partial_sums, 0, NULL, NULL) != CL_SUCCESS){
		printf("Error reading partial sums buffer\n");
		free(input);
		free(output);
		free(partial_sums);
		return 1;
	}
	double average = 0;
	for(size_t ix = 0; ix < n_work_groups; ++ix){
		average += partial_sums[ix];
		printf("%f \n", partial_sums[ix]);
	}
	free(partial_sums);
	average /= global;
	printf("Average temperature: %f\n", average);
		

	//read buffer
	if(clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE,
  		0, ix_m*sizeof(double), output, 0, NULL, NULL) != CL_SUCCESS){
		printf("Error reading buffer\n");
		free(input);
		free(output);
		return 1;
	}

	if(clFinish(command_queue) != CL_SUCCESS){
		printf("Error while finish!!\n");
		free(input);
		free(output);
		return 1;
	}

	//print result here
	for(int i = 0; i < height; ++i){
		for(int j = 0; j < width; ++j){
			printf("%.2f ", output[height*i+j]);
		}
		printf("\n");
	}

	free(input);
	free(output);

	//release context and command queue
	if(clReleaseCommandQueue(command_queue) != CL_SUCCESS){
		printf("Error releasing command queue\n");
		return 1;
	}
	if(clReleaseContext(context) != CL_SUCCESS){
		printf("Error releasing context\n");
		return 1;
	}
	if(clReleaseMemObject(input_buffer) != CL_SUCCESS){
		printf("Error releasing input buffer\n");
	}
	if(clReleaseMemObject(output_buffer) != CL_SUCCESS){
		printf("Error releasing output buffer\n");
	}
	
	return 0;
}
