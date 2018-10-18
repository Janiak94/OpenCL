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
	long *initial_central,
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
			*initial_central = strtol(&argv[i][2],&ptr,10);
			if(*ptr == 'e'){
				e = (short) strtol(ptr+1,NULL,10);
				for(unsigned short i = 0; i < e; ++i){
					*initial_central*=10;
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

void initialize_grid(long **input,long **output, size_t height, size_t width,
	long initial){
	*output = calloc(height*width, sizeof(long));
	*input = calloc(height*width, sizeof(long));

	if(height%2 && width%2){
		input[0][height*height/2+width/2-1] = initial;
	}else if(height%2 == 0 && width%2 == 0){
		input[0][height*height/2+width/2] = initial/4;
		input[0][height*(height/2-1)+width/2] = initial/4;
		input[0][height*height/2+width/2-1] = initial/4;
		input[0][height*(height/2-1)+width/2-1] = initial/4;
	}else if(height%2 == 0){
		input[height*height/2+width/2] = initial/4;
		input[0][height*(height/2-1)+width/2] = initial/4;
	}else{
		input[height*height/2+width/2] = initial/4;
		input[0][height*height/2+width/2-1] = initial/4;
	}
	//print result here
	for(int i = 0; i < height; ++i){
		for(int j = 0; j < width; ++j){
			printf("%ld ", input[0][height*i+j]);
		}
		printf("\n");
	}
}

int main(int argc, char *argv[]){
	/*
 	 * read and set arguments
 	 */
	size_t height, width, n_iterations;
	long initial_central;
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
	
	//create kernel
	cl_kernel kernel;
	kernel = clCreateKernel(program, "local_heat_diffusion", &error);
	if(error != CL_SUCCESS){
		printf("Error while creating kernel\n");
	}
	
	//create buffers
	cl_mem input_buffer, output_buffer;
	input_buffer = clCreateBuffer(
		context, CL_MEM_READ_ONLY, sizeof(long)*ix_m, NULL, &error);
	if(error != CL_SUCCESS){
		printf("Error while creating input buffer\n");
		return 1;
	}
	output_buffer = clCreateBuffer(
		context, CL_MEM_WRITE_ONLY, sizeof(long)*ix_m, NULL, &error);
	if(error != CL_SUCCESS){
		printf("Error while creating output buffer\n");
		return 1;
	}

	long *input, *output;
	initialize_grid(&input,&output, height, width, initial_central);
	
	//enqueue buffers
	if(clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE,
  		0, ix_m*sizeof(long), input, 0, NULL, NULL) != CL_SUCCESS){
		printf("Error enqueueing input\n");
		free(input);
		free(output);
		return 1;
	}
	
	//set kernel arguments
	if(clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer) != CL_SUCCESS ||
	clSetKernelArg(kernel, 1, sizeof(float), &diffusion_constant) != CL_SUCCESS ||
	clSetKernelArg(kernel, 2, sizeof(unsigned int), &width) != CL_SUCCESS ||
	clSetKernelArg(kernel, 3, sizeof(unsigned int), &height) != CL_SUCCESS ||
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &output_buffer) != CL_SUCCESS){
		printf("Error passing argumnets to kernal\n");
		free(input);
		free(output);
		return 1;
	}
	
	//set length of argumnet vector
	const size_t global = ix_m;
	//Execute
	if(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
  		(const size_t *)&global, NULL, 0, NULL, NULL) != CL_SUCCESS){
		printf("Error execute\n");
		free(input);
		free(output);
		return 1;
	}
	
	//read buffer
	if(clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE,
  		0, ix_m*sizeof(long), output, 0, NULL, NULL) != CL_SUCCESS){
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
			printf("%ld ", output[height*i+j]);
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
	
	clReleaseMemObject(input_buffer);
	clReleaseMemObject(output_buffer);
	return 0;
}
