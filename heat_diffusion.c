#include <CL/cl.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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



	//release context and command queue
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	return 0;
}
