__kernel void
local_heat_diffusion(
	__global double * h,
	const float c,
	const unsigned int width,
	const unsigned int height,
	const unsigned int grid_width
)
{
	uint global_id = get_global_id(0);
	uint ix = global_id/grid_width;
	uint jx = global_id%grid_width;
	double h_local = 0;
	if(ix > 0 && jx > 0 && ix < height-1 && jx < width-1){
		h_local =
		h[grid_width*ix+jx] + c*((h[grid_width*(ix-1)+jx] + h[grid_width*(ix+1)+jx] +
			h[grid_width*ix+jx-1] + h[grid_width*ix+jx+1])/4 -
			h[grid_width*ix+jx]);
	}
	else if(ix < height && jx < width){
		int local_temp = 0;
		if(ix > 0){
			local_temp += h[grid_width*(ix-1)+jx];
		}
		if(jx > 0){
			local_temp += h[grid_width*ix+jx-1];
		}
		if(ix < height-1){
			local_temp += h[grid_width*(ix+1)+jx];
		}
		if(jx < width-1){
			local_temp += h[grid_width*ix+jx+1];
		}
		h_local = h[grid_width*ix+jx] +
			c*(local_temp/4-h[grid_width*ix+jx]);
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	h[global_id] = h_local;
}

__kernel void
partial_sum(
	__global const double * input,
	__local double * local_sums,
	__global double * partial_sums
)
{
	uint local_id = get_local_id(0);
  	uint group_size = get_local_size(0);

  	// Copy from global to local memory
  	local_sums[local_id] = input[get_global_id(0)];

  	// Loop for computing local_sums : divide workgroup into 2 parts
  	for (uint i = group_size/2; i>0; i /=2)
     	{
      		// Waiting for each 2x2 addition into given workgroup
      		barrier(CLK_LOCAL_MEM_FENCE);

      		// Add elements 2 by 2 between local_id and local_id + i
      		if (local_id < i)
        	local_sums[local_id] += local_sums[local_id + i];
     	}

  	// Write result into partialSums[nWorkGroups]
  	if (local_id == 0)
    		partial_sums[get_group_id(0)] = local_sums[0];
}

__kernel void
difference(
	__global double * input,
	const int height,
	const int width,
	const int grid_width,
	const double average
)
{
	uint global_id = get_global_id(0);
	uint ix = global_id/grid_width;
	uint jx = global_id%grid_width;

	double diff = input[global_id] - average;
	barrier(CLK_GLOBAL_MEM_FENCE);
	if(ix < height && jx < width)
		input[global_id] = (diff < 0 ? -diff : diff);
}

