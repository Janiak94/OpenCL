__kernel void
local_heat_diffusion(
	__global const double * h,
	const float c,
	const unsigned int width,
	const unsigned int height,
	__global double * h_updated
)
{
	//thought: handle with global barrier, will only need array that is both input and output
	unsigned int array_idx = get_global_id(0);
	unsigned int ix = array_idx/height;
	unsigned int jx = array_idx%width;
	if(ix > 0 && jx > 0 && ix < height-1 && jx < width-1){
		h_updated[height*ix+jx] =
		h[height*ix+jx] + c*((h[height*(ix-1)+jx] + h[height*(ix+1)+jx] +
			h[height*ix+jx-1] + h[height*ix+jx+1])/4 -
			h[height*ix+jx]);
		return;
	}

	int local_temp = 0;
	if(ix > 0){
		local_temp += h[height*(ix-1)+jx];
	}
	if(jx > 0){
		local_temp += h[height*ix+jx-1];
	}
	if(ix < height-1){
		local_temp += h[height*(ix+1)+jx];
	}
	if(jx < width-1){
		local_temp += h[height*ix+jx+1];
	}
	h_updated[height*ix+jx] = h[height*ix+jx] +
		 c*(local_temp/4-h[height*ix+jx]);

}

__kernel void
move_buffer(
	__global double * m1,
	__global const double * m2
)
{
	unsigned int array_idx = get_global_id(0);
	m1[array_idx] = m2[array_idx];
}	

__kernel void
partial_sum(
	__global const double * h,
	__local double * local_sum,
	__global double * partial_sum
)
{
	uint local_id = get_local_id(0);
	uint group_size = get_local_size(0);

	//get the local element we wish to add to the average
	local_sum[local_id] = h[get_global_id(0)];

	//will only work if the number of elements in h is even
	for(uint i = group_size/2; i > 0; i/=2){
		barrier(CLK_LOCAL_MEM_FENCE);
		if(local_id < i){
			local_sum[local_id] += local_sum[local_id + i];
		}
	}
	
	if(local_id == 0){
		partial_sum[get_group_id(0)] = local_sum[0];
	}
}
