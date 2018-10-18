__kernel void
local_heat_diffusion(
	__global const long * h,
	const float c,
	const unsigned int width,
	const unsigned int height,
	__global long * h_updated
)
{
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
	
