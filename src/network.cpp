#include "network.h"
#include "weight_definitions.h"
#include "tanh.h"

l_quantized_type ReLU(l_quantized_type res)
{
	if (res < 0)
		return 0;

	return res;
}

l_quantized_type tanh(l_quantized_type res)
{
	if (res >= 2)
		return 1;
	else if (res < -2)
		return -1;
	else
	{
		ap_int <BITS+2> i = res.range();			//prepare result to match tanh value
		return tanh_vals[(BITS_EXP/2) + i.to_int()];
	}
}

extern "C" {
void network(float *x, float *y, int INPUT_IMAGES)
{

	#pragma HLS INTERFACE m_axi port = x offset = slave bundle = gmem0
	#pragma HLS INTERFACE m_axi port = y offset = slave bundle = gmem1
	#pragma HLS INTERFACE s_axilite port = x
	#pragma HLS INTERFACE s_axilite port = y
	#pragma HLS INTERFACE s_axilite port = INPUT_IMAGES
	#pragma HLS INTERFACE s_axilite port = return

	quantized_type xbuf[N1];
	l_quantized_type layer_1_out[M1];
	l_quantized_type layer_2_out[M2];


	//partition memory to create parallel read/write ports
	#pragma HLS array_partition variable=layer_1_out block factor=15 dim=1
	#pragma HLS array_partition variable=layer_2_out block factor=25 dim=1
	#pragma HLS array_partition variable=W1 block factor=15 dim=2
	#pragma HLS array_partition variable=W2 block factor=15 dim=1
	#pragma HLS array_partition variable=W3 block factor=25 dim=1

	//limit resources to fit in edge SoC (only for single kernel)
	//#pragma HLS ALLOCATION instances=mul limit=80 operation
	
	//forward propagation for multiple images
	for(int iter = 0; iter<INPUT_IMAGES; iter++){

		read_input:
		for (int i=0; i<N1; i++)
		{
			#pragma HLS PIPELINE II=1
			xbuf[i] = x[iter*N1+i];
		}

		// Layer 1
		layer_1:
		for(int i=0; i<N1; i++)
		{
			#pragma HLS PIPELINE II=1

			for(int j=0; j<M1; j++)
			{
				#pragma HLS unroll factor=30
				l_quantized_type last = (i==0) ? (l_quantized_type) 0 : layer_1_out[j];
				quantized_type term = xbuf[i] * W1[i][j];
				layer_1_out[j] = last + term;
			}
		}
		layer_1_act:
		for(int i=0; i<M1; i++)
		{
			#pragma HLS unroll factor = 30
			layer_1_out[i] = ReLU(layer_1_out[i]);
		}

		// Layer 2
		layer_2:
		for(int i=0; i<M2; i++)
		{
			#pragma HLS PIPELINE II=1

			l_quantized_type result = 0;
			for(int j=0; j<N2; j++)
			{
				#pragma HLS unroll factor=30
				l_quantized_type term = layer_1_out[j] * W2[j][i];
				result += term;
			}
			layer_2_out[i] = ReLU(result);
		}

		// Layer 3
		layer_3:
		for(int i=0; i<M3; i++)
		{
			#pragma HLS PIPELINE II=1

			l_quantized_type result = 0;
			for(int j=0; j<N3; j++)
			{
				#pragma HLS unroll factor=50
				l_quantized_type term = layer_2_out[j] * W3[j][i];
				result += term;
			}
			y[iter*M3+i] = tanh(result).to_float();
		}
	}
}
}
