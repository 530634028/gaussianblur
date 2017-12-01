#include "../gaussian.hh"

int main()
{
	GaussCache gcache_cpu(2);
	std::cout << "kernel (cpu):" << std::endl;
	for (int i = 0; i < gcache_cpu.kw; ++i)
		std::cout << gcache_cpu.kernel_buf.get()[i] << ", ";
	std::cout << std::endl;

	GaussCacheFull gcache_gpu(2);
	std::cout << "kernel (gpu):" << std::endl;
	for (int i = 0; i < gcache_gpu.kw; ++i)
	{
		for (int j = 0; j < gcache_gpu.kw; ++j)
			std::cout << gcache_gpu.kernel_buf.get()[i*gcache_gpu.kw+j] << ", ";
		std::cout << std::endl;	
	}
}
